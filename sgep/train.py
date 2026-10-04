"""Known-only training, model-selection evaluation, and checkpoint inference."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baselines import PostprocessorFitError, apply_postprocessor, fit_postprocessor
from .config import epoch_lr
from .data import ImageDataset, perturb_masks
from .losses import compute_loss
from .metrics import evaluate_predictions
from .models import build_model, initialize_prototypes

POSTPROCESSORS = {"openmax", "postmax", "cac"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(config):
    if config.get("threads", 0):
        torch.set_num_threads(config["threads"])
    device = config.get("device", "auto")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def read_manifest(path, dataset=None):
    """Select one dataset and validate the training/validation semantic boundary."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"dataset", "image_id", "relative_path", "mask_path", "category",
                "mapped_class", "role", "split", "group_id", "group_type"}
    if not required.issubset(frame):
        raise ValueError(f"Manifest lacks columns: {sorted(required - set(frame))}")
    if dataset is not None:
        frame = frame[frame.dataset.eq(dataset)].reset_index(drop=True)
    if frame.empty or frame.dataset.nunique() != 1:
        raise ValueError("Select exactly one nonempty dataset with --dataset.")
    active = frame[~frame.split.eq("excluded")]
    if active.group_id.eq("").any() or active.group_type.eq("").any():
        raise ValueError("Real group_id and group_type values are required.")
    if (active.groupby("group_id").split.nunique() > 1).any():
        raise ValueError("A group crosses dataset partitions.")
    if not frame.loc[frame.split.eq("train"), "role"].eq("known").all():
        raise ValueError("Only known images may enter training.")
    if not frame.loc[frame.role.eq("unknown"), "split"].isin(["test", "excluded"]).all():
        raise ValueError("Final unknowns are restricted to test/excluded.")
    if frame.loc[frame.role.eq("proxy"), "split"].isin(["train", "test"]).any():
        raise ValueError("Proxy categories are restricted to selection/calibration.")
    if frame.duplicated(["dataset", "image_id"]).any():
        raise ValueError("Duplicate image IDs in manifest.")
    # A semantic category cannot acquire different roles in different partitions.
    if (active.groupby("category").role.nunique() > 1).any():
        raise ValueError("Category roles must be fixed before splitting.")
    return frame


def make_loader(frame, root, split, classes, config, augment=False, shuffle=False):
    dataset = ImageDataset(
        frame, root, split, classes, image_size=config["image_size"],
        augment=augment, mask_policy=config["mask_policy"],
        fill=config["fill"], mask_seed=config["mask_seed"],
    )
    if not len(dataset):
        raise ValueError(f"The {split} partition is empty.")
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=config["workers"], generator=generator)


def masked_rgb(images, masks, fill):
    fill_value = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) if fill == "mean" else 0.0
    return images * masks + fill_value * (1 - masks)


@torch.no_grad()
def predict_loader(model, loader, device, config, postprocessor=None, diagnostics=False,
                   intervention="original"):
    """Return individual predictions; all loaders here preserve manifest order."""
    model.eval()
    arrays = {}
    for batch in loader:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        if intervention in {"translation", "erosion", "dilation", "jitter"}:
            mode = None if intervention == "jitter" else intervention
            masks, _ = perturb_masks(masks, mode=mode)
        elif intervention == "dropout":
            original = masks
            masks = masks * (torch.rand_like(masks) >= 0.2)
            empty = masks.flatten(1).sum(1).eq(0)
            masks[empty] = original[empty]
        elif intervention == "background":
            masks = 1 - masks
            images = masked_rgb(images, masks, config["fill"])
        elif intervention != "original":
            raise ValueError(f"Unknown intervention: {intervention}")
        output = model(images, masks)
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                arrays.setdefault(key, []).append(value.detach().cpu().numpy())
        arrays.setdefault("labels", []).append(batch["label"].numpy())
        if "dice" in batch:
            arrays.setdefault("mask_dice", []).append(batch["dice"].numpy())
        if diagnostics and "evidence" in output:
            complement = 1 - masks
            bg = model(masked_rgb(images, complement, config["fill"]), complement)
            evidence, background = output["evidence"], bg["evidence"]
            total, bg_total = evidence.sum(1), background.sum(1)
            top = evidence.topk(2, dim=1).values
            arrays.setdefault("background_evidence", []).append(background.cpu().numpy())
            arrays.setdefault("rho", []).append((total / (total + bg_total + 1e-7)).cpu().numpy())
            arrays.setdefault("delta_e", []).append(((top[:, 0] - top[:, 1]) / (total + 1e-7)).cpu().numpy())
    result = {key: np.concatenate(values) for key, values in arrays.items()}
    for key in ("image_id", "group_id", "group_type", "category", "role", "dataset",
                "release", "source_cohort", "mapped_class"):
        if key in loader.dataset.frame:
            result[key] = loader.dataset.frame[key].to_numpy(dtype=str)
    if postprocessor is not None:
        result.update(apply_postprocessor(config["method"], result, postprocessor, config))
    result["prediction"] = result["probabilities"].argmax(1)
    return result


def serializable_state(value):
    """Keep post-hoc state compatible with safe torch weights-only loading."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable_state(item) for item in value]
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(serializable_state(value), indent=2, allow_nan=False) + "\n", "utf-8")


def train(config, manifest, root, output, dataset=None):
    """Train with known labels and retain the highest selection-OSCR checkpoint."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists():
        raise FileExistsError(f"{output}/checkpoint.pt exists; use a new run directory.")
    frame = read_manifest(manifest, dataset)
    classes = sorted(frame.loc[frame.role.eq("known"), "mapped_class"].unique())
    if len(classes) < 2:
        raise ValueError("At least two known classes are required.")
    for split in ("selection", "calibration"):
        roles = set(frame.loc[frame.split.eq(split), "role"])
        if roles != {"known", "proxy"}:
            raise ValueError(f"{split} must contain known and class-disjoint proxy images.")
    seed_everything(config["seed"])
    device = device_for(config)
    training = make_loader(frame, root, "train", classes, config, augment=True, shuffle=True)
    initial = make_loader(frame, root, "train", classes, config)
    selection = make_loader(frame, root, "selection", classes, config)
    model = build_model(config, len(classes)).to(device)
    initialize_prototypes(model, initial, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    save_json(output / "config.json", config)
    history = []
    best = -float("inf")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        lr = epoch_lr(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals, n, fallback_count = {}, 0, 0
        for batch in training:
            images, masks, labels = batch["image"].to(device), batch["mask"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = compute_loss(model, model(images, masks), images, masks, labels, config, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            n += len(labels)
            for key, value in components.items():
                if key == "mask_fallback_count":
                    fallback_count += int(value)
                    continue
                totals[key] = totals.get(key, 0.0) + float(value) * len(labels)
        postprocessor = None
        if config["method"] in POSTPROCESSORS:
            predictions = predict_loader(model, initial, device, config)
            try:
                postprocessor = fit_postprocessor(config["method"], predictions["logits"],
                                                  predictions["features"], predictions["labels"], config)
            except PostprocessorFitError as error:
                history.append({"epoch": epoch, "lr": lr, "mask_fallback_count": fallback_count,
                                **{k: v / n for k, v in totals.items()},
                                "selection_oscr": None, "fit_error": str(error)})
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                print(f"epoch {epoch}: post-hoc fit unavailable: {error}")
                continue
        predictions = predict_loader(model, selection, device, config, postprocessor)
        metrics = evaluate_predictions(predictions, threshold=None)
        record = {"epoch": epoch, "lr": lr, "mask_fallback_count": fallback_count,
                  **{k: v / n for k, v in totals.items()},
                  "selection_oscr": metrics["oscr"]}
        history.append(record)
        print(f"epoch {epoch}/{config['epochs']} loss={record['loss']:.4f} selection_oscr={metrics['oscr']:.3f}")
        if metrics["oscr"] > best:
            best = metrics["oscr"]
            torch.save({"state_dict": model.state_dict(), "config": config, "classes": classes,
                        "dataset": str(frame.dataset.iloc[0]), "epoch": epoch,
                        "selection_oscr": best, "postprocessor": serializable_state(postprocessor)},
                       output / "checkpoint.pt")
            np.savez_compressed(output / "selection_predictions.npz", **predictions)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    if not (output / "checkpoint.pt").is_file():
        raise RuntimeError("No epoch had enough correctly classified training samples for post-hoc fitting; inspect history.csv.")
    return output / "checkpoint.pt"


def predict(checkpoint, manifest, root, split="test", dataset=None, diagnostics=True,
            intervention="original", mask_policy=None, fill=None, device="auto", batch_size=None):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"].copy()
    config.update({"pretrained": False, "device": device})
    if mask_policy is not None:
        config["mask_policy"] = mask_policy
    if fill is not None:
        config["fill"] = fill
    if batch_size is not None:
        config["batch_size"] = batch_size
    seed_everything(config["seed"])
    device = device_for(config)
    model = build_model(config, len(checkpoint["classes"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    frame = read_manifest(manifest, dataset)
    loader = make_loader(frame, root, split, checkpoint["classes"], config)
    return predict_loader(model, loader, device, config, checkpoint["postprocessor"],
                          diagnostics, intervention)
