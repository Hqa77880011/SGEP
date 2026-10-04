"""Data preparation from source labels and explicitly supplied group linkage."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .data import read_manifest, relative_file


RESOURCES = Path(__file__).parent / "resources"
MANIFEST_FIELDS = ["dataset", "release", "image_id", "relative_path", "category",
                   "group_id", "group_type", "source_cohort", "mask_path"]


def _metadata(metadata):
    frame = read_manifest(metadata).reset_index(drop=True)
    if "mask_path" not in frame:
        frame["mask_path"] = ""
    missing = set(MANIFEST_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError("Missing metadata columns: " + ", ".join(sorted(missing)))
    provenance = [field for field in MANIFEST_FIELDS if field != "mask_path"]
    if frame.empty or frame[provenance].astype(str).apply(lambda c: c.str.strip().eq("")).any().any():
        raise ValueError("Metadata requires genuine, nonempty provenance and group identifiers.")
    if frame.duplicated(["dataset", "image_id"]).any():
        raise ValueError("Duplicate image identifiers must be reconciled before partitioning.")
    if not set(frame.group_type).issubset({"patient", "lesion", "procedure", "sequence"}):
        raise ValueError("group_type must be patient, lesion, procedure or sequence.")
    if frame.groupby(["dataset", "group_id"]).group_type.nunique().gt(1).any():
        raise ValueError("Each provenance group must use one consistent group_type.")
    endoscopy = frame.dataset.isin(["GastroVision", "HyperKvasir"])
    if not frame.loc[endoscopy, "group_type"].isin(["patient", "procedure", "sequence"]).all():
        raise ValueError("Endoscopy requires genuine patient/procedure/sequence linkage; "
                         "public random filenames do not supply it.")
    for column in ("relative_path", "mask_path", "reference_mask_path"):
        if column in frame:
            for value in frame.loc[frame[column].ne(""), column]:
                relative_file(Path.cwd(), value)
    if "reference_mask_path" in frame:
        if (frame.mask_path.ne("") & frame.mask_path.eq(frame.reference_mask_path)).any():
            raise ValueError("Reference masks are evaluation-only; mask_path must name an automatic ROI.")
    return frame


def make_split(metadata, category_roles=None, seed=2026):
    """Allocate whole source groups with final-unknown test precedence.

    The four known partitions target 70/10/5/15 percent of groups. Each class
    stratum receives at least one group per partition. Proxy groups use a 2:1
    selection/calibration split and never update recognition-model weights.
    """
    frame = _metadata(metadata)
    roles = read_manifest(category_roles if category_roles is not None else RESOURCES / "roles.csv")
    frame = frame.drop(columns=[c for c in ("role", "mapped_class", "split") if c in frame])
    frame = frame.merge(roles[["dataset", "category", "role"]], on=["dataset", "category"],
                        how="left", validate="many_to_one")
    if frame.role.isna().any():
        missing = frame.loc[frame.role.isna(), ["dataset", "category"]].drop_duplicates()
        raise ValueError("Missing category role assignments: " + missing.to_json(orient="records"))
    if not set(frame.role).issubset({"known", "proxy", "unknown"}):
        raise ValueError("Source roles must be known, proxy or unknown.")
    frame["mapped_class"] = frame.category
    frame["split"] = ""
    if "exclusion_reason" not in frame:
        frame["exclusion_reason"] = ""
    excluded = frame.exclusion_reason.ne("")
    rng = np.random.default_rng(seed)
    for dataset, dataset_rows in frame.groupby("dataset", sort=True):
        known_groups, proxy_groups = {}, []
        for _, rows in dataset_rows.groupby("group_id", sort=True):
            group_roles = set(rows.role)
            if "unknown" in group_roles:
                frame.loc[rows.index, "split"] = "test"
                proxy_ids = rows.index[rows.role.eq("proxy")]
                frame.loc[proxy_ids, "split"] = "excluded"
                frame.loc[proxy_ids, "exclusion_reason"] = "proxy image linked to final-unknown test group"
            elif "proxy" in group_roles:
                proxy_groups.append(rows.index)
            else:
                stratum = rows.category.value_counts().sort_index().idxmax()
                known_groups.setdefault(stratum, []).append(rows.index)
        if proxy_groups and len(proxy_groups) < 2:
            raise ValueError(f"{dataset}: at least two independent proxy groups are required.")
        proxy_order = rng.permutation(len(proxy_groups))
        n_select = max(1, min(len(proxy_groups) - 1, round(len(proxy_groups) * 2 / 3)))
        for position, index in enumerate(proxy_order):
            frame.loc[proxy_groups[index], "split"] = "selection" if position < n_select else "calibration"
        for category, groups in known_groups.items():
            n_groups = len(groups)
            if n_groups < 4:
                raise ValueError(f"{dataset}/{category}: {n_groups} independent groups "
                                 "cannot populate four partitions.")
            weights = np.array([0.70, 0.10, 0.05, 0.15])
            target = np.maximum(weights * n_groups - 1, 0)
            target = target / target.sum() * (n_groups - 4) if n_groups > 4 else np.zeros(4)
            allocation = np.floor(target).astype(int) + 1
            remainder = n_groups - allocation.sum()
            order = np.argsort(-(target - np.floor(target)), kind="stable")
            allocation[order[:remainder]] += 1
            group_order, cursor = rng.permutation(n_groups), 0
            for split, count in zip(("train", "selection", "calibration", "test"), allocation):
                for index in group_order[cursor:cursor + count]:
                    frame.loc[groups[index], "split"] = split
                cursor += count
    frame.loc[excluded, "split"] = "excluded"
    included = frame.loc[frame.split.ne("excluded")]
    if included.groupby(["dataset", "group_id"]).split.nunique().gt(1).any():
        raise AssertionError("A provenance group crossed partitions.")
    if not frame.loc[frame.role.eq("unknown"), "split"].isin(["test", "excluded"]).all():
        raise AssertionError("A final unknown entered fitting or validation.")
    if not frame.loc[frame.split.eq("train"), "role"].eq("known").all():
        raise AssertionError("A non-known sample entered training.")
    return frame.sort_values(["dataset", "split", "category", "image_id"]).reset_index(drop=True)


def _write(frame, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def prepare_generic(metadata, output, roles=None, seed=2026):
    """Write a source split from normalized metadata with real group linkage."""
    return _write(make_split(metadata, roles, seed), output)


def _image_paths(images_root):
    root = Path(images_root).resolve()
    paths = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            if path.stem in paths:
                raise ValueError(f"Ambiguous image stem {path.stem!r}; use explicit relative_path metadata.")
            paths[path.stem] = path.relative_to(root).as_posix()
    return paths


def prepare_ham(metadata, images_root, output, dataset="HAM10000", roles=None,
                release="HAM10000", source_cohort="HAM10000", seed=2026):
    """Prepare HAM10000 using lesion_id, or complete supplied patient linkage."""
    frame = read_manifest(metadata)
    if not {"image_id", "dx", "lesion_id"}.issubset(frame):
        raise ValueError("HAM metadata requires image_id, dx and lesion_id.")
    frame["dataset"], frame["category"], frame["release"] = dataset, frame.dx, release
    if "source_cohort" not in frame:
        frame["source_cohort"] = source_cohort
    if not {"group_id", "group_type"}.issubset(frame):
        if "patient_id" in frame and frame.patient_id.ne("").any():
            if frame.patient_id.eq("").any():
                raise ValueError("Patient linkage is incomplete; provide reconciled group_id/group_type for every image.")
            frame["group_id"], frame["group_type"] = frame.patient_id, "patient"
        else:
            frame["group_id"], frame["group_type"] = frame.lesion_id, "lesion"
    if "relative_path" not in frame:
        paths = _image_paths(images_root)
        frame["relative_path"] = frame.image_id.map(paths)
        if frame.relative_path.isna().any():
            raise FileNotFoundError("HAM image files are missing: "
                                    + ", ".join(frame.loc[frame.relative_path.isna(), "image_id"].head(5)))
    for value in frame.relative_path:
        if not relative_file(images_root, value).is_file():
            raise FileNotFoundError(f"Missing source image: {value}")
    return prepare_generic(frame, output, roles, seed)


def prepare_gastro(metadata, images_root, linkage, output, roles=None,
                   release="GastroVision", seed=2026):
    """Join GastroVision Filename/Class metadata to provider group linkage."""
    frame, linkage = read_manifest(metadata), read_manifest(linkage)
    if not {"Filename", "Class"}.issubset(frame):
        raise ValueError("GastroVision release metadata requires Filename and Class.")
    required = {"image_id", "group_id", "group_type", "source_cohort"}
    if not required.issubset(linkage):
        raise ValueError("Linkage requires image_id, group_id, group_type and source_cohort.")
    frame["image_id"] = frame.Filename.map(lambda value: Path(value).stem)
    # Linkage image_id accepts the released filename with or without extension.
    linkage["image_id"] = linkage.image_id.map(lambda value: Path(value).stem)
    frame = frame.merge(linkage, on="image_id", how="left", validate="one_to_one")
    if frame[list(required - {"image_id"})].isna().any().any():
        raise ValueError("Every GastroVision image requires genuine linked group metadata.")
    frame["dataset"], frame["category"], frame["release"] = "GastroVision", frame.Class, release
    if "relative_path" not in frame:
        paths = _image_paths(images_root)
        frame["relative_path"] = frame.image_id.map(paths)
        if frame.relative_path.isna().any():
            raise FileNotFoundError("Some GastroVision image files are missing.")
    for value in frame.relative_path:
        if not relative_file(images_root, value).is_file():
            raise FileNotFoundError(f"Missing source image: {value}")
    return prepare_generic(frame, output, roles, seed)


def prepare_external(metadata, output, mapping=None):
    """Apply the fixed external taxonomy and preserve provenance exclusions."""
    frame = _metadata(metadata)
    crosswalk = read_manifest(mapping if mapping is not None else RESOURCES / "external_mapping.csv")
    exclusions = frame.get("exclusion_reason", pd.Series("", index=frame.index))
    frame = frame.drop(columns=[c for c in ("role", "mapped_class", "split", "exclusion_reason") if c in frame])
    frame = frame.merge(crosswalk, on=["dataset", "category"], how="left", validate="many_to_one")
    if frame.role.isna().any():
        raise ValueError("Every external category must have an explicit taxonomy mapping.")
    frame["exclusion_reason"] = frame.exclusion_reason.fillna("")
    frame.loc[exclusions.ne(""), "exclusion_reason"] = exclusions.loc[exclusions.ne("")]
    frame["split"] = "test"
    frame.loc[frame.role.eq("excluded") | frame.exclusion_reason.ne(""), "split"] = "excluded"
    if frame.loc[frame.split.eq("test"), "role"].eq("proxy").any():
        raise ValueError("External test categories must not map to source proxy roles.")
    return _write(frame.fillna(""), output)


def select_sam_mask(candidates, shape):
    """Select SAM's fixed area/quality candidate, or a counted full-image ROI."""
    total = int(np.prod(shape))
    eligible = [(index, item) for index, item in enumerate(candidates)
                if 0.05 <= item["area"] / total <= 0.80]
    if not eligible:
        return np.ones(shape, dtype=bool), True, -1
    index, item = min(eligible, key=lambda pair: (
        -pair[1]["predicted_iou"] * pair[1]["stability_score"], pair[1]["area"], pair[0]))
    return np.asarray(item["segmentation"], dtype=bool), False, index


def generate_masks(manifest, root, checkpoint, device="cpu", masks_dir="masks/sam_vit_b", output=None):
    """Generate frozen SAM ViT-B masks and per-image selection/fallback logs."""
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Missing SAM ViT-B checkpoint: {checkpoint}")
    frame = _metadata(manifest)
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    sam.eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    sam.mask_threshold = 0.0
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=32, points_per_batch=64,
        pred_iou_thresh=0.88, stability_score_thresh=0.95,
        stability_score_offset=1.0, box_nms_thresh=0.7, crop_n_layers=0,
        crop_nms_thresh=0.7, crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1, min_mask_region_area=0,
    )
    logs = []
    for index, row in frame.iterrows():
        if row.get("split", "") == "excluded":
            continue
        with Image.open(relative_file(root, row.relative_path)) as source:
            rgb = np.array(source.convert("RGB"))
        with torch.inference_mode():
            candidates = generator.generate(rgb)
        mask, fallback, candidate_index = select_sam_mask(candidates, rgb.shape[:2])
        relative = (Path(masks_dir) / row.dataset / (row.image_id + ".png")).as_posix()
        target = relative_file(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(target)
        frame.at[index, "mask_path"] = relative
        selected = candidates[candidate_index] if candidate_index >= 0 else {}
        logs.append({"dataset": row.dataset, "image_id": row.image_id,
                     "candidates": len(candidates), "selected_candidate": candidate_index,
                     "foreground_fraction": float(mask.mean()), "full_image_fallback": fallback,
                     "predicted_iou": selected.get("predicted_iou", float("nan")),
                     "stability_score": selected.get("stability_score", float("nan"))})
    logs = pd.DataFrame(logs)
    if output is not None:
        _write(frame, output)
        _write(logs, Path(output).with_name(Path(output).stem + "_mask_log.csv"))
    return frame, logs


def evaluate_annotation_quality(annotation_csv, manifest, root, output):
    """Evaluate native-resolution automatic and independently annotated masks.

    Annotation rows require dataset, image_id, annotator_a_path,
    annotator_b_path, reference_mask_path and adjudicated (true/false or 1/0).
    Human masks are evaluation-only. The companion *_manifest.csv retains all
    original partitions while adding references for annotated test images.
    """
    annotations = read_manifest(annotation_csv).reset_index(drop=True)
    required = {"dataset", "image_id", "annotator_a_path", "annotator_b_path",
                "reference_mask_path", "adjudicated"}
    missing = required - set(annotations)
    if missing:
        raise ValueError("Missing annotation columns: " + ", ".join(sorted(missing)))
    if annotations.empty or annotations[list(required)].astype(str).apply(
            lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Every selected annotation requires both annotators, a reference and adjudication status.")
    if annotations.duplicated(["dataset", "image_id"]).any():
        raise ValueError("Annotation image identifiers must be unique.")
    manifest = _metadata(manifest)
    joined = annotations.merge(
        manifest.drop(columns=[c for c in required - {"dataset", "image_id"} if c in manifest]),
        on=["dataset", "image_id"], how="left", validate="one_to_one", indicator=True)
    if joined._merge.ne("both").any():
        raise ValueError("Every annotated image must occur in the supplied manifest.")
    joined = joined.drop(columns="_merge")
    if not joined.split.eq("test").all():
        raise ValueError("Mask-quality references must come from held-out test images.")
    values = joined.adjudicated.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if values.isna().any():
        raise ValueError("adjudicated must be true/false or 1/0.")
    joined["adjudicated"] = values
    measurements = []
    for _, row in joined.iterrows():
        human_paths = [row.annotator_a_path, row.annotator_b_path, row.reference_mask_path]
        if row.mask_path in human_paths:
            raise ValueError(f"Human annotation cannot be used as the automatic ROI for {row.image_id}.")
        masks = []
        for path in [row.mask_path] + human_paths:
            with Image.open(relative_file(root, path)) as source:
                masks.append(np.asarray(source.convert("L")) > 0)
        with Image.open(relative_file(root, row.relative_path)) as source:
            image_shape = (source.height, source.width)
        if any(mask.shape != image_shape for mask in masks):
            raise ValueError(f"All native mask dimensions must match the image for {row.image_id}.")
        automatic, a, b, reference = masks
        if not np.array_equal(a, b) and not row.adjudicated:
            raise ValueError(f"Annotator disagreement needs adjudication for {row.image_id}.")

        def dice(left, right):
            denominator = int(left.sum()) + int(right.sum())
            return 2 * int(np.logical_and(left, right).sum()) / denominator if denominator else float("nan")

        measurements.append({"dice": dice(automatic, reference),
                             "annotator_dice": dice(a, b),
                             "automatic_foreground_pixels": int(automatic.sum()),
                             "reference_foreground_pixels": int(reference.sum()),
                             "annotator_a_foreground_pixels": int(a.sum()),
                             "annotator_b_foreground_pixels": int(b.sum()),
                             "both_empty_quality": not automatic.any() and not reference.any(),
                             "both_empty_annotators": not a.any() and not b.any()})
    quality = pd.concat([joined, pd.DataFrame(measurements)], axis=1)
    _write(quality, output)
    paths = annotations[["dataset", "image_id", "reference_mask_path"]].rename(
        columns={"reference_mask_path": "annotated_reference_mask_path"})
    updated = manifest.merge(paths, on=["dataset", "image_id"], how="left", validate="one_to_one")
    if "reference_mask_path" not in updated:
        updated["reference_mask_path"] = ""
    annotated = updated.annotated_reference_mask_path.notna()
    updated.loc[annotated, "reference_mask_path"] = updated.loc[annotated, "annotated_reference_mask_path"]
    updated = updated.drop(columns="annotated_reference_mask_path").fillna("")
    _metadata(updated)
    _write(updated, Path(output).with_name(Path(output).stem + "_manifest.csv"))
    return quality
