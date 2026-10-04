"""A small synthetic integration fixture, separate from benchmark experiments."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .config import load_config
from .metrics import calibrate_threshold, evaluate_predictions
from .prepare import prepare_generic
from .train import (device_for, make_loader, predict, read_manifest, save_json,
                    seed_everything, train)
from .models import build_model, initialize_prototypes


def create_fixture(root):
    """Make distinct known/proxy/unknown groups with two image views each."""
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    records = []
    rng = np.random.default_rng(2026)
    for category, role, group_count in (("a", "known", 6), ("b", "known", 6),
                                         ("c", "proxy", 3), ("d", "unknown", 2)):
        for group in range(group_count):
            for view in range(2):
                image_id = f"{category}_{group}_{view}"
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[6:26, 7:25] = 255
                image = rng.integers(0, 35, (32, 32, 3), dtype=np.uint8)
                channel = (ord(category) - ord("a")) % 3
                image[mask > 0, channel] = 180 + group * 5
                Image.fromarray(image).save(root / "images" / f"{image_id}.png")
                Image.fromarray(mask).save(root / "masks" / f"{image_id}.png")
                records.append({"dataset": "Fixture", "release": "generated-integration-fixture",
                    "image_id": image_id, "relative_path": f"images/{image_id}.png",
                    "mask_path": f"masks/{image_id}.png", "category": category,
                    "group_id": f"{category}_{group}", "group_type": "lesion",
                    "source_cohort": "generated-fixture"})
    metadata = pd.DataFrame(records)
    roles = pd.DataFrame([{"dataset": "Fixture", "category": category, "role": role}
                          for category, role in (("a", "known"), ("b", "known"),
                                                  ("c", "proxy"), ("d", "unknown"))])
    manifest = root / "manifest.csv"
    prepare_generic(metadata, manifest, roles)
    return manifest


def check_pipeline(output):
    """Assert optimizer, safe reload, calibration, score geometry and decisions."""
    output = Path(output)
    root = output / "fixture"
    manifest = create_fixture(root)
    config = load_config(overrides={"backbone": "tiny", "pretrained": False, "feature_dim": 16,
        "image_size": 32, "epochs": 1, "batch_size": 4, "threads": 1, "device": "cpu"})
    frame = read_manifest(manifest)
    classes = ["a", "b"]
    assert frame.groupby("group_id").split.nunique().eq(1).all()
    assert frame.loc[frame.split.eq("train"), "role"].eq("known").all()
    seed_everything(config["seed"])
    device = device_for(config)
    initial = build_model(config, 2).to(device)
    initialize_prototypes(initial, make_loader(frame, root, "train", classes, config), device)
    initial_weight = initial.global_encoder[0].weight.detach().clone()
    checkpoint = train(config, manifest, root, output / "sgep")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    update = float((state["state_dict"]["global_encoder.0.weight"] - initial_weight).abs().max())
    assert update > 0, "The optimizer must update the recognition encoder."
    calibration = predict(checkpoint, manifest, root, "calibration", device="cpu")
    threshold = calibrate_threshold(calibration, "coverage95")
    known = calibration["labels"] >= 0
    assert np.mean(calibration["uncertainty"][known] <= threshold) >= 0.95
    test = predict(checkpoint, manifest, root, "test", device="cpu")
    np.testing.assert_allclose(test["uncertainty"], 2 / (test["evidence"].sum(1) + 2), rtol=1e-6)
    np.testing.assert_allclose(test["probabilities"].sum(1), 1, rtol=1e-6)
    np.testing.assert_allclose(test["rho"], test["evidence"].sum(1) /
        (test["evidence"].sum(1) + test["background_evidence"].sum(1) + 1e-7), rtol=1e-6)
    second = predict(checkpoint, manifest, root, "test", device="cpu")
    np.testing.assert_array_equal(test["uncertainty"], second["uncertainty"])
    metrics = evaluate_predictions(test, threshold)
    assert metrics["ccr"] <= metrics["known_accuracy"] + 1e-9
    assert metrics["oscr"] <= metrics["known_accuracy"] + 1e-9
    from .cli import main
    main(["calibrate", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
          "--root", str(root), "--device", "cpu", "--output", str(output / "calibration")])
    main(["evaluate", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
          "--root", str(root), "--device", "cpu", "--threshold", str(output / "calibration" / "threshold.json"),
          "--output", str(output / "evaluation")])
    exported = pd.read_csv(output / "evaluation" / "predictions.csv")
    np.testing.assert_array_equal(exported.accepted, test["uncertainty"] <= threshold)
    assert exported.loc[~exported.accepted, "decision"].eq("unknown").all()
    from .analysis import analyze_predictions, paired_cluster_bootstrap
    analyze_predictions(test, threshold, output / "analysis", calibration=calibration,
                        near_category="d", n_resamples=8)
    bootstrap = paired_cluster_bootstrap(test, second, n_resamples=8)
    for result in bootstrap["metrics"].values():
        assert result["difference"] == result["ci_low"] == result["ci_high"] == 0
    baseline = config | {"method": "softmax", "view": "global", "lambda_proto": 0.0}
    baseline_checkpoint = train(baseline, manifest, root, output / "softmax")
    baseline_test = predict(baseline_checkpoint, manifest, root, "test", device="cpu")
    np.testing.assert_allclose(baseline_test["uncertainty"],
                               1 - baseline_test["probabilities"].max(1), atol=1e-7)
    from .experiments import run_search, run_suite
    plans = run_suite(config, manifest, root, output / "plan", suite="all", seeds=[11], plan_only=True)
    assert any(item["config"]["mask_policy"] == "displaced" for item in plans)
    candidates = run_search(config, manifest, root, output / "search_plan", count=2, plan_only=True)
    assert candidates[0]["lambda_roi"] == 0.05
    save_json(output / "validation.json", {"fixture": "synthetic integration check", "encoder_update": update,
              "checks": ["group partitioning", "known-only optimizer update", "safe checkpoint reload",
                         "calibration", "individual decisions", "evidence identities", "paired cluster draws",
                         "analysis plots", "baseline training", "experiment plans"]})
    return {"encoder_update": update, "threshold": threshold, "baseline_score_verified": True}


def run_smoke(output=None):
    if output is None:
        with tempfile.TemporaryDirectory(prefix="sgep-smoke-") as temporary:
            result = check_pipeline(temporary)
    else:
        result = check_pipeline(output)
    print("CPU workflow assertions passed on generated fixtures.")
    return result
