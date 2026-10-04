"""Command-line workflow from public metadata to experiment analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config


def threshold_value(path):
    """Read a saved calibration threshold or an explicitly supplied number."""
    if Path(str(path)).is_file():
        return float(json.loads(Path(path).read_text("utf-8"))["threshold"])
    return float(path)


def data_arguments(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Root for all manifest-relative paths")
    parser.add_argument("--dataset", help="Dataset to select if the manifest contains several")


def run_arguments(parser):
    data_arguments(parser)
    parser.add_argument("--config", type=Path, default=Path("configs/sgep.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")


def checkpoint_arguments(parser):
    data_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)


def parser_for_cli():
    parser = argparse.ArgumentParser(description="SGEP implementation for open-set medical image recognition")
    commands = parser.add_subparsers(dest="command", required=True)
    ham = commands.add_parser("prepare-ham", help="Prepare grouped HAM10000 partitions")
    ham.add_argument("--metadata", type=Path, required=True)
    ham.add_argument("--images", type=Path, required=True)
    ham.add_argument("--output", type=Path, required=True)
    gastro = commands.add_parser("prepare-gastro", help="Join GastroVision labels to real group linkage")
    gastro.add_argument("--metadata", type=Path, required=True)
    gastro.add_argument("--images", type=Path, required=True)
    gastro.add_argument("--linkage", type=Path, required=True)
    gastro.add_argument("--output", type=Path, required=True)
    generic = commands.add_parser("prepare", help="Split normalized source metadata by genuine groups")
    generic.add_argument("--metadata", type=Path, required=True)
    generic.add_argument("--roles", type=Path)
    generic.add_argument("--output", type=Path, required=True)
    external = commands.add_parser("prepare-external", help="Apply the fixed PH2/HyperKvasir semantic crosswalk")
    external.add_argument("--metadata", type=Path, required=True)
    external.add_argument("--mapping", type=Path)
    external.add_argument("--output", type=Path, required=True)
    masks = commands.add_parser("masks", help="Precompute the fixed automatic SAM ViT-B prior")
    data_arguments(masks)
    masks.add_argument("--checkpoint", type=Path, required=True)
    masks.add_argument("--device", default="auto")
    masks.add_argument("--masks-dir", default="masks/sam_vit_b")
    masks.add_argument("--output", type=Path, required=True)
    overlap = commands.add_parser("audit-external", help="Check external/source image identity and supplied group linkage")
    for name in ("source", "external", "source-root", "external-root", "output"):
        overlap.add_argument("--" + name, type=Path, required=True)
    overlap.add_argument("--near-distance", type=int, default=6, help="DCT signature Hamming cutoff; candidates are excluded")
    annotation = commands.add_parser("select-reference", help="Select HAM test images for independent reference annotation")
    annotation.add_argument("--manifest", type=Path, required=True)
    annotation.add_argument("--output", type=Path, required=True)
    annotation.add_argument("--per-class", type=int, default=100)
    quality = commands.add_parser("mask-quality", help="Score automatic and independently annotated masks")
    data_arguments(quality)
    quality.add_argument("--annotations", type=Path, required=True)
    quality.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train", help="Train known classes; choose a checkpoint by selection OSCR")
    run_arguments(training)
    training.add_argument("--seed", type=int, default=11)
    training.add_argument("--method", help="Override the configured recognition method")
    calibration = commands.add_parser("calibrate", help="Fit an operating threshold on the reserved calibration partition")
    checkpoint_arguments(calibration)
    calibration.add_argument("--rule", choices=["coverage90", "coverage95", "coverage97", "youden"], default="coverage95")
    evaluation = commands.add_parser("evaluate", help="Execute the fixed checkpoint and evaluate individual predictions")
    checkpoint_arguments(evaluation)
    evaluation.add_argument("--split", choices=["test", "selection", "calibration"], default="test")
    evaluation.add_argument("--threshold", required=True, help="Source threshold.json or an explicit numeric threshold")
    evaluation.add_argument("--intervention", choices=["original", "translation", "jitter", "erosion", "dilation", "dropout", "background"], default="original")
    evaluation.add_argument("--mask-policy", choices=["aligned", "random_area", "displaced"])
    evaluation.add_argument("--fill", choices=["black", "mean"])
    analysis = commands.add_parser("analyze", help="Produce spatial, calibration, difficulty and threshold diagnostics")
    analysis.add_argument("--predictions", type=Path, required=True)
    analysis.add_argument("--calibration", type=Path)
    analysis.add_argument("--threshold", required=True)
    analysis.add_argument("--near-category")
    analysis.add_argument("--mask-quality", type=Path)
    analysis.add_argument("--resamples", type=int, default=2000)
    analysis.add_argument("--output", type=Path, required=True)
    suite = commands.add_parser("suite", help="Run paired baselines, ablations, factorial and spatial controls")
    run_arguments(suite)
    suite.add_argument("--suite", choices=["main", "ablation", "factorial", "spatial", "all"], default="main")
    suite.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    suite.add_argument("--plan", action="store_true", help="Write the run plan without training")
    search = commands.add_parser("search", help="Select hyperparameters using selection data only")
    run_arguments(search)
    search.add_argument("--method", default="sgep")
    search.add_argument("--count", type=int, default=24)
    search.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    search.add_argument("--plan", action="store_true")
    compare = commands.add_parser("compare", help="Aggregate completed runs and paired-seed comparisons")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--baseline", default="roi_guided")
    compare.add_argument("--proposed", default="sgep")
    compare.add_argument("--output", type=Path, required=True)
    bootstrap = commands.add_parser("bootstrap", help="Paired cluster intervals on actual per-image predictions")
    bootstrap.add_argument("--baseline", type=Path, required=True)
    bootstrap.add_argument("--proposed", type=Path, required=True)
    bootstrap.add_argument("--baseline-threshold")
    bootstrap.add_argument("--proposed-threshold")
    bootstrap.add_argument("--resamples", type=int, default=2000)
    bootstrap.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("smoke", help="Check the complete CPU workflow on generated test fixtures")
    smoke.add_argument("--output", type=Path, help="Optional directory; default fixtures are temporary")
    return parser


def config_from_args(args):
    overrides = {name: getattr(args, name) for name in ("seed", "device")
                 if getattr(args, name, None) is not None}
    config = load_config(args.config, overrides)
    if getattr(args, "method", None):
        from .experiments import baseline_settings
        config.update(baseline_settings(args.method))
    return config


def write_prediction_outputs(predictions, checkpoint, output, threshold, intervention="original"):
    """Store machine-readable scores and human-readable open-set decisions."""
    from .metrics import evaluate_predictions
    from .train import save_json
    from .visualize import plot_predictions

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", **predictions)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    accepted = predictions["uncertainty"] <= threshold
    labels = predictions["probabilities"].argmax(1)
    classes = np.asarray(state["classes"])
    table = {key: predictions[key] for key in ("image_id", "group_id", "category", "role")}
    table.update(uncertainty=predictions["uncertainty"], predicted_class=classes[labels],
                 accepted=accepted, decision=np.where(accepted, classes[labels], "unknown"))
    for index, name in enumerate(classes):
        table["p_" + name] = predictions["probabilities"][:, index]
    pd.DataFrame(table).to_csv(output / "predictions.csv", index=False)
    metrics = evaluate_predictions(predictions, threshold)
    save_json(output / "metrics.json", {"dataset": str(predictions["dataset"][0]),
               "method": state["config"]["method"], "seed": state["config"]["seed"],
               "intervention": intervention, "metrics": metrics})
    plot_predictions(predictions, output, threshold)
    print(json.dumps(metrics, indent=2, allow_nan=False))


def main(argv=None):
    args = parser_for_cli().parse_args(argv)
    command = args.command
    if command.startswith("prepare") or command == "masks" or command == "mask-quality":
        from . import prepare
        if command == "prepare-ham":
            prepare.prepare_ham(args.metadata, args.images, args.output)
        elif command == "prepare-gastro":
            prepare.prepare_gastro(args.metadata, args.images, args.linkage, args.output)
        elif command == "prepare":
            prepare.prepare_generic(args.metadata, args.output, args.roles)
        elif command == "prepare-external":
            prepare.prepare_external(args.metadata, args.output, args.mapping)
        elif command == "mask-quality":
            prepare.evaluate_annotation_quality(args.annotations, args.manifest, args.root, args.output)
        else:
            from .train import device_for
            frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
            if args.dataset:
                frame = frame[frame.dataset.eq(args.dataset)]
            _, log = prepare.generate_masks(frame, args.root, args.checkpoint,
                         str(device_for({"device": args.device})), args.masks_dir, args.output)
            print(log.to_string(index=False))
    elif command == "audit-external":
        from .overlap import audit_external
        audit_external(args.source, args.external, args.source_root, args.external_root,
                       args.output, args.near_distance)
    elif command == "select-reference":
        from .overlap import select_reference_images
        frame = select_reference_images(args.manifest, args.output, args.per_class)
        print(frame.groupby("category").agg(images=("image_id", "size"), groups=("group_id", "nunique")))
    elif command == "train":
        from .train import train
        print(train(config_from_args(args), args.manifest, args.root, args.output, args.dataset))
    elif command in {"suite", "search"}:
        from .experiments import run_search, run_suite
        config = config_from_args(args)
        if command == "suite":
            run_suite(config, args.manifest, args.root, args.output, args.suite, args.seeds, args.dataset, args.plan)
        else:
            run_search(config, args.manifest, args.root, args.output, args.method, args.count, args.seeds, args.dataset, args.plan)
    elif command in {"calibrate", "evaluate"}:
        from .train import predict, save_json
        from .metrics import calibrate_threshold
        if command == "calibrate":
            predictions = predict(args.checkpoint, args.manifest, args.root, "calibration", args.dataset, device=args.device)
            threshold = calibrate_threshold(predictions, args.rule)
            args.output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output / "calibration_predictions.npz", **predictions)
            save_json(args.output / "threshold.json", {"rule": args.rule, "threshold": threshold})
            print(f"{args.rule}: threshold={threshold:.8f}")
        else:
            predictions = predict(args.checkpoint, args.manifest, args.root, args.split, args.dataset,
                                  intervention=args.intervention, mask_policy=args.mask_policy,
                                  fill=args.fill, device=args.device)
            write_prediction_outputs(predictions, args.checkpoint, args.output,
                                     threshold_value(args.threshold), args.intervention)
    elif command == "analyze":
        from .analysis import analyze_predictions
        analyze_predictions(args.predictions, threshold_value(args.threshold), args.output,
                            args.calibration, args.near_category, args.mask_quality, args.resamples)
    elif command == "compare":
        from .analysis import aggregate_runs
        candidates = {path for root in args.runs for path in root.rglob("metrics.json")}
        runs = sorted(path.parent for path in candidates
                      if {"dataset", "method", "seed"}.issubset(json.loads(path.read_text("utf-8"))))
        aggregate_runs(runs, args.output, args.baseline, args.proposed)
    elif command == "bootstrap":
        from .analysis import load_predictions, paired_cluster_bootstrap
        from .train import save_json
        thresholds = tuple(threshold_value(path) if path else None
                           for path in (args.baseline_threshold, args.proposed_threshold))
        result = paired_cluster_bootstrap(load_predictions(args.baseline), load_predictions(args.proposed),
                                         thresholds, n_resamples=args.resamples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, result)
    elif command == "smoke":
        from .smoke import run_smoke
        run_smoke(args.output)
