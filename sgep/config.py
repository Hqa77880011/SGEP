"""Recognition and optimization settings shared by command-line workflows."""

from pathlib import Path

import yaml

DEFAULTS = {
    "method": "sgep", "backbone": "resnet18", "pretrained": True,
    "feature_dim": 256, "view": "dual", "fill": "black", "beta_init": 3.0,
    "image_size": 224, "epochs": 100, "batch_size": 32, "workers": 0,
    "lr": 1e-4, "weight_decay": 1e-4, "warmup_epochs": 5,
    "start_lr": 1e-5, "minimum_lr": 1e-6,
    "lambda_proto": 0.1, "lambda_roi": 0.05, "lambda_bg": 0.05,
    "lambda_kl_max": 0.01, "kl_ramp_epochs": 10, "margin": 1.0,
    "mask_policy": "aligned", "mask_seed": 2026,
    "energy_temperature": 1.0, "openmax_tail_size": 20, "openmax_rank": 3,
    "openmax_distance": "eucos",
    "cac_anchor": 10.0, "lambda_cac": 0.1, "arpl_weight": 0.1,
    "arpl_temperature": 1.0,
    "seed": 11, "device": "auto", "threads": 0,
}


def load_config(path=None, overrides=None):
    """Load a flat YAML config, rejecting misspelled parameters."""
    config = DEFAULTS.copy()
    values = {} if path is None else yaml.safe_load(Path(path).read_text("utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config.update(values)
    config.update(overrides or {})
    for key in ("epochs", "batch_size", "image_size", "feature_dim", "kl_ramp_epochs"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive.")
    if config["mask_policy"] not in {"aligned", "random_area", "displaced"}:
        raise ValueError("mask_policy must be aligned, random_area or displaced.")
    return config


def epoch_lr(config, epoch):
    """One-based warmup, followed by cosine decay to the final-epoch minimum."""
    import math

    warmup = min(config["warmup_epochs"], config["epochs"])
    if epoch <= warmup:
        fraction = (epoch - 1) / max(warmup - 1, 1)
        return config["start_lr"] + fraction * (config["lr"] - config["start_lr"])
    progress = (epoch - warmup) / (config["epochs"] - warmup)
    return config["minimum_lr"] + 0.5 * (config["lr"] - config["minimum_lr"]) * (
        1 + math.cos(math.pi * progress)
    )
