"""Paper comparisons and validation-only hyperparameter selection."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import calibrate_threshold, evaluate_predictions
from .train import predict, save_json, train

REPORTING_SEEDS = [11, 22, 33, 44, 55]
METHODS = ["softmax", "maxlogit", "openmax", "energy", "cac", "arpl",
           "postmax", "edl", "prototype", "roi_guided", "sgep"]


def baseline_settings(method):
    if method == "sgep":
        return {"method": "sgep", "view": "dual"}
    return {"method": method, "view": "roi" if method == "roi_guided" else "global",
            "lambda_proto": 0.1 if method == "prototype" else 0.0,
            "lambda_roi": 0.0, "lambda_bg": 0.0}


def variants(suite, config=None):
    """Return explicit model and mask changes for every paper training comparison."""
    prototype_weight = (config or {}).get("lambda_proto", 0.1)
    main = {method: baseline_settings(method) for method in METHODS}
    ablation = {
        "global_only": baseline_settings("softmax"),
        "roi_only": baseline_settings("roi_guided"),
        "global_roi": {"method": "dual_ce", "view": "dual", "lambda_proto": 0.0, "lambda_roi": 0.0, "lambda_bg": 0.0},
        "fusion_proto": {"method": "dual_ce", "view": "dual", "lambda_proto": prototype_weight, "lambda_roi": 0.0, "lambda_bg": 0.0},
        "fusion_edl": {"method": "sgep", "view": "dual", "lambda_proto": 0.0, "lambda_roi": 0.0, "lambda_bg": 0.0},
        "fusion_proto_edl": {"method": "sgep", "view": "dual", "lambda_proto": prototype_weight, "lambda_roi": 0.0, "lambda_bg": 0.0},
        "sgep": baseline_settings("sgep"),
    }
    factorial = {f"roi{roi}_bg{bg}": {"method": "sgep", "view": "dual",
                                             "lambda_roi": roi * 0.05, "lambda_bg": bg * 0.05}
                 for roi, bg in itertools.product((0, 1), repeat=2)}
    spatial = {
        "aligned": baseline_settings("sgep") | {"mask_policy": "aligned", "fill": "black"},
        "random_area": baseline_settings("sgep") | {"mask_policy": "random_area", "fill": "black"},
        "displaced": baseline_settings("sgep") | {"mask_policy": "displaced", "fill": "black"},
        "mean_fill": baseline_settings("sgep") | {"mask_policy": "aligned", "fill": "mean"},
        "dual_ce": ablation["global_roi"],
    }
    blocks = {"main": main, "ablation": ablation, "factorial": factorial, "spatial": spatial}
    if suite == "all":
        return {f"{block}/{name}": settings for block, mapping in blocks.items() for name, settings in mapping.items()}
    return blocks[suite]


def finish_run(checkpoint, manifest, root, output, dataset=None, device="auto", analyze=False):
    """Calibrate on its reserved split, then evaluate the fixed test population."""
    output = Path(output)
    calibration = predict(checkpoint, manifest, root, "calibration", dataset, device=device)
    np.savez_compressed(output / "calibration_predictions.npz", **calibration)
    threshold = calibrate_threshold(calibration, "coverage95")
    save_json(output / "threshold.json", {"rule": "coverage95", "threshold": threshold})
    test = predict(checkpoint, manifest, root, "test", dataset, device=device)
    np.savez_compressed(output / "test_predictions.npz", **test)
    metrics = evaluate_predictions(test, threshold)
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    save_json(output / "metrics.json", {
        "dataset": state["dataset"], "method": state["config"]["method"],
        "seed": state["config"]["seed"], "lambda_roi": state["config"]["lambda_roi"],
        "lambda_bg": state["config"]["lambda_bg"], "metrics": metrics,
    })
    if analyze:
        from .analysis import analyze_predictions
        analyze_predictions(test, threshold, output / "analysis", calibration=calibration)
    return metrics


def run_suite(config, manifest, root, output, suite="main", seeds=None, dataset=None, plan_only=False):
    """Train paired variants; --plan lists work without executing it."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = [{"variant": name, "seed": seed, "config": config | settings | {"seed": seed}}
            for name, settings in variants(suite, config).items() for seed in (seeds or REPORTING_SEEDS)]
    save_json(output / "run_plan.json", plan)
    print(f"{len(plan)} runs in {suite}; each uses {config['epochs']} epochs.")
    if plan_only:
        return plan
    rows = []
    for item in plan:
        run = output / item["variant"] / f"seed_{item['seed']}"
        checkpoint = train(item["config"], manifest, root, run, dataset)
        metrics = finish_run(checkpoint, manifest, root, run, dataset, config["device"])
        saved = __import__("json").loads((run / "metrics.json").read_text("utf-8"))
        saved["variant"] = item["variant"]
        save_json(run / "metrics.json", saved)
        rows.append({"dataset": dataset or "", "variant": item["variant"], "method": item["config"]["method"],
                     "seed": item["seed"], "run_dir": str(run), **metrics})
        pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    return rows


def search_candidates(config, method, count=24, seed=2026):
    """SGEP uses unique draws; baseline budgets also record repeated draws."""
    if count < 1:
        raise ValueError("The configuration count must be positive.")
    base = config | baseline_settings(method)
    grid = {"lr": [3e-5, 1e-4, 3e-4], "weight_decay": [1e-5, 1e-4]}
    if method == "sgep":
        grid.update({"lambda_proto": [0.05, 0.1, 0.2], "lambda_roi": [0.01, 0.05, 0.1],
                     "lambda_bg": [0.01, 0.05, 0.1], "lambda_kl_max": [0.001, 0.01],
                     "margin": [0.5, 1.0, 2.0]})
    elif method == "energy":
        grid["energy_temperature"] = [0.5, 1.0, 2.0]
    elif method == "openmax":
        grid.update({"openmax_tail_size": [10, 20, 40], "openmax_rank": [1, 3, 1000000]})
    elif method == "cac":
        grid.update({"cac_anchor": [5.0, 10.0, 20.0], "lambda_cac": [0.01, 0.1, 1.0]})
    elif method == "arpl":
        grid["arpl_weight"] = [0.01, 0.1, 1.0]
    keys = list(grid)
    population = [base | dict(zip(keys, values)) for values in itertools.product(*grid.values())]
    rng = np.random.default_rng(seed)
    # The supplied SGEP configuration is the first trial in the protocol.
    if method == "sgep":
        population = [candidate for candidate in population if candidate != base]
        if count - 1 > len(population):
            raise ValueError("Search count exceeds the unique SGEP grid.")
        return [base] + [population[i] for i in rng.choice(len(population), count - 1, replace=False)]
    indices = rng.choice(len(population), count, replace=count > len(population))
    return [population[i] for i in indices]


def run_search(config, manifest, root, output, method="sgep", count=24, seeds=(11, 22),
               dataset=None, plan_only=False):
    """Rank configurations by mean selection OSCR, leaving calibration/test unread."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = search_candidates(config, method, count)
    save_json(output / "search_plan.json", {"seeds": list(seeds), "configurations": candidates})
    if plan_only:
        print(f"{len(candidates) * len(seeds)} validation-only search runs.")
        return candidates
    records = []
    ranking = []
    for index, candidate in enumerate(candidates):
        scores = []
        for seed in seeds:
            run = output / f"config_{index + 1:02d}" / f"seed_{seed}"
            record = {"configuration": index + 1, "seed": seed, "config": candidate | {"seed": seed},
                      "status": "running"}
            records.append(record)
            save_json(output / "search_runs.json", records)
            try:
                checkpoint = train(candidate | {"seed": seed}, manifest, root, run, dataset)
            except (ValueError, RuntimeError, OSError, FloatingPointError) as error:
                record.update({"status": "failed", "error": str(error)})
                print(f"search configuration {index + 1}, seed {seed} failed: {error}")
            else:
                import torch
                selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
                scores.append(selected["selection_oscr"])
                record.update({"status": "complete", "selection_oscr": scores[-1]})
            save_json(output / "search_runs.json", records)
        if len(scores) == len(seeds):
            ranking.append((float(np.mean(scores)), index))
    if not ranking:
        raise RuntimeError("No configuration completed all selection seeds; inspect search_runs.json.")
    score, index = max(ranking, key=lambda item: (item[0], -item[1]))
    import yaml
    (output / "selected.yaml").write_text(yaml.safe_dump(candidates[index], sort_keys=False), "utf-8")
    save_json(output / "selection.json", {"configuration": index + 1, "mean_selection_oscr": score})
    return candidates[index]
