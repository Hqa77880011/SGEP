"""Prediction diagnostics and paired statistical analysis of completed runs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import calibrate_threshold, evaluate_predictions, validate_predictions


def load_predictions(path) -> dict:
    """Read a prediction NPZ without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as source:
        return validate_predictions({key: source[key] for key in source.files})


def subset_predictions(predictions, selected) -> dict:
    """Select image rows while retaining scalar metadata arrays."""
    n_images = len(predictions["labels"])
    return {key: values[selected] if np.ndim(values) and len(values) == n_images else values
            for key, values in predictions.items()}


def paired_summary(baseline, proposed) -> dict:
    """Student-t interval and two-sided paired test for proposed minus baseline."""
    baseline = np.asarray(baseline, dtype=float)
    proposed = np.asarray(proposed, dtype=float)
    if baseline.ndim != 1 or baseline.shape != proposed.shape or len(baseline) < 2:
        raise ValueError("Supply at least two equal-length vectors of paired observations.")
    if not np.isfinite(baseline).all() or not np.isfinite(proposed).all():
        raise ValueError("Paired observations must be finite.")
    difference = proposed - baseline
    n = len(difference)
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1) * se)
    # A constant difference has no variance estimate for a conventional t test.
    t_value = mean / se if se else None
    p_value = float(2 * stats.t.sf(abs(t_value), n - 1)) if se else None
    return {
        "n_pairs": n, "baseline_mean": float(baseline.mean()),
        "baseline_sd": float(baseline.std(ddof=1)),
        "proposed_mean": float(proposed.mean()),
        "proposed_sd": float(proposed.std(ddof=1)),
        "difference": mean, "difference_sd": float(difference.std(ddof=1)),
        "ci_low": mean - half, "ci_high": mean + half,
        "t": t_value, "df": n - 1, "p_two_sided": p_value,
        "test_defined": bool(se),
    }


def holm_adjust(p_values) -> np.ndarray:
    """Holm correction preserving the order of the supplied hypothesis family."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be a finite vector in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def _cluster_pools(data):
    if "group_id" not in data:
        raise ValueError("Cluster analysis requires supplied group_id values.")
    ids = np.asarray(data["group_id"]).astype(str)
    if ids.shape != data["labels"].shape or np.isin(ids, ["", "nan", "None"]).any():
        raise ValueError("Every image needs a nonempty group_id.")
    groups = {name: np.flatnonzero(ids == name) for name in np.unique(ids)}
    pools = {}
    for indices in groups.values():
        roles = data["labels"][indices] >= 0
        # Mixed-role clusters are a separate stratum. Sampling one cluster
        # always takes all its images; it never splits a patient by role.
        stratum = (bool(roles.any()), bool((~roles).any()))
        pools.setdefault(stratum, []).append(indices)
    return pools


def _draw_clusters(pools, rng):
    return np.concatenate([pool[i] for pool in pools.values()
                           for i in rng.integers(len(pool), size=len(pool))])


def _align_predictions(baseline, proposed):
    baseline = validate_predictions(baseline)
    proposed = validate_predictions(proposed)
    if "image_id" not in baseline or "image_id" not in proposed:
        raise ValueError("Pair models using genuine image_id values.")
    for data in (baseline, proposed):
        ids = data["image_id"].astype(str)
        if len(np.unique(ids)) != len(ids) or (ids == "").any():
            raise ValueError("image_id values must be unique and nonempty within each model.")
    baseline = subset_predictions(baseline, np.argsort(baseline["image_id"].astype(str)))
    proposed = subset_predictions(proposed, np.argsort(proposed["image_id"].astype(str)))
    for name in ("image_id", "labels", "group_id"):
        if name not in baseline or name not in proposed:
            raise ValueError(f"Paired cluster analysis requires {name} in both models.")
        if not np.array_equal(baseline[name], proposed[name]):
            raise ValueError(f"Models have different {name}; paired comparisons need the same images and groups.")
    return baseline, proposed


def paired_cluster_bootstrap(baseline, proposed, thresholds=(None, None),
                             metrics=("known_accuracy", "auroc", "oscr", "ece", "fpr95"),
                             n_resamples=2000, seed=2026) -> dict:
    """Paired role-stratified cluster intervals, conditional on fitted models.

    Known-only, unknown-only and mixed-role groups form strata. The same draws
    feed both models; a mixed-role group is drawn once with all its images.
    Operating thresholds stay fixed. Descriptive FPR95 is recalculated in each
    bootstrap sample. The returned differences are proposed minus baseline.
    """
    if n_resamples < 1 or len(thresholds) != 2:
        raise ValueError("Supply positive n_resamples and one fixed threshold per model.")
    baseline, proposed = _align_predictions(baseline, proposed)
    pools = _cluster_pools(baseline)
    point = [evaluate_predictions(data, tau)
             for data, tau in zip((baseline, proposed), thresholds)]
    for metric in metrics:
        if metric not in point[0] or point[0][metric] is None or point[1][metric] is None:
            raise ValueError(f"Metric {metric!r} cannot be evaluated on this population.")
    samples = np.empty((n_resamples, 2, len(metrics)))
    rng = np.random.default_rng(seed)
    for index in range(n_resamples):
        draw = _draw_clusters(pools, rng)
        for model_id, (data, tau) in enumerate(zip((baseline, proposed), thresholds)):
            result = evaluate_predictions(subset_predictions(data, draw), tau)
            samples[index, model_id] = [result[metric] for metric in metrics]
    report = {"n_resamples": n_resamples, "seed": seed,
              "n_images": len(baseline["labels"]),
              "n_groups": sum(len(pool) for pool in pools.values()),
              "strata": {"known" if roles == (True, False) else
                          "unknown" if roles == (False, True) else "mixed": len(pool)
                          for roles, pool in pools.items()}, "metrics": {}}
    for index, metric in enumerate(metrics):
        differences = samples[:, 1, index] - samples[:, 0, index]
        ci = np.quantile(differences, [0.025, 0.975])
        base_ci = np.quantile(samples[:, 0, index], [0.025, 0.975])
        prop_ci = np.quantile(samples[:, 1, index], [0.025, 0.975])
        report["metrics"][metric] = {
            "baseline": point[0][metric], "proposed": point[1][metric],
            "difference": point[1][metric] - point[0][metric],
            "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "baseline_ci": base_ci.tolist(), "proposed_ci": prop_ci.tolist(),
        }
    return report


def _diagnostic_row(name, values, group_ids, n_resamples, rng):
    group_ids = np.asarray(group_ids).astype(str)
    groups, inverse = np.unique(group_ids, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.column_stack([np.bincount(inverse, weights=array)
                              for array in values.values()])
    samples = np.empty((n_resamples, len(values)))
    for index in range(n_resamples):
        draw = rng.integers(len(groups), size=len(groups))
        samples[index] = totals[draw].sum(axis=0) / counts[draw].sum()
    row = {"group": name, "n_images": len(group_ids), "n_groups": len(groups)}
    for index, (metric, array) in enumerate(values.items()):
        row[metric] = float(array.mean())
        lower, upper = np.quantile(samples[:, index], [0.025, 0.975])
        row[metric + "_ci_low"] = float(lower)
        row[metric + "_ci_high"] = float(upper)
    return row


def _evidence_arrays(data):
    evidence = np.asarray(data["evidence"], dtype=float)
    if evidence.shape != data["probabilities"].shape or not np.isfinite(evidence).all() or (evidence < 0).any():
        raise ValueError("evidence must be a finite nonnegative N x K matrix.")
    k = evidence.shape[1]
    total = evidence.sum(axis=1)
    uncertainty = k / (total + k)
    if not np.allclose(uncertainty, data["uncertainty"], atol=1e-5):
        raise ValueError("Evidential diagnostic scores must satisfy u = K / (E + K).")
    top = np.sort(evidence, axis=1)
    margin = top[:, -1] - (top[:, -2] if k > 1 else 0)
    values = {"evidence": total, "uncertainty": uncertainty,
              "delta_e": margin / (total + 1e-7)}
    background = None
    if "background_evidence" in data:
        bg = np.asarray(data["background_evidence"], dtype=float)
        if bg.shape != evidence.shape or not np.isfinite(bg).all() or (bg < 0).any():
            raise ValueError("background_evidence must be a finite nonnegative N x K matrix.")
        bg_total = bg.sum(axis=1)
        values["rho"] = total / (total + bg_total + 1e-7)
        bg_top = np.sort(bg, axis=1)
        bg_margin = bg_top[:, -1] - (bg_top[:, -2] if k > 1 else 0)
        background = {"evidence": bg_total, "uncertainty": k / (bg_total + k),
                      "rho": bg_total / (total + bg_total + 1e-7),
                      "delta_e": bg_margin / (bg_total + 1e-7)}
    elif "rho" in data:
        values["rho"] = np.asarray(data["rho"], dtype=float)
    return values, background


def evidence_diagnostics(predictions, threshold, proxy_predictions=None,
                          n_resamples=2000, seed=2026) -> pd.DataFrame:
    """Per-image evidence means and cluster intervals by semantic/acceptance group.

    Background diagnostics use complementary views of known test images. Its
    rho swaps the numerator to background evidence. A separately supplied
    calibration file contributes only proxy examples, not its known images.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    data = validate_predictions(predictions)
    if "evidence" not in data:
        return pd.DataFrame()
    _cluster_pools(data)
    values, background = _evidence_arrays(data)
    known = data["labels"] >= 0
    proxy = np.zeros(len(known), dtype=bool)
    if "role" in data:
        proxy = np.char.find(np.char.lower(data["role"].astype(str)), "proxy") >= 0
    groups = {"accepted_known": known & (data["uncertainty"] <= threshold),
              "rejected_known": known & (data["uncertainty"] > threshold),
              "proxy_unknown": (~known) & proxy,
              "final_unknown": (~known) & (~proxy)}
    rows = []
    rng = np.random.default_rng(seed)
    for name, selected in groups.items():
        if selected.any():
            rows.append(_diagnostic_row(name, {key: array[selected] for key, array in values.items()},
                                        data["group_id"][selected], n_resamples, rng))
    if background is not None and known.any():
        rows.append(_diagnostic_row("background_only", {key: array[known] for key, array in background.items()},
                                    data["group_id"][known], n_resamples, rng))
    if proxy_predictions is not None:
        proxy_data = validate_predictions(proxy_predictions)
        selected = proxy_data["labels"] == -1
        if selected.any() and "evidence" in proxy_data:
            _cluster_pools(proxy_data)
            proxy_values, _ = _evidence_arrays(proxy_data)
            rows.append(_diagnostic_row("calibration_proxy_unknown", {
                key: array[selected] for key, array in proxy_values.items()},
                proxy_data["group_id"][selected], n_resamples, rng))
    return pd.DataFrame(rows)


def threshold_sensitivity(predictions, calibration=None, thresholds=None) -> pd.DataFrame:
    """Evaluate fixed thresholds or calibration-selected coverage/J rules."""
    rows = []
    if thresholds is not None:
        choices = [("fixed", float(tau)) for tau in thresholds]
    elif calibration is not None:
        choices = [(rule, calibrate_threshold(calibration, rule))
                   for rule in ("coverage90", "coverage95", "coverage97", "youden")]
    else:
        raise ValueError("Supply calibration predictions or explicit fixed thresholds.")
    for rule, threshold in choices:
        rows.append({"rule": rule, **evaluate_predictions(predictions, threshold)})
    return pd.DataFrame(rows)


def unknown_difficulty(predictions, threshold, near_category) -> pd.DataFrame:
    """Compare all final unknowns with the predetermined near-class subset."""
    data = validate_predictions(predictions)
    if "category" not in data:
        raise ValueError("Unknown difficulty analysis requires original category names.")
    canonical = lambda array: np.char.replace(np.char.lower(np.asarray(array).astype(str)), "_", " ")
    known = data["labels"] >= 0
    near = (~known) & (canonical(data["category"]) == canonical([near_category])[0])
    if not near.any():
        raise ValueError(f"No final unknown test images match near category {near_category!r}.")
    rows = []
    for name, selected in (("mixed", np.ones(len(known), dtype=bool)), ("near", known | near)):
        subset = subset_predictions(data, selected)
        row = {"condition": name, **evaluate_predictions(subset, threshold)}
        if "group_id" in subset:
            row["n_groups"] = len(np.unique(subset["group_id"]))
        rows.append(row)
    return pd.DataFrame(rows)


def mask_quality_analysis(predictions, threshold, quality=None) -> pd.DataFrame:
    """Apply per-image Dice strata to evaluation-only reference-mask scores.

    ``quality`` may be a CSV path or DataFrame with unique image_id and dice
    columns. Both-empty reference/prediction masks should have missing Dice and
    are counted as excluded, rather than assigned a perfect lesion score.
    """
    data = validate_predictions(predictions)
    if quality is None:
        score_key = "mask_dice" if "mask_dice" in data else "dice"
        if score_key not in data:
            raise ValueError("Supply evaluation-only Dice scores or a mask-quality CSV.")
        dice = np.asarray(data[score_key], dtype=float)
        annotation = None
    else:
        if "image_id" not in data:
            raise ValueError("Mask-quality matching requires prediction image_id values.")
        frame = pd.read_csv(quality) if not isinstance(quality, pd.DataFrame) else quality.copy()
        if not {"image_id", "dice"}.issubset(frame) or frame.image_id.duplicated().any():
            raise ValueError("Mask-quality input needs unique image_id and dice columns.")
        scores = frame.set_index(frame.image_id.astype(str))["dice"]
        dice = scores.reindex(data["image_id"].astype(str)).to_numpy(dtype=float)
        annotation = frame.set_index(frame.image_id.astype(str)).reindex(data["image_id"].astype(str))
    valid = np.isfinite(dice)
    if ((dice[valid] < 0) | (dice[valid] > 1)).any():
        raise ValueError("Dice scores must be in [0, 1].")
    strata = {"high": valid & (dice >= 0.85),
              "medium": valid & (dice >= 0.65) & (dice < 0.85),
              "low": valid & (dice < 0.65)}
    rows = []
    for name, selected in strata.items():
        if selected.any():
            subset = subset_predictions(data, selected)
            row = {"stratum": name, "dice_mean": float(dice[selected].mean()),
                   "n_images": int(selected.sum()),
                   "n_groups": len(np.unique(subset["group_id"])) if "group_id" in subset else None,
                   "n_excluded_or_unscored": int((~valid).sum()),
                   **evaluate_predictions(subset, threshold)}
            if annotation is not None and "annotator_dice" in annotation:
                scores = pd.to_numeric(annotation.loc[selected, "annotator_dice"], errors="raise").dropna()
                row["inter_annotator_dice_mean"] = float(scores.mean()) if len(scores) else None
                row["n_inter_annotator_scored"] = len(scores)
            if annotation is not None and "adjudicated" in annotation:
                values = annotation.loc[selected, "adjudicated"].dropna().astype(str).str.lower()
                if not values.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
                    raise ValueError("adjudicated must contain boolean or 0/1 values.")
                row["n_adjudicated"] = int(values.isin(["true", "1", "1.0"]).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def mask_quality_coverage(predictions, quality=None) -> pd.DataFrame:
    """Reference-mask coverage against all eligible images, by category and role."""
    data = validate_predictions(predictions)
    if quality is not None:
        frame = pd.read_csv(quality) if not isinstance(quality, pd.DataFrame) else quality.copy()
        if not {"image_id", "dice"}.issubset(frame) or frame.image_id.duplicated().any():
            raise ValueError("Mask-quality input needs unique image_id and dice columns.")
        if "image_id" not in data:
            raise ValueError("Mask-quality coverage requires prediction image_id values.")
        score = frame.set_index(frame.image_id.astype(str))["dice"].reindex(data["image_id"].astype(str)).to_numpy(float)
    else:
        score = data.get("mask_dice", data.get("dice"))
        if score is None:
            raise ValueError("Supply per-image Dice scores.")
    scored = np.isfinite(score)
    masks = [("all", "all", np.ones(len(scored), dtype=bool))]
    for field in ("category", "role"):
        if field in data:
            masks.extend((field, str(value), data[field] == value) for value in np.unique(data[field]))
    rows = []
    for field, value, selected in masks:
        row = {"field": field, "value": value, "n_eligible_images": int(selected.sum()),
               "n_scored_images": int((selected & scored).sum()),
               "reference_coverage": float(100 * scored[selected].mean())}
        if "group_id" in data:
            row.update(n_eligible_groups=len(np.unique(data["group_id"][selected])),
                       n_scored_groups=len(np.unique(data["group_id"][selected & scored])))
        rows.append(row)
    return pd.DataFrame(rows)


def annotation_summary(predictions, quality) -> pd.DataFrame:
    """Summarize supplied annotator agreement and adjudications with coverage."""
    data = validate_predictions(predictions)
    frame = pd.read_csv(quality) if not isinstance(quality, pd.DataFrame) else quality.copy()
    if not {"image_id", "dice"}.issubset(frame) or frame.image_id.duplicated().any():
        raise ValueError("Annotation input needs unique image_id and dice columns.")
    annotation = frame.set_index(frame.image_id.astype(str)).reindex(data["image_id"].astype(str))
    rows = mask_quality_coverage(data, frame)
    agreement = None
    if "annotator_dice" in annotation:
        agreement = pd.to_numeric(annotation.annotator_dice, errors="raise").to_numpy(float)
        finite = np.isfinite(agreement)
        if ((agreement[finite] < 0) | (agreement[finite] > 1)).any():
            raise ValueError("Inter-annotator Dice must be in [0, 1].")
    adjudicated = None
    if "adjudicated" in annotation:
        values = annotation.adjudicated.dropna().astype(str).str.lower()
        if not values.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
            raise ValueError("adjudicated must contain boolean or 0/1 values.")
        adjudicated = annotation.adjudicated.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
    for index, row in rows.iterrows():
        selected = np.ones(len(data["labels"]), dtype=bool) if row.field == "all" else data[row.field].astype(str) == row.value
        if agreement is not None:
            valid = selected & np.isfinite(agreement)
            rows.loc[index, "n_inter_annotator_scored"] = int(valid.sum())
            rows.loc[index, "inter_annotator_dice_mean"] = float(agreement[valid].mean()) if valid.any() else np.nan
        if adjudicated is not None:
            rows.loc[index, "n_adjudicated"] = int((selected & adjudicated).sum())
    return rows


def factorial_effects(records) -> pd.DataFrame:
    """Paired 2 x 2 effects from dataset/seed/roi/bg rows and metric columns."""
    data = pd.DataFrame(records)
    required = {"dataset", "seed", "roi", "bg"}
    if not required.issubset(data):
        raise ValueError("Factorial records need dataset, seed, roi and bg columns.")
    if not data.roi.isin([0, 1]).all() or not data.bg.isin([0, 1]).all():
        raise ValueError("Factorial roi and bg switches must be 0 or 1.")
    if data.duplicated(["dataset", "seed", "roi", "bg"]).any():
        raise ValueError("Each dataset/seed/factor combination must occur once.")
    rows = []
    for dataset, subset in data.groupby("dataset", sort=True):
        for metric in ("known_accuracy", "auroc", "oscr", "ece", "fpr95"):
            if metric not in subset:
                continue
            wide = subset.pivot(index="seed", columns=["roi", "bg"], values=metric)
            expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
            if not all(cell in wide for cell in expected) or wide[expected].isna().any().any():
                raise ValueError("Every paired seed must have all four completed factorial runs.")
            a, b, c, d = (wide[cell].to_numpy(dtype=float) for cell in expected)
            contrasts = {"roi_main": ((c + d) - (a + b)) / 2,
                         "background_main": ((b + d) - (a + c)) / 2,
                         "interaction": d - c - b + a}
            for effect, values in contrasts.items():
                summary = paired_summary(np.zeros(len(values)), values)
                rows.append({"dataset": dataset, "metric": metric, "effect": effect, **summary})
    return pd.DataFrame(rows)


def _write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def analyze_predictions(predictions, threshold, output_dir, calibration=None,
                         near_category=None, mask_quality=None, n_resamples=2000, seed=2026) -> dict:
    """Write metrics, operating-point tables, diagnostics and score plots."""
    from .visualize import plot_predictions

    data = load_predictions(predictions) if isinstance(predictions, (str, Path)) else validate_predictions(predictions)
    cal = load_predictions(calibration) if isinstance(calibration, (str, Path)) else calibration
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(data, threshold)
    _write_json(output / "metrics.json", metrics)
    if cal is not None:
        threshold_sensitivity(data, cal).to_csv(output / "threshold_sensitivity.csv", index=False)
    if near_category is not None:
        unknown_difficulty(data, threshold, near_category).to_csv(output / "unknown_difficulty.csv", index=False)
    has_dice = any(key in data and np.isfinite(data[key]).any() for key in ("mask_dice", "dice"))
    if mask_quality is not None or has_dice:
        mask_quality_analysis(data, threshold, mask_quality).to_csv(output / "mask_quality.csv", index=False)
        mask_quality_coverage(data, mask_quality).to_csv(output / "mask_quality_coverage.csv", index=False)
        if mask_quality is not None:
            annotation_summary(data, mask_quality).to_csv(output / "annotation_summary.csv", index=False)
    if "evidence" in data and "group_id" in data:
        evidence_diagnostics(data, threshold, cal, n_resamples, seed).to_csv(output / "evidence_diagnostics.csv", index=False)
    plot_predictions(data, output, threshold)
    return metrics


def aggregate_runs(run_dirs, output_dir, baseline="roi_guided", proposed="sgep") -> dict:
    """Aggregate metric JSON files with explicit dataset/method/seed metadata.

    Each JSON contains dataset, method, seed and a metrics mapping. Extra
    roi/bg flags allow the same completed records to drive factorial analysis.
    The primary AUROC/FPR95 family is Holm corrected across supplied datasets.
    """
    from .visualize import plot_method_comparison

    rows = []
    for source in run_dirs:
        path = Path(source)
        if path.is_dir():
            path = path / "metrics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record.get("metadata", record)
        missing = {"dataset", "method", "seed"}.difference(metadata)
        if missing:
            raise ValueError(f"{path} is missing run metadata {sorted(missing)}.")
        values = record.get("metrics", {key: value for key, value in record.items()
                                         if key not in {"dataset", "method", "seed", "metadata"}})
        variant = metadata.get("variant", metadata["method"])
        label = variant.removeprefix("main/")
        factorial_flags = {}
        if variant.removeprefix("factorial/") in {"roi0_bg0", "roi0_bg1", "roi1_bg0", "roi1_bg1"}:
            factorial_flags = {"roi": int(float(metadata["lambda_roi"]) > 0),
                               "bg": int(float(metadata["lambda_bg"]) > 0)}
        rows.append({"dataset": metadata["dataset"], "method": label, "head_method": metadata["method"],
                     "variant": variant,
                     "seed": metadata["seed"], **values,
                     **factorial_flags,
                     **{key: metadata[key] for key in ("roi", "bg") if key in metadata}})
    data = pd.DataFrame(rows)
    if not len(data) or data.duplicated(["dataset", "method", "seed"]).any():
        raise ValueError("Supply nonduplicate completed dataset/method/seed records.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "per_run_metrics.csv", index=False)
    metric_names = [name for name in ("known_accuracy", "auroc", "oscr", "ece", "fpr95", "nll", "brier",
                                      "known_coverage", "accepted_known_accuracy", "unknown_fpr") if name in data]
    summary = data.groupby(["dataset", "method"])[metric_names].agg(["mean", "std", "count"])
    summary.columns = ["_".join(pair) for pair in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "method_summary.csv", index=False)
    comparisons = []
    for dataset, subset in data.groupby("dataset", sort=True):
        first = subset[subset.method == baseline].set_index("seed")
        second = subset[subset.method == proposed].set_index("seed")
        if not len(first) or not len(second):
            continue
        if set(first.index) != set(second.index):
            raise ValueError("Primary methods must have the same completed seed IDs.")
        if len(first) < 2:
            continue
        for metric in ("auroc", "fpr95"):
            if metric not in data or first[metric].isna().any() or second[metric].isna().any():
                continue
            seeds = sorted(first.index)
            comparisons.append({"dataset": dataset, "metric": metric,
                                "baseline_method": baseline, "proposed_method": proposed,
                                "seeds": seeds,
                                **paired_summary(first.loc[seeds, metric], second.loc[seeds, metric])})
    adjusted = holm_adjust([row["p_two_sided"] if row["p_two_sided"] is not None else 1.0
                            for row in comparisons])
    for row, p_value in zip(comparisons, adjusted):
        row["p_holm"] = float(p_value) if row["p_two_sided"] is not None else None
    _write_json(output / "paired_seed_statistics.json", comparisons)
    if {"roi", "bg"}.issubset(data):
        factorial = data[data.roi.notna() & data.bg.notna()]
        if len(factorial):
            factorial_effects(factorial).to_csv(output / "factorial_effects.csv", index=False)
    plot_method_comparison(data, output)
    return {"n_runs": len(data), "comparisons": comparisons,
            "n_primary_hypotheses": len(comparisons)}
