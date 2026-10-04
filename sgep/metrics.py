"""Image-weighted recognition, rejection, and calibration metrics."""

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import roc_auc_score


def validate_predictions(predictions: Mapping) -> dict:
    """Validate the prediction-file boundary and return NumPy arrays."""
    required = {"probabilities", "uncertainty", "labels"}
    missing = required.difference(predictions)
    if missing:
        raise ValueError(f"Missing prediction arrays: {sorted(missing)}")
    data = {key: np.asarray(value) for key, value in predictions.items()}
    p = np.asarray(data["probabilities"], dtype=float)
    u = np.asarray(data["uncertainty"], dtype=float)
    y = data["labels"]
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("probabilities must be a nonempty N x K matrix.")
    if u.shape != (len(p),) or y.shape != (len(p),):
        raise ValueError("uncertainty and labels must have shape (N,).")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    if not np.isfinite(u).all():
        raise ValueError("Uncertainty scores must be finite.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must contain integer class indices.")
    if (y < -1).any() or (y >= p.shape[1]).any():
        raise ValueError("Known labels must index probability columns; unknown labels are -1.")
    for key, values in data.items():
        if values.ndim and len(values) != len(p):
            raise ValueError(f"Prediction array {key!r} has a different row count.")
    if "prediction" in data:
        pred = data["prediction"]
        if pred.shape != (len(p),) or not np.issubdtype(pred.dtype, np.integer):
            raise ValueError("prediction must contain N integer class indices.")
        if (pred < 0).any() or (pred >= p.shape[1]).any():
            raise ValueError("prediction must index the known-class columns.")
    data.update(probabilities=p, uncertainty=u, labels=y.astype(int))
    return data


def threshold_at_coverage(known_uncertainty, target: float = 0.95) -> float:
    """Smallest observed score accepting at least ``target`` known images."""
    scores = np.asarray(known_uncertainty, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Supply a nonempty vector of finite known scores.")
    if not 0 < target <= 1:
        raise ValueError("Coverage target must be in (0, 1].")
    rank = int(np.ceil(target * len(scores))) - 1
    return float(np.partition(scores, rank)[rank])


def oscr_curve(known_u, unknown_u, known_correct):
    """Return (unknown FPR, known CCR, thresholds), with tied scores grouped.

    Rates are fractions. The first threshold rejects all images; the final
    threshold accepts all, and its CCR equals closed-set known accuracy.
    """
    known_u = np.asarray(known_u, dtype=float)
    unknown_u = np.asarray(unknown_u, dtype=float)
    correct = np.asarray(known_correct, dtype=bool)
    if known_u.ndim != 1 or unknown_u.ndim != 1 or correct.shape != known_u.shape:
        raise ValueError("Expected score vectors and one correctness flag per known image.")
    if not len(known_u) or not len(unknown_u):
        raise ValueError("OSCR requires both known and unknown images.")
    scores = np.concatenate((known_u, unknown_u))
    if not np.isfinite(scores).all():
        raise ValueError("OSCR scores must be finite.")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    unknown = np.r_[np.zeros(len(known_u)), np.ones(len(unknown_u))]
    correct_all = np.r_[correct.astype(float), np.zeros(len(unknown_u))]
    fpr = np.r_[0.0, np.cumsum(unknown[order])[ends] / len(unknown_u)]
    ccr = np.r_[0.0, np.cumsum(correct_all[order])[ends] / len(known_u)]
    thresholds = np.r_[-np.inf, sorted_scores[ends]]
    return fpr, ccr, thresholds


def oscr(known_u, unknown_u, known_correct) -> float:
    """Area under the full OSCR curve as a fraction."""
    fpr, ccr, _ = oscr_curve(known_u, unknown_u, known_correct)
    return float(np.sum(np.diff(fpr) * (ccr[1:] + ccr[:-1]) / 2))


def reliability_bins(probabilities, labels, n_bins: int = 15) -> dict:
    """Top-label reliability data for equal-width bins; the final bin includes 1."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    data = validate_predictions({"probabilities": p, "labels": y,
                                 "uncertainty": np.zeros(len(y))})
    if (y < 0).any():
        raise ValueError("Calibration metrics require known labels only.")
    p = data["probabilities"]
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    indices = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins)
    accuracy = np.zeros(n_bins, dtype=float)
    mean_confidence = np.zeros(n_bins, dtype=float)
    np.divide(np.bincount(indices, weights=correct, minlength=n_bins), counts,
              out=accuracy, where=counts > 0)
    np.divide(np.bincount(indices, weights=confidence, minlength=n_bins), counts,
              out=mean_confidence, where=counts > 0)
    return {"count": counts, "accuracy": accuracy, "confidence": mean_confidence,
            "lower": np.arange(n_bins) / n_bins,
            "upper": np.arange(1, n_bins + 1) / n_bins}


def calibration_metrics(probabilities, labels, n_bins: int = 15) -> dict:
    """Known-only ECE in percent, NLL, and multiclass Brier without division by K."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    bins = reliability_bins(p, y, n_bins)
    ece = np.sum(bins["count"] / len(y)
                 * np.abs(bins["accuracy"] - bins["confidence"]))
    targets = np.eye(p.shape[1])[y]
    return {
        "ece": float(100 * ece),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(np.square(p - targets).sum(axis=1).mean()),
    }


def calibrate_threshold(predictions: Mapping, rule: str = "coverage95") -> float:
    """Select a threshold exclusively from supplied calibration predictions.

    Coverage rules use known images only. ``youden`` maximizes known correct
    acceptance minus proxy false acceptance, using smaller thresholds in ties.
    Final unknowns must never be supplied as calibration examples.
    """
    data = validate_predictions(predictions)
    known = data["labels"] >= 0
    if not known.any():
        raise ValueError("Threshold calibration requires known images.")
    if "role" in data:
        roles = np.char.lower(data["role"].astype(str))
        if np.isin(roles, ["unknown", "final_unknown", "final-unknown"]).any():
            raise ValueError("Final unknown images cannot select a calibration threshold.")
    if rule.startswith("coverage"):
        try:
            target = float(rule.removeprefix("coverage")) / 100
        except ValueError as exc:
            raise ValueError("Use a coverage percentage, such as coverage95.") from exc
        return threshold_at_coverage(data["uncertainty"][known], target)
    if rule != "youden":
        raise ValueError("Threshold rule must be coverage<PERCENT> or youden.")
    if known.all():
        raise ValueError("Youden calibration requires disjoint proxy unknown images.")
    prediction = data.get("prediction", data["probabilities"].argmax(axis=1))
    fpr, ccr, thresholds = oscr_curve(
        data["uncertainty"][known], data["uncertainty"][~known],
        prediction[known] == data["labels"][known],
    )
    # Compare integer numerators to preserve exact objective ties even when
    # the two empirical rates have different denominators.
    n_known, n_proxy = int(known.sum()), int((~known).sum())
    accepted_correct = np.rint(ccr[1:] * n_known).astype(np.int64)
    accepted_proxy = np.rint(fpr[1:] * n_proxy).astype(np.int64)
    objective = accepted_correct * n_proxy - accepted_proxy * n_known
    # The protocol maximizes over observed calibration scores, not reject-all.
    best = np.argmax(objective) + 1
    return float(thresholds[best])


def evaluate_predictions(predictions: Mapping, threshold: float | None = None) -> dict:
    """Evaluate prediction arrays using the revision's metric definitions.

    Accuracy, AUROC, OSCR, ECE, coverage and FPR are percentages. Missing
    populations yield ``None`` rather than an invented open-set statistic.
    FPR95 recalculates a descriptive test threshold; operating metrics use only
    the supplied fixed threshold.
    """
    data = validate_predictions(predictions)
    p, y, u = (data[key] for key in ("probabilities", "labels", "uncertainty"))
    pred = data.get("prediction", p.argmax(axis=1))
    known = y >= 0
    unknown = ~known
    n_known, n_unknown = int(known.sum()), int(unknown.sum())
    metrics = {
        "n_known": n_known, "n_unknown": n_unknown,
        "known_accuracy": None, "auroc": None, "oscr": None, "fpr95": None,
        "fpr95_threshold": None, "ece": None, "nll": None, "brier": None,
        "threshold": threshold, "known_coverage": None,
        "accepted_known_accuracy": None, "n_accepted_known": None,
        "ccr": None, "unknown_fpr": None, "j": None,
    }
    if n_known:
        correct = pred[known] == y[known]
        metrics["known_accuracy"] = float(100 * correct.mean())
        metrics.update(calibration_metrics(p[known], y[known]))
    if n_known and n_unknown:
        metrics["auroc"] = float(100 * roc_auc_score(unknown.astype(int), u))
        metrics["oscr"] = float(100 * oscr(u[known], u[unknown], correct))
        tau95 = threshold_at_coverage(u[known])
        metrics["fpr95_threshold"] = tau95
        metrics["fpr95"] = float(100 * (u[unknown] <= tau95).mean())
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("The operating threshold must be finite.")
        if n_known:
            accepted = u[known] <= threshold
            metrics["n_accepted_known"] = int(accepted.sum())
            metrics["known_coverage"] = float(100 * accepted.mean())
            metrics["ccr"] = float(100 * (correct & accepted).mean())
            if accepted.any():
                metrics["accepted_known_accuracy"] = float(100 * correct[accepted].mean())
        if n_unknown:
            metrics["unknown_fpr"] = float(100 * (u[unknown] <= threshold).mean())
        if n_known and n_unknown:
            metrics["j"] = metrics["ccr"] - metrics["unknown_fpr"]
    return metrics
