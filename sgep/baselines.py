"""Known-training-only OpenMax, PostMax, and CAC post-processing.

OpenMax uses SciPy Weibull maximum-likelihood fitting in place of libMR.
PostMax follows the pooled generalized-Pareto fitting rule in Algorithm 1.
These estimators are fitted on correctly classified known training examples.
"""

import numpy as np
from scipy.special import softmax
from scipy.stats import genpareto, weibull_min


POSTPROCESSORS = ("openmax", "postmax", "cac")


class PostprocessorFitError(ValueError):
    """Training predictions do not yet support the requested fitted estimator."""


def _distances(activations, centers, metric):
    difference = activations[:, None] - centers[None]
    euclidean = np.linalg.norm(difference, axis=-1)
    if metric == "euclidean":
        return euclidean
    dot = activations @ centers.T
    denominator = np.linalg.norm(activations, axis=-1)[:, None] * np.linalg.norm(centers, axis=-1)[None]
    cosine = 1 - np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    cosine = np.clip(cosine, 0, 2)
    if metric == "cosine":
        return cosine
    if metric == "eucos":
        return euclidean / 200.0 + cosine
    raise ValueError(f"Unsupported OpenMax distance metric: {metric}")


def fit_postprocessor(method, logits, features, labels, config):
    """Fit a serializable postprocessor using known training predictions only."""
    if method not in POSTPROCESSORS:
        return {}
    logits = np.asarray(logits, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or labels.shape != (len(logits),):
        raise ValueError("Training logits must be [N,K] and labels [N].")
    if features.ndim != 2 or len(features) != len(logits):
        raise ValueError("Training features must be [N,D] and match logits.")
    if not np.isfinite(logits).all() or not np.isfinite(features).all():
        raise ValueError("Postprocessor training inputs must be finite.")
    if ((labels < 0) | (labels >= logits.shape[1])).any():
        raise ValueError("Postprocessors can only fit known training labels.")
    correct = logits.argmax(axis=1) == labels
    if method == "postmax":
        norms = np.linalg.norm(features, axis=1)
        eligible = correct & (norms > 0)
        values = logits[eligible].max(axis=1) / norms[eligible]
        if len(values) < 3 or np.ptp(values) <= 1e-12:
            raise PostprocessorFitError("PostMax requires at least three nondegenerate correctly classified training scores.")
        shape, location, scale = genpareto.fit(values)
        if not np.isfinite((shape, location, scale)).all() or scale <= 0:
            raise PostprocessorFitError("PostMax generalized-Pareto fitting did not produce finite parameters.")
        return {
            "method": method, "shape": float(shape), "location": float(location),
            "scale": float(scale), "fit_count": int(eligible.sum()),
        }

    means = []
    tails = []
    counts = []
    metric = config.get("openmax_distance", "eucos")
    tail_size = int(config.get("openmax_tail_size", 20))
    if tail_size < 2:
        raise ValueError("openmax_tail_size must be at least two.")
    for class_index in range(logits.shape[1]):
        class_activations = logits[correct & (labels == class_index)]
        minimum = 2 if method == "openmax" else 1
        if len(class_activations) < minimum:
            raise PostprocessorFitError(
                f"{method} requires at least {minimum} correctly classified training "
                f"examples for class {class_index}; found {len(class_activations)}."
            )
        center = class_activations.mean(axis=0)
        means.append(center.tolist())
        counts.append(len(class_activations))
        if method == "openmax":
            distances = _distances(class_activations, center[None], metric)[:, 0]
            tail = np.sort(distances)[-tail_size:]
            if np.ptp(tail) <= 1e-12:
                raise PostprocessorFitError(f"OpenMax class {class_index} has a degenerate distance tail.")
            shape, location, scale = weibull_min.fit(np.maximum(tail, 1e-12), floc=0)
            if not np.isfinite((shape, location, scale)).all() or scale <= 0:
                raise PostprocessorFitError(f"OpenMax Weibull fitting failed for class {class_index}.")
            tails.append([float(shape), float(location), float(scale)])
    state = {"method": method, "means": means, "fit_counts": counts}
    if method == "openmax":
        state.update(tails=tails, distance=metric, tail_size=tail_size)
    return state


def apply_postprocessor(method, output, state, config):
    """Apply rejection scoring, retaining conditional known probabilities."""
    result = dict(output)
    if method not in POSTPROCESSORS:
        return result
    if not state or state.get("method") != method:
        raise ValueError(f"A fitted {method} postprocessor is required.")
    logits = np.asarray(output["logits"], dtype=np.float64)
    if method == "postmax":
        features = np.asarray(output["features"], dtype=np.float64)
        norms = np.linalg.norm(features, axis=1)
        normalized = np.divide(
            logits, norms[:, None], out=np.full_like(logits, -np.inf),
            where=norms[:, None] > 0,
        )
        support = genpareto.cdf(
            normalized, state["shape"], loc=state["location"], scale=state["scale"],
        )
        # PostMax support values need not sum to one. Calibration uses the
        # classifier's conditional known-label posterior; rejection uses the CDF.
        result.update(
            probabilities=softmax(logits, axis=1),
            uncertainty=1 - support.max(axis=1),
            class_support=support,
        )
        return result
    centers = np.asarray(state["means"])
    if method == "cac":
        distances = _distances(logits, centers, "euclidean")
        probabilities = softmax(-distances, axis=1)
        result.update(
            probabilities=probabilities,
            uncertainty=(distances * (1 - probabilities)).min(axis=1),
            distances=distances,
        )
        return result

    classes = logits.shape[1]
    ranks = min(int(config.get("openmax_rank", min(3, classes))), classes)
    if ranks < 1:
        raise ValueError("openmax_rank must be positive.")
    order = np.argsort(-logits, axis=1, kind="stable")[:, :ranks]
    rank_weights = np.zeros_like(logits)
    np.put_along_axis(
        rank_weights, order,
        np.broadcast_to((ranks - np.arange(ranks)) / ranks, order.shape), axis=1,
    )
    distances = _distances(logits, centers, state["distance"])
    tails = np.asarray(state["tails"])
    outlier_probability = weibull_min.cdf(
        distances, tails[None, :, 0], loc=tails[None, :, 1], scale=tails[None, :, 2],
    )
    removed_activation = logits * rank_weights * outlier_probability
    revised_logits = logits - removed_activation
    unknown_logit = removed_activation.sum(axis=1, keepdims=True)
    open_probabilities = softmax(np.concatenate((revised_logits, unknown_logit), axis=1), axis=1)
    # Renormalization preserves the OpenMax known-class prediction and provides
    # a K-class conditional distribution for the shared calibration metrics.
    result.update(
        probabilities=softmax(revised_logits, axis=1),
        uncertainty=open_probabilities[:, -1],
        open_probabilities=open_probabilities,
    )
    return result
