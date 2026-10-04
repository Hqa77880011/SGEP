"""Assertions for mathematical objectives and the image/mask gradient path."""

import numpy as np
import pytest
import torch
from scipy.special import softmax
from scipy.stats import genpareto
from torch.distributions import Dirichlet, kl_divergence

from sgep.baselines import PostprocessorFitError, apply_postprocessor, fit_postprocessor
from sgep.losses import (
    compute_loss, dirichlet_kl, evidential_loss, prototype_regularizer,
    symmetric_dirichlet_kl,
)
from sgep.models import build_model, initialize_prototypes, prototype_evidence


def test_prototype_evidence_monotonicity_and_probability_identity():
    prototypes = torch.zeros(3, 2, dtype=torch.float64)
    beta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    features = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    output = prototype_evidence(features, prototypes, beta)
    assert torch.all(output["evidence"][:-1] > output["evidence"][1:])
    assert torch.all(output["uncertainty"][:-1] < output["uncertainty"][1:])
    torch.testing.assert_close(output["probabilities"].sum(-1), torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(output["uncertainty"], 3 / (output["evidence"].sum(-1) + 3))
    output["uncertainty"].sum().backward()
    assert beta.grad < 0


def test_dirichlet_kl_matches_distribution_and_has_symmetric_view_gradients():
    alpha = torch.tensor([[1.2, 3.1, 2.4]], dtype=torch.float64, requires_grad=True)
    other = torch.tensor([[2.1, 1.4, 4.2]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(dirichlet_kl(alpha, other), kl_divergence(Dirichlet(alpha), Dirichlet(other)))
    torch.testing.assert_close(dirichlet_kl(alpha, alpha), torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0)
    forward = symmetric_dirichlet_kl(alpha, other)
    torch.testing.assert_close(forward, symmetric_dirichlet_kl(other, alpha))
    forward.backward()
    assert torch.isfinite(alpha.grad).all() and alpha.grad.abs().sum() > 0
    assert torch.isfinite(other.grad).all() and other.grad.abs().sum() > 0


def test_evidential_kl_removes_only_true_class_evidence():
    labels = torch.tensor([0])
    alpha = torch.tensor([[9.0, 1.0, 1.0]], dtype=torch.float64)
    _, ce, kl = evidential_loss(alpha, labels, 0.01)
    torch.testing.assert_close(kl, torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(ce, torch.digamma(torch.tensor(11.0, dtype=torch.float64)) - torch.digamma(torch.tensor(9.0, dtype=torch.float64)))
    _, _, unsupported = evidential_loss(torch.tensor([[9.0, 5.0, 1.0]], dtype=torch.float64), labels, 0.01)
    assert unsupported > 0


def test_prototype_regularizer_exact_margin_and_gradient():
    distances = torch.tensor([[0.2, 0.4, 2.0], [1.5, 0.1, 0.5]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = prototype_regularizer(distances, labels, 1.0)
    torch.testing.assert_close(loss, torch.tensor(((0.2 + 0.6 / 2) + (0.1 + 0.5 / 2)) / 2))
    loss.backward()
    torch.testing.assert_close(distances.grad, torch.tensor([[0.5, -0.25, 0], [0, 0.5, -0.25]]))


def test_rgb_masking_independent_branches_and_complete_objective_gradients():
    torch.manual_seed(7)
    config = {"method": "sgep", "backbone": "tiny", "pretrained": False, "feature_dim": 8}
    model = build_model(config, 3)
    global_weight = next(model.global_encoder.parameters())
    roi_weight = next(model.roi_encoder.parameters())
    torch.testing.assert_close(global_weight, roi_weight)
    assert global_weight.data_ptr() != roi_weight.data_ptr()
    assert model.fusion.bias is None
    images = torch.rand(3, 3, 16, 16)
    masks = torch.zeros(3, 1, 16, 16)
    masks[:, :, 4:12, 4:12] = 1
    assert torch.equal(model.mask_image(images, masks)[..., :4, :], torch.zeros_like(images[..., :4, :]))
    output = model(images, masks)
    torch.testing.assert_close(output["features"].norm(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
    loss, stats = compute_loss(model, output, images, masks, torch.tensor([0, 1, 2]), config, 11)
    assert stats["background"] > 0 and stats["prototype"] > 0
    loss.backward()
    for parameter in (global_weight, roi_weight, model.fusion.weight, model.prototypes, model.beta):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_prototype_initialization_is_class_mean_and_restores_training_mode():
    model = build_model({"backbone": "tiny", "pretrained": False, "feature_dim": 4}, 2)
    batch = {"image": torch.rand(4, 3, 16, 16), "mask": torch.ones(4, 1, 16, 16), "label": torch.tensor([0, 1, 0, 1])}
    model.eval()
    with torch.no_grad():
        features = model.encode(batch["image"], batch["mask"])
    model.train()
    initialize_prototypes(model, [batch], "cpu")
    assert model.training
    torch.testing.assert_close(model.prototypes[0], features[[0, 2]].mean(0))
    torch.testing.assert_close(model.prototypes[1], features[[1, 3]].mean(0))
    with pytest.raises(ValueError, match="missing known classes"):
        initialize_prototypes(model, [{**batch, "label": torch.zeros(4, dtype=torch.long)}], "cpu")
    assert model.training


@pytest.mark.parametrize("method", ["softmax", "maxlogit", "energy", "edl", "prototype", "cac", "arpl", "dual_ce"])
def test_baseline_scores_and_objectives_match_their_definitions(method):
    config = {"method": method, "backbone": "tiny", "pretrained": False, "feature_dim": 6, "energy_temperature": 2.0, "lambda_proto": 0.1, "lambda_cac": 0.7}
    model = build_model(config, 3)
    images, masks = torch.rand(3, 3, 16, 16), torch.ones(3, 1, 16, 16)
    labels = torch.tensor([0, 1, 2])
    output = model(images, masks)
    torch.testing.assert_close(output["probabilities"].sum(-1), torch.ones(3))
    loss, _ = compute_loss(model, output, images, masks, labels, config, 1)
    if method == "cac":
        expected_distances = torch.linalg.vector_norm(output["logits"][:, None] - 10 * torch.eye(3)[None], dim=-1)
        torch.testing.assert_close(output["distances"], expected_distances)
        torch.testing.assert_close(output["uncertainty"], (expected_distances * (1 - torch.softmax(-expected_distances, -1))).min(-1).values)
        expected_loss = torch.nn.functional.cross_entropy(-expected_distances, labels) + 0.7 * expected_distances[torch.arange(3), labels].mean()
        torch.testing.assert_close(loss, expected_loss)
    elif method == "arpl":
        features = output["features"]
        expected = (features[:, None] - model.reciprocal_points[None]).square().mean(-1) - features @ model.reciprocal_points.t()
        torch.testing.assert_close(output["logits"], expected)
        torch.testing.assert_close(output["uncertainty"], -expected.max(-1).values)
        expected_loss = torch.nn.functional.cross_entropy(expected, labels) + 0.1 * torch.relu(output["distances"][torch.arange(3), labels] - model.radius + 1).mean()
        torch.testing.assert_close(loss, expected_loss)
    elif method == "energy":
        torch.testing.assert_close(output["uncertainty"], -2 * torch.logsumexp(output["logits"] / 2, -1))
    elif method == "maxlogit":
        torch.testing.assert_close(output["uncertainty"], -output["logits"].max(-1).values)
    elif method == "prototype":
        torch.testing.assert_close(output["uncertainty"], output["distances"].min(-1).values)
    elif method == "edl":
        torch.testing.assert_close(output["uncertainty"], 3 / output["alpha"].sum(-1))
    else:
        torch.testing.assert_close(output["uncertainty"], 1 - output["probabilities"].max(-1).values)
    loss.backward()
    assert torch.isfinite(model.fusion.weight.grad).all() and model.fusion.weight.grad.abs().sum() > 0


def _training_activations():
    rng = np.random.default_rng(123)
    labels = np.repeat([0, 1], 12)
    logits = 0.3 + rng.uniform(0, 0.2, (24, 2))
    logits[np.arange(24), labels] += rng.uniform(2, 4, 24)
    return logits, rng.uniform(0.5, 2, (24, 5)), labels


def test_postmax_uses_raw_feature_norm_and_filters_incorrect_training_samples():
    logits, features, labels = _training_activations()
    altered = logits.copy()
    altered[-1] = [10.0, 0.0]
    state = fit_postprocessor("postmax", altered, features, labels, {})
    assert state["fit_count"] == 23
    expected = genpareto.fit(altered[:-1].max(-1) / np.linalg.norm(features[:-1], axis=1))
    np.testing.assert_allclose([state["shape"], state["location"], state["scale"]], expected)
    output = {"logits": np.repeat([[2.0, 0.0]], 2, axis=0), "features": np.array([[1.0, 0], [10.0, 0]])}
    applied = apply_postprocessor("postmax", output, state, {})
    assert applied["uncertainty"][0] < applied["uncertainty"][1]
    np.testing.assert_allclose(applied["probabilities"], softmax(output["logits"], axis=-1))
    with pytest.raises(ValueError, match="known training"):
        fit_postprocessor("postmax", logits, features, np.full_like(labels, -1), {})


def test_openmax_transfers_ranked_activation_to_unknown_and_keeps_conditional_probs():
    logits, features, labels = _training_activations()
    state = fit_postprocessor("openmax", logits, features, labels, {"openmax_distance": "euclidean", "openmax_tail_size": 10})
    query = {"logits": np.array([state["means"][0], [12.0, 9.0]]), "features": features[:2]}
    output = apply_postprocessor("openmax", query, state, {"openmax_rank": 1})
    assert output["uncertainty"][1] > output["uncertainty"][0]
    np.testing.assert_allclose(output["open_probabilities"].sum(-1), 1)
    np.testing.assert_allclose(output["probabilities"].sum(-1), 1)
    np.testing.assert_allclose(output["probabilities"], output["open_probabilities"][:, :-1] / (1 - output["uncertainty"][:, None]))
    all_ranks = apply_postprocessor("openmax", query, state, {"openmax_rank": 1000000})
    np.testing.assert_allclose(all_ranks["open_probabilities"].sum(-1), 1)
    with pytest.raises(PostprocessorFitError, match="class 1"):
        fit_postprocessor("openmax", logits[:12], features[:12], labels[:12], {})


def test_cac_postfit_uses_correct_class_means_and_distance_weighted_softmin():
    logits, features, labels = _training_activations()
    state = fit_postprocessor("cac", logits, features, labels, {})
    np.testing.assert_allclose(state["means"][0], logits[:12].mean(0))
    output = apply_postprocessor("cac", {"logits": logits, "features": features}, state, {})
    distances = np.linalg.norm(logits[:, None] - np.asarray(state["means"])[None], axis=-1)
    probabilities = softmax(-distances, axis=-1)
    np.testing.assert_allclose(output["probabilities"], probabilities)
    np.testing.assert_allclose(output["uncertainty"], (distances * (1 - probabilities)).min(-1))
