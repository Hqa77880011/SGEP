"""Paper objectives and the CAC/ARPL training losses."""

import torch
from torch.nn import functional as F


def dirichlet_kl(alpha, beta):
    """KL(Dir(alpha) || Dir(beta)) for each leading batch element."""
    alpha_sum = alpha.sum(-1, keepdim=True)
    beta_sum = beta.sum(-1, keepdim=True)
    log_normalizer = (
        torch.lgamma(alpha_sum) - torch.lgamma(beta_sum)
        - torch.lgamma(alpha).sum(-1, keepdim=True)
        + torch.lgamma(beta).sum(-1, keepdim=True)
    )
    expected_log = torch.digamma(alpha) - torch.digamma(alpha_sum)
    return (log_normalizer + ((alpha - beta) * expected_log).sum(-1, keepdim=True)).squeeze(-1)


def evidential_loss(alpha, labels, kl_weight):
    """Expected cross entropy with incorrect-evidence KL regularization."""
    one_hot = F.one_hot(labels, alpha.shape[-1]).to(alpha.dtype)
    expected_ce = torch.digamma(alpha.sum(-1)) - torch.digamma(alpha.gather(1, labels[:, None]).squeeze(1))
    adjusted_alpha = one_hot + (1 - one_hot) * alpha
    kl = dirichlet_kl(adjusted_alpha, torch.ones_like(alpha))
    return expected_ce.mean() + kl_weight * kl.mean(), expected_ce.mean(), kl.mean()


def prototype_regularizer(distances, labels, margin=1.0):
    """Within-class squared distance plus incorrect-class hinge repulsion."""
    known_distance = distances.gather(1, labels[:, None]).squeeze(1)
    incorrect = 1 - F.one_hot(labels, distances.shape[-1]).to(distances.dtype)
    repulsion = (F.relu(margin - distances) * incorrect).sum(-1) / (distances.shape[-1] - 1)
    return (known_distance + repulsion).mean()


def symmetric_dirichlet_kl(alpha, perturbed_alpha):
    """Symmetric Dirichlet KL; gradients propagate through both views."""
    return 0.5 * (
        dirichlet_kl(alpha, perturbed_alpha)
        + dirichlet_kl(perturbed_alpha, alpha)
    ).mean()


def compute_loss(model, output, images, masks, labels, config, epoch):
    """Compute the configured objective; epoch is one-based for KL annealing."""
    method = config.get("method", "sgep")
    stats = {}
    if method in ("sgep", "edl"):
        ramp_epochs = int(config.get("kl_ramp_epochs", 10))
        ramp = min(1.0, max(0.0, (epoch - 1) / ramp_epochs)) if ramp_epochs > 0 else 1.0
        kl_weight = float(config.get("lambda_kl_max", 0.01)) * ramp
        loss, expected_ce, kl = evidential_loss(output["alpha"], labels, kl_weight)
        stats.update(edl_ce=float(expected_ce.detach()), edl_kl=float(kl.detach()), kl_weight=kl_weight)
    elif method == "cac":
        tuplet = F.cross_entropy(-output["distances"], labels)
        anchor = output["distances"].gather(1, labels[:, None]).mean()
        loss = tuplet + float(config.get("lambda_cac", 0.1)) * anchor
        stats.update(tuplet=float(tuplet.detach()), anchor=float(anchor.detach()))
    elif method == "arpl":
        classification = F.cross_entropy(output["logits"] / model.arpl_temperature, labels)
        known_distance = output["distances"].gather(1, labels[:, None]).squeeze(1)
        reciprocal = F.relu(known_distance - model.radius + 1.0).mean()
        loss = classification + float(config.get("arpl_weight", 0.1)) * reciprocal
        stats.update(cross_entropy=float(classification.detach()), reciprocal=float(reciprocal.detach()))
    else:
        loss = F.cross_entropy(output["logits"], labels)
        stats["cross_entropy"] = float(loss.detach())

    proto_weight = float(config.get("lambda_proto", 0.1 if method == "sgep" else 0.0))
    if proto_weight > 0 and method in ("sgep", "prototype", "dual_ce"):
        proto = prototype_regularizer(output["distances"], labels, float(config.get("margin", 1.0)))
        loss = loss + proto_weight * proto
        stats["prototype"] = float(proto.detach())
    if method == "sgep":
        roi_weight = float(config.get("lambda_roi", 0.05))
        background_weight = float(config.get("lambda_bg", 0.05))
        if roi_weight > 0:
            from .data import perturb_masks

            perturbed_masks, fallback_count = perturb_masks(masks)
            perturbed = model(images, perturbed_masks)
            roi = symmetric_dirichlet_kl(output["alpha"], perturbed["alpha"])
            loss = loss + roi_weight * roi
            stats.update(roi=float(roi.detach()), mask_fallback_count=int(fallback_count))
        if background_weight > 0:
            background_mask = 1 - masks
            background = model(model.mask_image(images, background_mask), background_mask)
            background_loss = torch.log1p(background["evidence"].sum(-1)).mean()
            loss = loss + background_weight * background_loss
            stats["background"] = float(background_loss.detach())
    stats["loss"] = float(loss.detach())
    return loss, stats
