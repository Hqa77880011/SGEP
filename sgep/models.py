"""SGEP and matched recognition heads with a common image/mask interface."""

import copy

import torch
from torch import nn
from torch.nn import functional as F


METHODS = (
    "sgep", "softmax", "maxlogit", "energy", "openmax", "postmax", "cac",
    "arpl", "edl", "prototype", "roi_guided", "dual_ce",
)


class TinyEncoder(nn.Sequential):
    """Small convolutional encoder for CPU interface checks."""

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )


def make_encoder(backbone, pretrained):
    """Construct a pooled feature extractor and its output dimension."""
    if backbone == "tiny":
        return TinyEncoder(), 32
    if backbone != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone}")
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    return encoder, 512


def prototype_evidence(features, prototypes, beta):
    """Map squared prototype distances to Dirichlet predictions."""
    distances = (features[:, None, :] - prototypes[None, :, :]).square().sum(-1)
    logits = beta - distances
    evidence = F.softplus(logits)
    alpha = evidence + 1
    strength = alpha.sum(-1)
    return {
        "features": features,
        "distances": distances,
        "logits": logits,
        "evidence": evidence,
        "alpha": alpha,
        "probabilities": alpha / strength[:, None],
        "uncertainty": prototypes.shape[0] / strength,
    }


class RecognitionModel(nn.Module):
    """Encode global/ROI views and apply the configured recognition head."""

    def __init__(self, config, num_classes):
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two known classes are required.")
        self.method = config.get("method", "sgep")
        if self.method not in METHODS:
            raise ValueError(f"Unsupported method: {self.method}")
        default_view = {
            "sgep": "dual", "roi_guided": "roi", "dual_ce": "dual",
        }.get(self.method, "global")
        self.view = config.get("view", default_view)
        if self.view not in ("global", "roi", "dual"):
            raise ValueError(f"Unsupported view: {self.view}")
        self.fill = config.get("fill", "black")
        if self.fill not in ("black", "mean"):
            raise ValueError(f"Unsupported fill: {self.fill}")
        self.num_classes = num_classes
        self.feature_dim = int(config.get("feature_dim", 256))
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        self.energy_temperature = float(config.get("energy_temperature", 1.0))
        self.arpl_temperature = float(config.get("arpl_temperature", 1.0))
        if min(self.energy_temperature, self.arpl_temperature) <= 0:
            raise ValueError("Score temperatures must be positive.")
        encoder, branch_dim = make_encoder(
            config.get("backbone", "resnet18"), config.get("pretrained", True),
        )
        if self.view == "roi":
            self.roi_encoder = encoder
        else:
            self.global_encoder = encoder
        if self.view == "dual":
            self.roi_encoder = copy.deepcopy(encoder)
        self.fusion = nn.Linear(
            branch_dim * (2 if self.view == "dual" else 1),
            self.feature_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.fusion.weight)
        self.normalized_features = (
            self.method in ("sgep", "prototype") or self.view == "dual"
        )
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

        has_aux_prototypes = self.method == "dual_ce" and config.get("lambda_proto", 0) > 0
        if self.method in ("sgep", "prototype") or has_aux_prototypes:
            self.prototypes = nn.Parameter(torch.randn(num_classes, self.feature_dim) * 0.1)
        if self.method == "sgep":
            self.beta = nn.Parameter(torch.tensor(float(config.get("beta_init", 3.0))))
        elif self.method == "arpl":
            self.reciprocal_points = nn.Parameter(torch.randn(num_classes, self.feature_dim) * 0.1)
            self.radius = nn.Parameter(torch.zeros(()))
        elif self.method != "prototype":
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        if self.method == "cac":
            magnitude = float(config.get("cac_anchor", 10.0))
            if magnitude <= 0:
                raise ValueError("cac_anchor must be positive.")
            self.register_buffer("anchors", magnitude * torch.eye(num_classes))

    def mask_image(self, images, masks):
        """Mask RGB [0,1] inputs before normalization, using the configured fill."""
        fill = self.image_mean if self.fill == "mean" else 0.0
        return images * masks + fill * (1 - masks)

    def encode(self, images, masks):
        """Return the fused representation for the configured image views."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W].")
        if masks.shape != (images.shape[0], 1, images.shape[2], images.shape[3]):
            raise ValueError("masks must have shape [B,1,H,W] matching images.")
        views = []
        if self.view in ("global", "dual"):
            views.append(self.global_encoder((images - self.image_mean) / self.image_std))
        if self.view in ("roi", "dual"):
            roi = self.mask_image(images, masks)
            views.append(self.roi_encoder((roi - self.image_mean) / self.image_std))
        features = self.fusion(torch.cat(views, dim=-1))
        if self.normalized_features:
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)
        return features

    def forward(self, images, masks):
        features = self.encode(images, masks)
        if self.method == "sgep":
            return prototype_evidence(features, self.prototypes, self.beta)
        output = {"features": features}
        if self.method == "prototype":
            distances = (features[:, None] - self.prototypes[None]).square().sum(-1)
            logits = -distances
            output["distances"] = distances
            uncertainty = distances.min(-1).values
        elif self.method == "arpl":
            # ARPL's distance combines mean squared L2 and an unnormalized dot product.
            distances = (features[:, None] - self.reciprocal_points[None]).square().mean(-1)
            logits = distances - features @ self.reciprocal_points.t()
            output["distances"] = distances
            uncertainty = -logits.max(-1).values
        else:
            logits = self.classifier(features)
            uncertainty = None
            if hasattr(self, "prototypes"):
                output["distances"] = (features[:, None] - self.prototypes[None]).square().sum(-1)
        output["logits"] = logits
        if self.method == "edl":
            evidence = F.softplus(logits)
            alpha = evidence + 1
            strength = alpha.sum(-1)
            output.update(evidence=evidence, alpha=alpha)
            probabilities = alpha / strength[:, None]
            uncertainty = self.num_classes / strength
        elif self.method == "cac":
            distances = torch.cdist(logits, self.anchors)
            probabilities = F.softmax(-distances, dim=-1)
            uncertainty = (distances * (1 - probabilities)).min(-1).values
            output["distances"] = distances
        else:
            temperature = self.arpl_temperature if self.method == "arpl" else 1.0
            probabilities = F.softmax(logits / temperature, dim=-1)
            if self.method == "maxlogit":
                uncertainty = -logits.max(-1).values
            elif self.method == "energy":
                temperature = self.energy_temperature
                uncertainty = -temperature * torch.logsumexp(logits / temperature, dim=-1)
            elif uncertainty is None:
                uncertainty = 1 - probabilities.max(-1).values
        output.update(probabilities=probabilities, uncertainty=uncertainty)
        return output


def build_model(config: dict, num_classes: int) -> nn.Module:
    """Build a recognition model from a JSON-compatible configuration."""
    return RecognitionModel(config, num_classes)


@torch.no_grad()
def initialize_prototypes(model, iterable, device):
    """Set trainable prototypes to unaugmented known-training class means."""
    if not hasattr(model, "prototypes"):
        return
    was_training = model.training
    model.eval()
    sums = torch.zeros_like(model.prototypes, device=device)
    counts = torch.zeros(model.num_classes, device=device)
    try:
        for batch in iterable:
            if isinstance(batch, dict):
                images, masks, labels = batch["image"], batch["mask"], batch["label"]
            else:
                images, masks, labels = batch[:3]
            labels = labels.to(device)
            features = model.encode(images.to(device), masks.to(device))
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
        if (counts == 0).any():
            missing = torch.where(counts == 0)[0].tolist()
            raise ValueError(f"Cannot initialize prototypes: missing known classes {missing}.")
        model.prototypes.copy_(sums / counts[:, None])
    finally:
        model.train(was_training)
