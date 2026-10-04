"""Grouped image manifests and synchronized RGB/ROI transforms."""

from pathlib import Path, PurePosixPath
import zlib

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def read_manifest(manifest):
    """Read a CSV without converting provenance identifiers to numbers."""
    if isinstance(manifest, pd.DataFrame):
        return manifest.copy().fillna("")
    return pd.read_csv(manifest, dtype=str, keep_default_na=False)


def relative_file(root, value):
    """Resolve a manifest-relative path inside its declared data root."""
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Expected a relative path inside the data root: {value!r}")
    root = Path(root).resolve()
    target = (root / Path(*path.parts)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path leaves the data root: {value!r}")
    return target


def mask_control(mask, policy, image_id, seed=2026):
    """Return an area-preserving spatial control for one binary HxW mask."""
    mask = np.asarray(mask, dtype=bool)
    if policy == "aligned":
        return mask.copy()
    if policy == "random_area":
        image_seed = seed + zlib.crc32(str(image_id).encode("utf-8"))
        rng = np.random.default_rng(image_seed)
        result = np.zeros(mask.size, dtype=bool)
        result[rng.choice(mask.size, size=int(mask.sum()), replace=False)] = True
        return result.reshape(mask.shape)
    if policy == "displaced":
        height, width = mask.shape
        shifts = sorted(
            (dy, dx)
            for dy in (-(height // 4), 0, height // 4)
            for dx in (-(width // 4), 0, width // 4)
            if (dy, dx) != (0, 0)
        )
        candidates = [np.roll(mask, shift, axis=(0, 1)) for shift in shifts]
        # argmin returns the first entry; shifts are lexicographic (dy, dx).
        overlaps = [int(np.logical_and(mask, candidate).sum())
                    for candidate in candidates]
        return candidates[int(np.argmin(overlaps))]
    raise ValueError(f"Unknown mask policy: {policy}")


class ImageDataset(Dataset):
    """Load RGB inputs in [0,1] and their automatic, binary ROI masks.

    Normalization and fill are applied by the recognition model. Reference
    masks, when supplied in the manifest, are never used as recognition inputs.
    """

    def __init__(self, manifest, root, split, classes, image_size=224,
                 augment=False, mask_policy="aligned", fill="black",
                 mask_seed=2026):
        self.manifest = read_manifest(manifest)
        required = {"image_id", "relative_path", "mask_path", "mapped_class",
                    "group_id", "category", "role", "split"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError("Missing manifest columns: " + ", ".join(sorted(missing)))
        if image_size < 4:
            raise ValueError("image_size must be at least 4 pixels.")
        if mask_policy not in {"aligned", "random_area", "displaced"}:
            raise ValueError(f"Unknown mask policy: {mask_policy}")
        if fill not in {"black", "mean"}:
            raise ValueError("fill must be black or mean.")
        if len(classes) != len(set(classes)):
            raise ValueError("classes must contain unique known class names.")
        self.frame = self.manifest.loc[self.manifest.split.eq(split)].reset_index(drop=True)
        self.root = Path(root)
        self.classes = list(classes)
        self.class_to_index = {name: i for i, name in enumerate(classes)}
        self.image_size = int(image_size)
        self.augment = augment
        self.mask_policy = mask_policy
        self.fill = fill
        self.mask_seed = int(mask_seed)
        invalid = self.frame.role.eq("known") & ~self.frame.mapped_class.isin(classes)
        if invalid.any():
            raise ValueError("Known manifest classes are absent from classes: "
                             + ", ".join(sorted(set(self.frame.loc[invalid, "mapped_class"]))))
        if self.frame.mask_path.eq("").any():
            raise ValueError("Automatic masks are missing. Run the masks command first.")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(relative_file(self.root, row.relative_path)) as source:
            image = source.convert("RGB")
        with Image.open(relative_file(self.root, row.mask_path)) as source:
            mask = source.convert("L")
        if mask.size != image.size:
            raise ValueError(f"Image and ROI dimensions differ for {row.image_id}.")
        native_binary = np.asarray(mask) > 0
        dice = float("nan")
        if "reference_mask_path" in row.index and row.reference_mask_path:
            if row.mask_path == row.reference_mask_path:
                raise ValueError("Reference masks are evaluation-only; provide an automatic mask_path.")
            with Image.open(relative_file(self.root, row.reference_mask_path)) as source:
                reference = source.convert("L")
            if reference.size != mask.size:
                raise ValueError(f"Reference and automatic ROI dimensions differ for {row.image_id}.")
            reference = np.asarray(reference) > 0
            denominator = int(native_binary.sum()) + int(reference.sum())
            if denominator:
                dice = 2 * int(np.logical_and(native_binary, reference).sum()) / denominator
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        binary = np.asarray(mask) > 0
        if not binary.any():
            raise ValueError(f"Empty input ROI for {row.image_id}; regenerate automatic masks.")
        binary = mask_control(binary, self.mask_policy, row.image_id, self.mask_seed)
        mask = Image.fromarray(binary.astype(np.uint8) * 255)
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            angle = torch.empty(()).uniform_(-15, 15).item()
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            brightness = torch.empty(()).uniform_(0.8, 1.2).item()
            contrast = torch.empty(()).uniform_(0.8, 1.2).item()
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255
        mask_tensor = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        label = self.class_to_index[row.mapped_class] if row.role == "known" else -1
        return {"image": image_tensor, "mask": mask_tensor, "label": label,
                "image_id": row.image_id, "group_id": row.group_id,
                "category": row.category, "role": row.role, "dice": dice}


def perturb_masks(masks, mode=None, pixels=5):
    """Perturb ROIs with 3x3 morphology or zero-padded integer translation.

    A random mode is chosen independently per sample when mode is None. Empty
    results revert to their original ROI, and the returned count records them.
    """
    if masks.ndim not in (3, 4) or masks.shape[-3] != 1:
        raise ValueError("masks must have shape 1xHxW or Nx1xHxW.")
    if mode not in {None, "erosion", "dilation", "translation"}:
        raise ValueError(f"Unknown ROI perturbation: {mode}")
    if pixels < 0:
        raise ValueError("pixels must be nonnegative.")
    original = masks.unsqueeze(0) if masks.ndim == 3 else masks
    values = original.float()
    result = values.clone()
    modes = ("erosion", "dilation", "translation")
    for index in range(len(values)):
        selected = mode if mode is not None else modes[torch.randint(3, ()).item()]
        item = values[index:index + 1]
        if selected == "erosion":
            result[index:index + 1] = -F.max_pool2d(-F.pad(item, (1, 1, 1, 1)), 3, stride=1)
        elif selected == "dilation":
            result[index:index + 1] = F.max_pool2d(item, 3, stride=1, padding=1)
        else:
            dy, dx = torch.randint(-pixels, pixels + 1, (2,)).tolist()
            height, width = item.shape[-2:]
            result[index].zero_()
            dst_y0, dst_y1 = max(dy, 0), min(height + dy, height)
            dst_x0, dst_x1 = max(dx, 0), min(width + dx, width)
            if dst_y1 > dst_y0 and dst_x1 > dst_x0:
                result[index, :, dst_y0:dst_y1, dst_x0:dst_x1] = values[
                    index, :, dst_y0 - dy:dst_y1 - dy, dst_x0 - dx:dst_x1 - dx]
    empty = result.flatten(1).sum(1).eq(0)
    result[empty] = values[empty]
    result = result.to(dtype=masks.dtype)
    return (result[0] if masks.ndim == 3 else result), int(empty.sum().item())
