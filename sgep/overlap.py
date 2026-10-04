"""Image and supplied acquisition linkage checks for external manifests."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from .data import read_manifest, relative_file


def image_signature(path):
    """Decoded-pixel identity and a perceptual signature across filenames."""
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        identity = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).digest()
        small = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(small, norm="ortho")[:8, :8].ravel()[1:]
    return identity, low > np.median(low)


def audit_external(source, external, source_root, external_root, output, near_distance=6):
    """Exclude fitting-partition overlap; supplied group links override image similarity."""
    if not 0 <= near_distance <= 63:
        raise ValueError("near_distance must be between 0 and 63 signature bits.")
    source, external = read_manifest(source), read_manifest(external)
    required = {"dataset", "image_id", "group_id", "split", "relative_path"}
    for name, frame in (("source", source), ("external", external)):
        if not required.issubset(frame):
            raise ValueError(f"{name} manifest lacks columns: {sorted(required - set(frame))}")
        if frame[list(required)].astype(str).apply(lambda column: column.str.strip().eq("")).any().any():
            raise ValueError(f"{name} manifest requires nonempty image and group provenance.")
    fitting = source[source.split.isin(["train", "selection", "calibration"])]
    if fitting.empty:
        raise ValueError("Source manifest has no training/validation images.")
    identities, perceptual = {}, []
    for row in fitting.itertuples():
        identity, signature = image_signature(relative_file(source_root, row.relative_path))
        identities.setdefault(identity, []).append(row.image_id)
        perceptual.append(signature)
    perceptual = np.stack(perceptual)
    source_groups = set(zip(fitting.dataset, fitting.group_id))
    source_datasets = set(fitting.dataset)
    external = external.copy().reset_index(drop=True)
    if "exclusion_reason" not in external:
        external["exclusion_reason"] = ""
    external["overlap_reason"] = ""
    external["nearest_source_image"] = ""
    external["perceptual_distance"] = ""
    for index, row in external.iterrows():
        reason = ""
        linked_group = row.get("linked_source_group_id", "")
        if linked_group:
            linked_dataset = row.get("linked_source_dataset", "")
            if not linked_dataset:
                if len(source_datasets) != 1:
                    raise ValueError("Multiple source datasets require linked_source_dataset for supplied group links.")
                linked_dataset = next(iter(source_datasets))
            if (linked_dataset, linked_group) in source_groups:
                reason = "supplied acquisition/patient linkage to source fitting group"
        if row["split"] != "excluded":
            identity, signature = image_signature(relative_file(external_root, row["relative_path"]))
            distances = np.count_nonzero(perceptual != signature, axis=1)
            nearest = int(distances.argmin())
            external.loc[index, "nearest_source_image"] = fitting.iloc[nearest].image_id
            external.loc[index, "perceptual_distance"] = str(int(distances[nearest]))
            if not reason and identity in identities:
                reason = "decoded-pixel duplicate of source fitting image"
                external.loc[index, "nearest_source_image"] = identities[identity][0]
            elif not reason and distances[nearest] <= near_distance:
                reason = "perceptual near-duplicate candidate (excluded conservatively)"
        if reason:
            external.loc[index, "split"] = "excluded"
            prior_reason = str(row.get("exclusion_reason", ""))
            external.loc[index, "exclusion_reason"] = prior_reason + "; " + reason if prior_reason else reason
            external.loc[index, "overlap_reason"] = reason
    # A linked group is excluded as a unit, including other images from the group.
    flagged_rows = external.loc[external.overlap_reason.ne("")]
    flagged = set(zip(flagged_rows.dataset, flagged_rows.group_id))
    linked = pd.Series([(dataset, group) in flagged for dataset, group
                        in zip(external.dataset, external.group_id)], index=external.index)
    linked &= external.split.ne("excluded")
    external.loc[linked, "split"] = "excluded"
    external.loc[linked, "exclusion_reason"] = "group linked to excluded source-overlap image"
    external.loc[linked, "overlap_reason"] = "group linked to excluded source-overlap image"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    external.to_csv(output, index=False)
    return external


def select_reference_images(manifest, output, per_class=100, seed=2026):
    """Select HAM test images for independent annotation before reading model scores."""
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(seed)
    selected = []
    for category in ("nv", "bkl", "mel", "akiec"):
        eligible = frame[frame.dataset.eq("HAM10000") & frame.split.eq("test") & frame.category.eq(category)]
        count = min(per_class, len(eligible))
        selected.append(eligible.iloc[rng.choice(len(eligible), count, replace=False)])
    result = pd.concat(selected, ignore_index=True)
    if result.empty:
        raise ValueError("No eligible HAM10000 test images for reference annotation.")
    for column in ("annotator_a_path", "annotator_b_path", "reference_mask_path"):
        result[column] = ""
    result["adjudicated"] = ""
    result["annotation_selection_seed"] = seed
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
