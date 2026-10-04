"""Assertions for provenance partitioning and geometric mask controls."""

from unittest.mock import patch
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from sgep.data import ImageDataset, mask_control, perturb_masks, relative_file
from sgep.prepare import (
    evaluate_annotation_quality, make_split, prepare_external, select_sam_mask,
)
from sgep.cli import main


def metadata_row(image_id, group_id, category="nv", dataset="HAM10000"):
    return {"dataset": dataset, "release": "test-release", "image_id": image_id,
            "relative_path": f"{image_id}.png", "category": category,
            "group_id": group_id, "group_type": "lesion",
            "source_cohort": "test-cohort", "mask_path": ""}


def test_group_roles_preserve_unknown_test_and_exclude_linked_proxy():
    rows = [metadata_row(f"known-{i}", f"known-group-{i}") for i in range(8)]
    rows += [metadata_row("second-view", "known-group-0")]
    rows += [metadata_row(f"proxy-{i}", f"proxy-group-{i}", "df") for i in range(3)]
    rows += [metadata_row("linked-known", "final-group"),
             metadata_row("linked-unknown", "final-group", "mel"),
             metadata_row("linked-proxy", "final-group", "df")]
    frame = make_split(pd.DataFrame(rows))
    included = frame.loc[frame.split.ne("excluded")]
    assert included.groupby(["dataset", "group_id"]).split.nunique().eq(1).all()
    assert set(frame.loc[frame.category.eq("nv"), "split"]) == {
        "train", "selection", "calibration", "test"}
    assert frame.loc[frame.role.eq("unknown"), "split"].eq("test").all()
    assert frame.loc[frame.split.eq("train"), "role"].eq("known").all()
    assert frame.set_index("image_id").at["linked-known", "split"] == "test"
    assert frame.set_index("image_id").at["linked-proxy", "split"] == "excluded"
    assert "final-unknown" in frame.set_index("image_id").at["linked-proxy", "exclusion_reason"]
    pd.testing.assert_frame_equal(frame, make_split(pd.DataFrame(rows)))


def test_gastrovision_requires_actual_sequence_or_patient_linkage():
    metadata = pd.DataFrame([metadata_row("frame", "random-file-name", "Colon polyps", "GastroVision")])
    with pytest.raises(ValueError, match="genuine patient/procedure/sequence"):
        make_split(metadata)


def test_external_crosswalk_keeps_semantic_roles_and_provenance_exclusions(tmp_path):
    rows = [metadata_row("common", "p1", "common nevi", "PH2"),
            metadata_row("atypical", "p2", "atypical nevi", "PH2"),
            metadata_row("melanoma", "p3", "melanoma", "PH2")]
    rows[1]["exclusion_reason"] = "provider-reported patient overlap"
    frame = prepare_external(pd.DataFrame(rows), tmp_path / "external.csv").set_index("image_id")
    assert frame.at["common", "mapped_class"] == "nv"
    assert frame.at["atypical", "mapped_class"] == "nv"
    assert frame.at["common", "role"] == "known"
    assert frame.at["melanoma", "role"] == "unknown"
    assert frame.at["atypical", "split"] == "excluded"
    assert frame.at["melanoma", "split"] == "test"


def test_random_area_is_fixed_per_image_and_exactly_area_matched():
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:8, 4:10] = True
    a = mask_control(mask, "random_area", "image-a")
    assert a.sum() == mask.sum()
    assert np.array_equal(a, mask_control(mask, "random_area", "image-a"))
    assert not np.array_equal(a, mask_control(mask, "random_area", "image-b"))


def test_displaced_control_chooses_minimum_overlap_and_lexicographic_tie():
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True
    shifts = sorted((dy, dx) for dy in (-3, 0, 3) for dx in (-3, 0, 3) if (dy, dx) != (0, 0))
    candidates = [np.roll(mask, shift, axis=(0, 1)) for shift in shifts]
    overlap = [np.logical_and(mask, candidate).sum() for candidate in candidates]
    expected = candidates[int(np.argmin(overlap))]
    actual = mask_control(mask, "displaced", "any-image")
    assert actual.sum() == mask.sum()
    assert np.array_equal(actual, expected)


def test_roi_morphology_geometry_and_empty_fallback():
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 3:6, 3:6] = 1
    eroded, fallbacks = perturb_masks(mask, "erosion")
    expected = torch.zeros_like(mask)
    expected[:, :, 4, 4] = 1
    assert torch.equal(eroded, expected)
    assert fallbacks == 0
    dilated, fallbacks = perturb_masks(mask, "dilation")
    expected.zero_()
    expected[:, :, 2:7, 2:7] = 1
    assert torch.equal(dilated, expected)
    assert fallbacks == 0
    tiny = torch.zeros_like(mask)
    tiny[:, :, 4, 4] = 1
    restored, fallbacks = perturb_masks(tiny, "erosion")
    assert torch.equal(restored, tiny)
    assert fallbacks == 1


def test_translation_uses_integer_zero_padding_and_reverts_empty():
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, 1, 1] = 1
    with patch("sgep.data.torch.randint", return_value=torch.tensor([2, -1])):
        translated, fallbacks = perturb_masks(mask, "translation")
    expected = torch.zeros_like(mask)
    expected[:, :, 3, 0] = 1
    assert torch.equal(translated, expected)
    assert fallbacks == 0
    with patch("sgep.data.torch.randint", return_value=torch.tensor([-5, -5])):
        restored, fallbacks = perturb_masks(mask, "translation")
    assert torch.equal(restored, mask)
    assert fallbacks == 1


def test_image_roi_transforms_are_synchronized_and_reference_is_evaluation_only(tmp_path):
    roi = np.zeros((8, 8), dtype=np.uint8)
    roi[1:5, 1:3] = 255
    rgb = np.repeat(roi[:, :, None], 3, axis=2)
    Image.fromarray(rgb).save(tmp_path / "image.png")
    Image.fromarray(roi).save(tmp_path / "automatic.png")
    Image.fromarray(np.fliplr(roi).copy()).save(tmp_path / "reference.png")
    row = metadata_row("image", "lesion")
    row.update(mask_path="automatic.png", reference_mask_path="reference.png",
               mapped_class="nv", role="known", split="test")
    dataset = ImageDataset(pd.DataFrame([row]), tmp_path, "test", ["nv"], image_size=8, augment=True)

    def midpoint(tensor, low, high):
        return tensor.fill_((low + high) / 2)

    with patch("sgep.data.torch.rand", return_value=torch.tensor(0.0)), \
            patch("torch.Tensor.uniform_", midpoint):
        sample = dataset[0]
    expected = torch.from_numpy(np.fliplr(roi).copy()).float() / 255
    assert torch.equal(sample["mask"][0], expected)
    assert torch.equal(sample["image"][0], expected)
    assert sample["label"] == 0
    assert sample["dice"] == 0.0
    assert set(sample["mask"].unique().tolist()) == {0.0, 1.0}


def test_sam_selection_uses_quality_area_and_generator_order():
    shape = (10, 10)
    masks = [np.zeros(shape, dtype=bool) for _ in range(4)]
    for mask, count in zip(masks, (4, 25, 20, 20)):
        mask.flat[:count] = True
    candidates = [{"area": int(mask.sum()), "predicted_iou": 0.9,
                   "stability_score": 0.95, "segmentation": mask} for mask in masks]
    chosen, fallback, index = select_sam_mask(candidates, shape)
    assert index == 2
    assert not fallback
    assert np.array_equal(chosen, masks[2])
    chosen, fallback, index = select_sam_mask([candidates[0]], shape)
    assert chosen.all()
    assert fallback and index == -1


def test_manifest_paths_cannot_leave_the_data_root(tmp_path):
    with pytest.raises(ValueError, match="relative path"):
        relative_file(tmp_path, "../outside.png")
    with pytest.raises(ValueError, match="relative path"):
        relative_file(tmp_path, "C:/private.png")


def test_native_mask_quality_is_independent_of_model_image_size(tmp_path):
    roi = np.zeros((16, 16), dtype=np.uint8)
    roi[2:14, 2:14] = 255
    reference = roi.copy()
    reference[3, 3] = 0
    Image.fromarray(np.repeat(roi[:, :, None], 3, axis=2)).save(tmp_path / "image.png")
    Image.fromarray(roi).save(tmp_path / "automatic.png")
    Image.fromarray(reference).save(tmp_path / "reference.png")
    row = metadata_row("image", "lesion")
    row.update(mask_path="automatic.png", reference_mask_path="reference.png",
               mapped_class="nv", role="known", split="test")
    manifest = pd.DataFrame([row])
    large = ImageDataset(manifest, tmp_path, "test", ["nv"], image_size=16)[0]
    small = ImageDataset(manifest, tmp_path, "test", ["nv"], image_size=4)[0]
    expected_dice = 2 * 143 / (144 + 143)
    assert large["dice"] == pytest.approx(expected_dice)
    assert small["dice"] == pytest.approx(expected_dice)
    assert small["dice"] < 1


def test_independent_annotation_quality_requires_adjudicated_disagreement(tmp_path):
    roi = np.zeros((8, 8), dtype=np.uint8)
    roi[2:6, 2:6] = 255
    reference = roi.copy()
    reference[2, 2] = 0
    Image.fromarray(np.repeat(roi[:, :, None], 3, axis=2)).save(tmp_path / "image.png")
    for path, values in (("automatic.png", roi), ("annotator_a.png", roi),
                         ("annotator_b.png", reference), ("reference.png", reference)):
        Image.fromarray(values).save(tmp_path / path)
    row = metadata_row("image", "lesion")
    row.update(mask_path="automatic.png", mapped_class="nv", role="known", split="test")
    annotation = {"dataset": "HAM10000", "image_id": "image",
                  "annotator_a_path": "annotator_a.png", "annotator_b_path": "annotator_b.png",
                  "reference_mask_path": "reference.png", "adjudicated": "false"}
    with pytest.raises(ValueError, match="disagreement needs adjudication"):
        evaluate_annotation_quality(pd.DataFrame([annotation]), pd.DataFrame([row]),
                                    tmp_path, tmp_path / "quality.csv")
    annotation["adjudicated"] = "true"
    quality = evaluate_annotation_quality(pd.DataFrame([annotation]), pd.DataFrame([row]),
                                         tmp_path, tmp_path / "quality.csv")
    assert quality.iloc[0].dice == pytest.approx(30 / 31)
    assert quality.iloc[0].annotator_dice == pytest.approx(30 / 31)
    assert quality.iloc[0].adjudicated
    assert quality.iloc[0].reference_foreground_pixels == 15


def test_sam_masks_cli_uses_fixed_settings_and_writes_selected_roi_pixels(tmp_path):
    calls = {}

    class Sam(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

    sam = Sam()

    def build(checkpoint):
        calls["checkpoint"] = checkpoint
        return sam

    class Generator:
        def __init__(self, model, **kwargs):
            calls["model"] = model
            calls["settings"] = kwargs
            self.index = 0

        def generate(self, rgb):
            assert rgb.shape == (10, 10, 3)
            assert rgb.dtype == np.uint8
            assert torch.is_inference_mode_enabled()
            counts = (4, 25, 20) if self.index == 0 else (4, 95)
            self.index += 1
            calls["generated_images"] = self.index
            candidates = []
            for count in counts:
                mask = np.zeros((10, 10), dtype=bool)
                mask.flat[:count] = True
                candidates.append({"segmentation": mask, "area": count,
                                   "predicted_iou": 0.9, "stability_score": 0.95})
            return candidates

    fake = SimpleNamespace(sam_model_registry={"vit_b": build}, SamAutomaticMaskGenerator=Generator)
    Image.fromarray(np.full((10, 10, 3), 100, dtype=np.uint8)).save(tmp_path / "first.png")
    Image.fromarray(np.full((10, 10, 3), 110, dtype=np.uint8)).save(tmp_path / "second.png")
    rows = [metadata_row("first", "lesion-a"), metadata_row("second", "lesion-b"),
            metadata_row("excluded", "lesion-c")]
    for row in rows:
        row.update(mapped_class="nv", role="known", split="test")
    rows[2]["split"] = "excluded"
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    checkpoint = tmp_path / "sam_vit_b_01ec64.pth"
    checkpoint.write_bytes(b"test checkpoint passed only to the SAM constructor seam")
    output = tmp_path / "masked.csv"
    with patch.dict(sys.modules, {"segment_anything": fake}):
        main(["masks", "--manifest", str(tmp_path / "manifest.csv"),
              "--root", str(tmp_path), "--checkpoint", str(checkpoint),
              "--device", "cpu", "--output", str(output)])
    assert calls["checkpoint"] == str(checkpoint)
    assert calls["model"] is sam
    assert not sam.training and not sam.weight.requires_grad
    assert sam.mask_threshold == 0.0
    assert calls["generated_images"] == 2
    assert calls["settings"] == {
        "points_per_side": 32, "points_per_batch": 64, "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95, "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7, "crop_n_layers": 0, "crop_nms_thresh": 0.7,
        "crop_overlap_ratio": 512 / 1500, "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 0,
    }
    prepared = pd.read_csv(output, keep_default_na=False).set_index("image_id")
    with Image.open(tmp_path / prepared.at["first", "mask_path"]) as source:
        selected = np.asarray(source) > 0
    expected = np.zeros((10, 10), dtype=bool)
    expected.flat[:20] = True
    assert np.array_equal(selected, expected)
    with Image.open(tmp_path / prepared.at["second", "mask_path"]) as source:
        assert (np.asarray(source) == 255).all()
    assert prepared.at["excluded", "mask_path"] == ""
    log = pd.read_csv(tmp_path / "masked_mask_log.csv").set_index("image_id")
    assert log.at["first", "selected_candidate"] == 2
    assert not log.at["first", "full_image_fallback"]
    assert log.at["second", "full_image_fallback"]
