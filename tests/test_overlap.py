"""Actual pixel similarity and provenance-group exclusion behavior."""

import numpy as np
import pandas as pd
from PIL import Image

from sgep.cli import main
from sgep.overlap import audit_external, image_signature


def image(root, name, seed):
    values = np.random.default_rng(seed).integers(10, 220, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(values).save(root / name)
    return values


def row(image_id, group, path, dataset="Source", split="test", reason=""):
    return {"dataset": dataset, "image_id": image_id, "group_id": group,
            "relative_path": path, "split": split, "exclusion_reason": reason}


def test_decoded_duplicate_excludes_whole_group_only_inside_its_dataset(tmp_path):
    values = image(tmp_path, "source.png", 1)
    # Different filename and encoding, identical recognition input pixels.
    Image.fromarray(values).save(tmp_path / "renamed.bmp")
    image(tmp_path, "sibling.png", 2)
    image(tmp_path, "independent.png", 3)
    source = pd.DataFrame([row("source", "fit-group", "source.png", split="train")])
    external = pd.DataFrame([
        row("duplicate", "shared-id", "renamed.bmp", "External-A"),
        row("sibling", "shared-id", "sibling.png", "External-A"),
        row("independent", "shared-id", "independent.png", "External-B"),
    ])
    audited = audit_external(source, external, tmp_path, tmp_path,
                             tmp_path / "audited.csv", near_distance=0).set_index("image_id")
    assert audited.at["duplicate", "split"] == "excluded"
    assert "decoded-pixel duplicate" in audited.at["duplicate", "overlap_reason"]
    assert audited.at["duplicate", "nearest_source_image"] == "source"
    assert audited.at["sibling", "split"] == "excluded"
    assert "group linked" in audited.at["sibling", "overlap_reason"]
    assert audited.at["independent", "split"] == "test"


def test_provider_linkage_from_taxonomy_excluded_frame_still_propagates(tmp_path):
    image(tmp_path, "source.png", 4)
    image(tmp_path, "retained-frame.png", 5)
    source = pd.DataFrame([row("source", "patient-42", "source.png", split="selection")])
    excluded = row("quality-frame", "procedure-8", "unused-quality.png", "External",
                   "excluded", "quality label outside source taxonomy")
    excluded["linked_source_group_id"] = "patient-42"
    external = pd.DataFrame([excluded, row("other-frame", "procedure-8", "retained-frame.png", "External")])
    audited = audit_external(source, external, tmp_path, tmp_path,
                             tmp_path / "audited.csv", near_distance=0).set_index("image_id")
    assert "quality label" in audited.at["quality-frame", "exclusion_reason"]
    assert "supplied acquisition/patient linkage" in audited.at["quality-frame", "overlap_reason"]
    assert audited.at["other-frame", "split"] == "excluded"
    assert "group linked" in audited.at["other-frame", "exclusion_reason"]


def test_perceptual_similarity_excludes_nonidentical_candidate_and_preserves_test_only_match(tmp_path):
    values = image(tmp_path, "source.png", 6)
    Image.fromarray(values + 5).save(tmp_path / "brightened.png")
    first_identity, first_signature = image_signature(tmp_path / "source.png")
    second_identity, second_signature = image_signature(tmp_path / "brightened.png")
    assert first_identity != second_identity
    distance = int(np.count_nonzero(first_signature != second_signature))
    assert distance <= 6
    test_values = image(tmp_path, "source-test.png", 7)
    Image.fromarray(test_values).save(tmp_path / "external-test.png")
    source = pd.DataFrame([row("source", "fit-group", "source.png", split="calibration"),
                           row("source-test", "test-group", "source-test.png", split="test")])
    external = pd.DataFrame([row("near", "near-group", "brightened.png", "External"),
                             row("test-match", "test-match-group", "external-test.png", "External")])
    audited = audit_external(source, external, tmp_path, tmp_path,
                             tmp_path / "audited.csv", near_distance=6).set_index("image_id")
    assert audited.at["near", "split"] == "excluded"
    assert "near-duplicate candidate" in audited.at["near", "overlap_reason"]
    assert int(audited.at["near", "perceptual_distance"]) == distance
    assert audited.at["test-match", "split"] == "test"


def test_audit_external_cli_applies_real_group_linkage(tmp_path):
    image(tmp_path, "source.png", 8)
    image(tmp_path, "external.png", 9)
    source = pd.DataFrame([row("source", "patient-a", "source.png", split="train")])
    external = pd.DataFrame([row("external", "patient-b", "external.png", "External")])
    external["linked_source_group_id"] = "patient-a"
    source.to_csv(tmp_path / "source.csv", index=False)
    external.to_csv(tmp_path / "external.csv", index=False)
    main(["audit-external", "--source", str(tmp_path / "source.csv"),
          "--external", str(tmp_path / "external.csv"), "--source-root", str(tmp_path),
          "--external-root", str(tmp_path), "--output", str(tmp_path / "output.csv"),
          "--near-distance", "0"])
    audited = pd.read_csv(tmp_path / "output.csv")
    assert audited.iloc[0].split == "excluded"
    assert "supplied acquisition/patient linkage" in audited.iloc[0].overlap_reason
