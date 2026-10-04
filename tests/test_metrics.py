"""Exact small fixtures for metric conventions and paired statistical behavior."""

import numpy as np
import pandas as pd
import pytest

from sgep.analysis import (annotation_summary, evidence_diagnostics, factorial_effects,
                           holm_adjust, mask_quality_analysis, paired_cluster_bootstrap,
                           paired_summary, subset_predictions, threshold_sensitivity)
from sgep.metrics import (calibrate_threshold, calibration_metrics, evaluate_predictions,
                          oscr, oscr_curve, reliability_bins, threshold_at_coverage)


@pytest.fixture
def predictions():
    return {
        "probabilities": np.array([[.9, .1], [.7, .3], [.4, .6], [.2, .8], [.6, .4], [.3, .7]]),
        "uncertainty": np.array([.1, .4, .4, .8, .4, .7]),
        "labels": np.array([0, 1, 1, 1, -1, -1]),
        "role": np.array(["known"] * 4 + ["unknown"] * 2),
        "category": np.array(["nv", "bkl", "nv", "bkl", "mel", "akiec"]),
        "image_id": np.array(["a", "b", "c", "d", "e", "f"]),
        "group_id": np.array(["lesion1", "lesion2", "lesion3", "lesion4", "lesion5", "lesion6"]),
    }


def test_tied_oscr_and_unknown_positive_auroc(predictions):
    metrics = evaluate_predictions(predictions, threshold=.4)
    assert metrics["known_accuracy"] == pytest.approx(75)
    assert metrics["auroc"] == pytest.approx(62.5)
    assert metrics["oscr"] == pytest.approx(43.75)
    assert metrics["known_coverage"] == pytest.approx(75)
    assert metrics["accepted_known_accuracy"] == pytest.approx(200 / 3)
    assert metrics["ccr"] == pytest.approx(50)
    assert metrics["unknown_fpr"] == pytest.approx(50)
    assert metrics["j"] == pytest.approx(0)
    assert metrics["fpr95"] == pytest.approx(100)
    assert metrics["fpr95_threshold"] == .8
    fpr, ccr, thresholds = oscr_curve([.1, .4, .4, .8], [.4, .7], [1, 0, 1, 1])
    np.testing.assert_allclose(fpr, [0, 0, .5, 1, 1])
    np.testing.assert_allclose(ccr, [0, .25, .5, .5, .75])
    np.testing.assert_allclose(thresholds, [-np.inf, .1, .4, .7, .8])
    assert oscr([.5] * 4, [.5] * 2, [1, 0, 1, 1]) == pytest.approx(.375)


def test_coverage_threshold_accepts_all_ties():
    known = np.array([.1] * 18 + [.2, .3])
    assert threshold_at_coverage(known, .95) == .2
    assert np.mean(known <= threshold_at_coverage(known, .5)) == pytest.approx(.9)
    data = {"probabilities": np.tile([.8, .2], (23, 1)),
            "labels": np.r_[np.zeros(20, dtype=int), [-1, -1, -1]],
            "uncertainty": np.r_[known, [.2, .21, .9]]}
    result = evaluate_predictions(data, .1)
    assert result["fpr95"] == pytest.approx(100 / 3)
    assert result["unknown_fpr"] == 0
    assert result["known_coverage"] == 90


def test_calibration_observed_scores_and_smaller_youden_tie(predictions):
    calibration = predictions | {"role": np.array(["known"] * 4 + ["proxy"] * 2)}
    assert calibrate_threshold(calibration, "coverage95") == .8
    assert calibrate_threshold(calibration, "coverage50") == .4
    assert calibrate_threshold(calibration, "youden") == .1
    tie = {"probabilities": np.tile([.8, .2], (4, 1)),
           "labels": np.array([0, 0, -1, -1]), "uncertainty": np.array([.1, .2, .15, .3])}
    assert calibrate_threshold(tie, "youden") == .1
    with pytest.raises(ValueError, match="Final unknown"):
        calibrate_threshold(predictions)
    # Changing final-test scores never changes the separately selected threshold.
    changed_test = predictions | {"uncertainty": predictions["uncertainty"] * 5}
    assert evaluate_predictions(changed_test, .8)["known_coverage"] == 25
    assert calibrate_threshold(calibration) == .8


def test_calibration_exact_ece_nll_brier_and_bin_boundaries(predictions):
    p, y = predictions["probabilities"][:4], predictions["labels"][:4]
    result = calibration_metrics(p, y)
    assert result["ece"] == pytest.approx(35)
    assert result["nll"] == pytest.approx(-np.log([.9, .3, .6, .8]).mean())
    assert result["brier"] == pytest.approx(.35)
    boundaries = reliability_bins(np.array([[.6, .4], [1., 0.]]), np.array([0, 1]))
    assert boundaries["count"][9] == 1
    assert boundaries["accuracy"][9] == 1
    assert boundaries["confidence"][9] == pytest.approx(.6)
    assert boundaries["count"][14] == 1
    assert boundaries["accuracy"][14] == 0
    assert calibration_metrics([[0., 1.]], [0])["nll"] == pytest.approx(-np.log(1e-12))


def test_no_population_does_not_invent_metrics(predictions):
    known = subset_predictions(predictions, predictions["labels"] >= 0)
    result = evaluate_predictions(known, .05)
    assert result["known_accuracy"] == 75
    assert result["known_coverage"] == 0
    assert result["ccr"] == 0
    assert result["accepted_known_accuracy"] is None
    for metric in ("auroc", "oscr", "fpr95", "unknown_fpr"):
        assert result[metric] is None
    unknown = subset_predictions(predictions, predictions["labels"] < 0)
    assert evaluate_predictions(unknown, .4)["unknown_fpr"] == 50
    assert evaluate_predictions(unknown, .4)["known_accuracy"] is None
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_predictions(known | {"probabilities": known["probabilities"] * .5})


def test_student_t_interval_and_holm_values():
    summary = paired_summary(np.zeros(5), np.arange(1., 6.))
    assert summary["difference"] == 3
    assert summary["difference_sd"] == pytest.approx(np.sqrt(2.5))
    assert summary["t"] == pytest.approx(4.242640687119285)
    assert summary["df"] == 4
    half = 2.7764451051977987 * np.sqrt(.5)
    assert summary["ci_low"] == pytest.approx(3 - half)
    assert summary["ci_high"] == pytest.approx(3 + half)
    assert summary["p_two_sided"] == pytest.approx(.013235599563682695)
    np.testing.assert_allclose(holm_adjust([.02, .001, .01, .04]), [.04, .004, .03, .04])
    constant = paired_summary([0, 0, 0], [2, 2, 2])
    assert constant["difference"] == 2
    assert constant["ci_low"] == constant["ci_high"] == 2
    assert constant["p_two_sided"] is None


def test_paired_cluster_bootstrap_preserves_mixed_groups_and_model_pairing():
    baseline = {"probabilities": np.tile([.9, .1], (4, 1)),
                "labels": np.array([0, 0, -1, -1]),
                "uncertainty": np.array([.8, .9, .1, .2]),
                "image_id": np.array(["a", "b", "c", "d"]),
                "group_id": np.array(["patient1", "patient2", "patient1", "patient2"])}
    proposed = baseline | {"uncertainty": np.array([.1, .2, .8, .9])}
    proposed = subset_predictions(proposed, [3, 1, 0, 2])
    report = paired_cluster_bootstrap(baseline, proposed, metrics=("auroc", "oscr", "fpr95"),
                                     n_resamples=32)
    assert report["strata"] == {"mixed": 2}
    for metric, difference in (("auroc", 100), ("oscr", 100), ("fpr95", -100)):
        result = report["metrics"][metric]
        assert result["difference"] == difference
        assert result["ci_low"] == result["ci_high"] == difference
    identical = paired_cluster_bootstrap(baseline, subset_predictions(baseline, [3, 1, 0, 2]),
                                        metrics=("auroc",), n_resamples=20)
    assert identical["metrics"]["auroc"]["ci_low"] == 0
    assert identical["metrics"]["auroc"]["ci_high"] == 0


def test_evidence_diagnostics_compute_uncertainty_before_averaging():
    evidence = np.array([[4., 2.], [1.5, .5]])
    data = {"probabilities": (evidence + 1) / (evidence.sum(axis=1, keepdims=True) + 2),
            "labels": np.array([0, 0]), "uncertainty": np.array([.25, .5]),
            "evidence": evidence, "background_evidence": np.array([[.7, .3], [.6, .4]]),
            "group_id": np.array(["lesion1", "lesion1"])}
    rows = evidence_diagnostics(data, .6, n_resamples=20).set_index("group")
    known = rows.loc["accepted_known"]
    assert known["evidence"] == 4
    assert known["uncertainty"] == pytest.approx(.375)
    assert known["uncertainty"] != pytest.approx(2 / (known["evidence"] + 2))
    assert known["rho"] == pytest.approx((6 / 7 + 2 / 3) / 2, abs=1e-7)
    assert known["delta_e"] == pytest.approx((2 / 6 + 1 / 2) / 2, abs=1e-7)
    assert known["uncertainty_ci_low"] == known["uncertainty_ci_high"] == .375
    assert rows.loc["background_only", "evidence"] == 1
    assert rows.loc["background_only", "uncertainty"] == pytest.approx(2 / 3)
    rejected = evidence_diagnostics(data, .3, n_resamples=20).set_index("group")
    assert rejected.loc["rejected_known", "uncertainty"] == .5
    assert rejected.loc["rejected_known", "uncertainty"] > .3


def test_mask_quality_strata_and_annotation_denominators(predictions):
    quality = pd.DataFrame({"image_id": ["a", "b", "c", "e", "f"],
                            "dice": [.85, .65, .64, .9, np.nan],
                            "annotator_dice": [.9, .8, .7, .95, .6],
                            "adjudicated": [0, 1, 0, 1, 0]})
    strata = mask_quality_analysis(predictions, .4, quality).set_index("stratum")
    assert strata.loc["high", "known_accuracy"] == 100
    assert strata.loc["high", "auroc"] == 100
    assert strata.loc["medium", "known_accuracy"] == 0
    assert strata.loc["medium", "auroc"] is None or np.isnan(strata.loc["medium", "auroc"])
    assert strata.loc["low", "dice_mean"] == .64
    assert strata.loc["high", "n_adjudicated"] == 1
    summary = annotation_summary(predictions, quality)
    overall = summary[summary.field == "all"].iloc[0]
    assert overall.n_eligible_images == 6
    assert overall.n_scored_images == 4
    assert overall.reference_coverage == pytest.approx(200 / 3)
    assert overall.n_adjudicated == 2
    assert overall.inter_annotator_dice_mean == pytest.approx(.79)


def test_threshold_sensitivity_uses_held_out_calibration(predictions):
    calibration = predictions | {"role": np.array(["known"] * 4 + ["proxy"] * 2)}
    table = threshold_sensitivity(predictions, calibration).set_index("rule")
    assert table.loc["coverage95", "threshold"] == .8
    assert table.loc["youden", "threshold"] == .1
    np.testing.assert_allclose(table["oscr"], [43.75] * 4)
    assert table.loc["youden", "ccr"] == 25
    assert table.loc["youden", "unknown_fpr"] == 0


def test_factorial_paired_contrasts():
    records = []
    for seed in range(5):
        for roi in (0, 1):
            for bg in (0, 1):
                value = 10 + seed + 4 * roi + 2 * bg + seed * roi * bg
                records.append({"dataset": "tiny", "seed": seed, "roi": roi, "bg": bg,
                                "known_accuracy": value})
    table = factorial_effects(records).set_index("effect")
    assert table.loc["roi_main", "difference"] == 5
    assert table.loc["background_main", "difference"] == 3
    assert table.loc["interaction", "difference"] == 2
    assert table.loc["interaction", "difference_sd"] == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError, match="all four"):
        factorial_effects(records[:-1])
