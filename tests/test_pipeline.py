from sgep.config import epoch_lr, load_config
from sgep.smoke import run_smoke
from sgep.experiments import variants, search_candidates


def test_experiment_runner_preserves_calibration_and_variant_metadata(tmp_path):
    import json
    import numpy as np
    from sgep.analysis import aggregate_runs, load_predictions
    from sgep.experiments import run_suite, run_search
    from sgep.metrics import calibrate_threshold
    from sgep.smoke import create_fixture

    root = tmp_path / "data"
    manifest = create_fixture(root)
    config = load_config(overrides={"backbone": "tiny", "pretrained": False,
        "feature_dim": 8, "image_size": 32, "epochs": 1, "batch_size": 8,
        "threads": 1, "device": "cpu", "lambda_proto": 0.2})
    rows = run_suite(config, manifest, root, tmp_path / "factorial", "factorial", [11, 22])
    for row in rows:
        run = tmp_path / "factorial" / row["variant"] / f"seed_{row['seed']}"
        actual_config = json.loads((run / "config.json").read_text())
        assert actual_config["lambda_proto"] == 0.2
        assert actual_config["seed"] == row["seed"]
        calibration = load_predictions(run / "calibration_predictions.npz")
        threshold = json.loads((run / "threshold.json").read_text())["threshold"]
        assert threshold == calibrate_threshold(calibration)
        test = load_predictions(run / "test_predictions.npz")
        assert set(calibration["image_id"]).isdisjoint(test["image_id"])
        assert set(calibration["category"]) == {"a", "b", "c"}
        assert set(test["category"]) == {"a", "b", "d"}
    runs = [path.parent for path in (tmp_path / "factorial").rglob("metrics.json")]
    aggregate_runs(runs, tmp_path / "comparison")
    import pandas as pd
    factorial = pd.read_csv(tmp_path / "comparison" / "factorial_effects.csv")
    assert set(factorial["effect"]) == {"roi_main", "background_main", "interaction"}
    for metric, group in factorial.groupby("metric"):
        values = {variant: np.mean([row[metric] for row in rows if row["variant"] == variant])
                  for variant in ("roi0_bg0", "roi0_bg1", "roi1_bg0", "roi1_bg1")}
        expected = values["roi1_bg1"] - values["roi1_bg0"] - values["roi0_bg1"] + values["roi0_bg0"]
        actual = group.loc[group.effect.eq("interaction"), "difference"].iloc[0]
        assert abs(actual - expected) < 1e-8
    chosen = run_search(config, manifest, root, tmp_path / "search", count=1, seeds=[11])
    assert chosen["lambda_proto"] == 0.2
    log = json.loads((tmp_path / "search" / "search_runs.json").read_text())
    assert log[0]["status"] == "complete"
    assert log[0]["selection_oscr"] is not None
    assert not list((tmp_path / "search").rglob("test_predictions.npz"))


def test_cpu_pipeline(tmp_path):
    result = run_smoke(tmp_path)
    assert result["encoder_update"] > 0
    assert result["baseline_score_verified"]


def test_paper_learning_rate_endpoints():
    config = load_config()
    assert epoch_lr(config, 1) == 1e-5
    assert abs(epoch_lr(config, 5) - 1e-4) < 1e-12
    assert abs(epoch_lr(config, 100) - 1e-6) < 1e-12


def test_selected_hyperparameters_survive_experiment_plan():
    config = load_config(overrides={"lambda_proto": 0.2, "lambda_roi": 0.1,
                                   "lambda_bg": 0.01, "lr": 3e-5})
    main = config | variants("main", config)["sgep"]
    assert main["lambda_proto"] == 0.2
    assert main["lambda_roi"] == 0.1
    factorial = config | variants("factorial", config)["roi1_bg0"]
    assert factorial["lambda_proto"] == 0.2
    assert factorial["lr"] == 3e-5
    assert factorial["lambda_roi"] == 0.05
    assert factorial["lambda_bg"] == 0
    spatial = config | variants("spatial", config)["displaced"]
    assert spatial["lambda_proto"] == 0.2
    assert spatial["lambda_roi"] == 0.1
    candidates = search_candidates(config, "sgep", 24)
    assert candidates[0]["lambda_proto"] == 0.2
    assert len({tuple(sorted(candidate.items())) for candidate in candidates}) == 24
