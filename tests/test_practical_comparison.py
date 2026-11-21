import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import practical_comparison as pc


@pytest.fixture
def splits():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(600, 3)), columns=list("abc"))
    y = (X.a + 0.5 * X.b + rng.normal(scale=0.5, size=600) > 0).astype(int)
    return X.iloc[:400], X.iloc[400:500], X.iloc[500:], y.iloc[:400], y.iloc[400:500], y.iloc[500:]


@pytest.fixture
def small_configs(tmp_path, monkeypatch):
    """Tiny network settings, so no saved task 9 runs are found or needed."""
    shallow = tmp_path / "shallow.json"
    shallow.write_text(json.dumps({"hidden_dim": 4, "learning_rate": 1e-2,
                                   "weight_decay": 1e-4, "batch_size": 64}))
    saved = tmp_path / "compare"
    saved.mkdir()
    (saved / "selected_config.json").write_text(json.dumps({
        "hidden_dims": [8, 4], "dropout": 0.1, "batch_norm": True,
        "learning_rate": 1e-2, "weight_decay": 1e-4, "batch_size": 64}))
    monkeypatch.setattr(pc, "SHALLOW_CONFIG", shallow)
    monkeypatch.setattr(pc, "SAVED_NN_DIR", saved)
    return saved


def test_threshold_is_lowest_reaching_target():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
    # Precision at >= 0.1, 0.4, 0.6, 0.8, 0.9: 3/5, 3/4, 2/3, 1, 1
    assert pc.threshold_for_precision(y, proba, 0.7) == 0.4
    assert pc.threshold_for_precision(y, proba, 0.95) == 0.8
    assert np.isnan(pc.threshold_for_precision(np.array([1, 0]), np.array([0.2, 0.9]), 0.6))


def test_rule_metrics_counts_and_weights():
    y = np.array([1, 1, 0, 0])
    placed = np.array([True, False, True, False])
    assert pc.rule_metrics(y, placed) == {"precision": 0.5, "recall": 0.5, "share_placed": 0.5}
    # Weights act as repeated rows
    weighted = pc.rule_metrics(y, placed, weights=[3, 1, 1, 0])
    repeated = pc.rule_metrics([1, 1, 1, 1, 0], [True, True, True, False, True])
    assert weighted == pytest.approx(repeated)
    assert np.isnan(pc.rule_metrics(y, np.zeros(4, bool))["precision"])


def test_thresholds_come_from_validation_only(splits):
    _, _, _, _, y_val, y_test = splits
    rng = np.random.default_rng(0)
    p_val = np.clip(y_val.to_numpy() * 0.4 + rng.uniform(0, 0.6, len(y_val)), 0, 1)
    p_test = np.clip(y_test.to_numpy() * 0.4 + rng.uniform(0, 0.6, len(y_test)), 0, 1)
    probas = {"logreg": (p_val, p_test)}
    table = pc.evaluate_rules(y_val, y_test, probas, [0.8], n_bootstrap=50)
    flipped = pc.evaluate_rules(y_val, 1 - y_test, probas, [0.8], n_bootstrap=50)
    # Changing the test labels must not move the chosen thresholds
    assert table.threshold.tolist() == flipped.threshold.tolist()
    assert table.threshold.iloc[0] == 0.5
    row = table.iloc[1]
    assert row.threshold == pc.threshold_for_precision(y_val, p_val, 0.8)
    assert row.val_precision >= 0.8
    for metric in ["precision", "recall", "share_placed"]:
        assert row[f"test_{metric}_ci_low"] <= row[f"test_{metric}"] <= row[f"test_{metric}_ci_high"]
    assert row.test_placed == row.test_true_positives + row.test_false_positives


def test_saved_network_round_trip(splits, small_configs):
    X_train, X_val, _, y_train, y_val, _ = splits
    model, _ = pc.fit_model("deep_nn", X_train, y_train, X_val, y_val, seed=42, device="cpu",
                            epochs=2)
    run = small_configs / "tuned_deep" / "seed_42"
    run.mkdir(parents=True)
    torch.save({"config": {"hidden_dims": [8, 4], "dropout": 0.1, "batch_norm": True,
                           "learning_rate": 1e-2, "weight_decay": 1e-4, "batch_size": 64},
                "feature_columns": model.feature_columns,
                "state_dict": model.state_dict()}, run / "model.pt")
    loaded = pc.load_saved_nn("deep_nn", X_train.shape[1], "cpu")
    np.testing.assert_allclose(pc.predict_proba(loaded, X_val), pc.predict_proba(model, X_val))
    assert pc.load_saved_nn("shallow_nn", X_train.shape[1], "cpu") is None


def test_main_dry_run(splits, small_configs, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pc, "run_pipeline", lambda: (*splits, None))
    out_dir = tmp_path / "out"
    pc.main(["--epochs", "2", "--device", "cpu", "--repeats", "2", "--inference-repeats", "1",
             "--n-bootstrap", "20", "--targets", "0.7", "0.8", "--output-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert "Part A" in out and "Part B" in out

    threshold = pd.read_csv(out_dir / "threshold.csv")
    assert set(threshold.model) == set(pc.MODELS)
    assert len(threshold) == len(pc.MODELS) * 3  # default 0.5 plus two targets
    check = pd.read_csv(out_dir / "model_check.csv")
    assert (check.source == "trained now (seed 42)").all()
    saved = np.load(out_dir / "probabilities.npz")
    assert saved["logreg_test"].shape == saved["y_test"].shape

    runs = pd.read_csv(out_dir / "timing_runs.csv")
    assert len(runs) == len(pc.MODELS) * 2 and (runs.fit_seconds > 0).all()
    # Networks change seed between repeats; sklearn models keep it
    assert runs[runs.model == "deep_nn"].seed.tolist() == [42, 43]
    assert runs[runs.model == "tree"].seed.tolist() == [42, 42]
    summary = pd.read_csv(out_dir / "timing_summary.csv")
    assert summary.repeats.tolist() == [2] * len(pc.MODELS)
    assert summary.set_index("model").fit_time_vs_logreg["logreg"] == pytest.approx(1)
    env = json.loads((out_dir / "environment.json").read_text())
    assert env["arguments"]["repeats"] == 2 and env["deep_settings"]["hidden_dims"] == [8, 4]
