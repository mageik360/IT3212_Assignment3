import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import advanced_neural_network as advanced
from neural_network import evaluate_model, make_dataset, predict


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(240, 3)), columns=list("abc"))
    y = (X.a + X.b > 0).astype(int)
    return X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("batch_norm", [True, False])
def test_learning_and_checkpoint(data, batch_norm):
    config = advanced.ModelConfig(
        hidden_dims=(16, 8),
        dropout=0,
        batch_norm=batch_norm,
        learning_rate=0.02,
        batch_size=32,
    )
    model, history, summary = advanced.train_model(
        *data, config=config, epochs=30, device="cpu", verbose=False
    )
    loss, probabilities = predict(model, make_dataset(data[2], data[3]))
    assert loss == pytest.approx(history.val_loss.min())
    assert summary["best_epoch"] == history.loc[history.val_loss.idxmin(), "epoch"]
    assert probabilities.shape == (60,)
    assert evaluate_model(model, data[2], data[3])["accuracy"] > 0.9
    assert model(torch.zeros(1, 3)).shape == (1,)
    assert not model.training


@pytest.mark.parametrize("batch_size", [1, 4])
def test_singleton_batches(batch_size):
    X = pd.DataFrame({"a": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    y = pd.Series([0, 0, 0, 1, 1])
    config = advanced.ModelConfig(hidden_dims=(4, 2), batch_size=batch_size)
    model, history, _ = advanced.train_model(
        X, y, X, y, config=config, epochs=2, device="cpu", verbose=False
    )
    assert np.isfinite(history.train_loss).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_sampling_and_ranking():
    configs = advanced.sample_configs(30, 42)
    assert configs == advanced.sample_configs(30, 42)
    assert configs != advanced.sample_configs(30, 43)
    assert configs[0] == advanced.ModelConfig()
    assert all(1e-4 <= c.learning_rate <= 3e-3 for c in configs)
    assert all(1e-6 <= c.weight_decay <= 1e-2 for c in configs)
    table = pd.DataFrame(
        {
            "trial": [0, 1, 2],
            "val_roc_auc": [0.8, 0.9, 0.9],
            "val_loss": [0.2, 0.4, 0.3],
        }
    )
    assert advanced.rank_trials(table).trial.tolist() == [2, 1, 0]
    assert not any(
        "test" in p for p in inspect.signature(advanced.tune_model).parameters
    )


@pytest.mark.parametrize("mode", ["train", "tune", "compare"])
def test_cli_smoke_and_artifacts(data, monkeypatch, tmp_path, mode):
    X_train, y_train, X_val, y_val = data
    X_test = X_val.copy()
    X_test.index = X_test.index + 1000
    y_test = pd.Series(y_val.to_numpy(), index=X_test.index)
    monkeypatch.setattr(
        advanced,
        "run_pipeline",
        lambda: (X_train, X_val, X_test, y_train, y_val, y_test, {"test_state": True}),
    )
    real_evaluate = advanced.evaluate_model
    events = []

    def evaluate(model, X, y):
        events.append("test" if X is X_test else "validation")
        return real_evaluate(model, X, y)

    monkeypatch.setattr(advanced, "evaluate_model", evaluate)
    advanced.main(
        [
            "--mode",
            mode,
            "--epochs",
            "1",
            "--trials",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )
    directory = tmp_path / mode
    assert (directory / "preprocessing.joblib").exists()
    assert (directory / "metrics_summary.csv").exists()
    metrics = pd.read_csv(directory / "metrics.csv")
    assert len(metrics) == (18 if mode == "compare" else 2)
    if mode != "train":
        assert events[:2] == ["validation", "validation"]
        assert (directory / "search/trials.csv").exists()
    name = (
        "shallow" if mode == "compare" else "tuned_deep" if mode == "tune" else "deep"
    )
    run = directory / name / "seed_42"
    assert (run / "neural_network_training.png").read_bytes().startswith(b"\x89PNG")
    checkpoint = torch.load(run / "model.pt", weights_only=True)
    if mode != "compare":
        model = advanced.AdvancedNeuralNetwork(
            3, advanced.ModelConfig(**checkpoint["config"])
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.feature_columns = checkpoint["feature_columns"]
        measured = real_evaluate(model, X_test, y_test)
        saved = metrics[metrics.split == "test"].iloc[0]
        assert measured["loss"] == pytest.approx(saved.loss)


def test_plateau_reduces_lr_and_stops(data, monkeypatch):
    monkeypatch.setattr(advanced, "predict", lambda *args: (0.7, np.zeros(60)))
    _, history, summary = advanced.train_model(
        *data,
        config=advanced.ModelConfig(hidden_dims=(8, 4)),
        epochs=20,
        patience=6,
        device="cpu",
        verbose=False,
    )
    assert summary["best_epoch"] == 1
    assert summary["epochs_run"] == 7
    assert history.learning_rate.iloc[5] == pytest.approx(0.0005)
