import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_network import evaluate_model, make_dataset, predict, train_model


def test_training_learns_and_restores_best_validation_epoch():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = (X["a"] + X["b"] > 0).astype(int)
    model, history, summary = train_model(
        X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:],
        epochs=25, batch_size=64, learning_rate=0.03, patience=3, verbose=False,
        device="cpu",
    )
    loss, probabilities = predict(model, make_dataset(X.iloc[300:], y.iloc[300:]))
    assert loss == pytest.approx(history.val_loss.min())
    assert summary["best_epoch"] == history.loc[history.val_loss.idxmin(), "epoch"]
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    metrics = evaluate_model(model, X.iloc[300:], y.iloc[300:])
    assert metrics["accuracy"] > 0.9
    assert np.sum(metrics["confusion_matrix"]) == 100
    assert not model.training
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(ValueError, match="columns"):
        evaluate_model(model, X[["b", "a", "c"]], y)


def test_dataset_rejects_misaligned_rows_and_invalid_data():
    X = pd.DataFrame({"a": [0.0, 1.0]}, index=[4, 8])
    y = pd.Series([0, 1], index=[4, 8])
    with pytest.raises(ValueError, match="indices"):
        make_dataset(X, y.iloc[::-1])
    with pytest.raises(ValueError, match="finite"):
        make_dataset(X.assign(a=np.nan), y)
    with pytest.raises(ValueError, match="0 or 1"):
        make_dataset(X, y.replace(1, 2))


def test_scheduler_reduces_learning_rate_and_saves_curves(monkeypatch, tmp_path):
    import neural_network as network

    X = pd.DataFrame({"a": [0.0, 1.0, -1.0, 2.0]})
    y = pd.Series([0, 1, 0, 1])
    # A fixed validation loss forces a plateau regardless of optimizer progress.
    monkeypatch.setattr(network, "predict", lambda *args: (0.7, np.zeros(4)))
    _, history, summary = train_model(
        X, y, X, y, epochs=7, device="cpu", verbose=False,
    )
    assert history.learning_rate.iloc[0] == pytest.approx(0.001)
    assert history.learning_rate.iloc[5] == pytest.approx(0.0005)
    assert summary["best_epoch"] == 1
    path = network.save_training_plots(history, tmp_path / "figures")
    assert path.read_bytes().startswith(b"\x89PNG")
    saved = pd.read_csv(path.parent / "neural_network_history.csv")
    pd.testing.assert_frame_equal(saved, history)
