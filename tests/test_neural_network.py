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


def test_refit_uses_shared_loop_with_all_rows_and_selected_schedule(monkeypatch, tmp_path):
    import neural_network as network

    X = pd.DataFrame({"a": [-2., -1., 1., 2., 3.]})
    y = pd.Series([0, 0, 1, 1, 1])
    selection = pd.DataFrame({"epoch": [1, 2, 3], "val_loss": [.6, .5, .7],
                              "learning_rate": [.01, .005, .0025]})
    monkeypatch.setattr(network, "predict", lambda *args: pytest.fail("Refit evaluated a holdout"))
    real_fit = network.fit
    calls = []

    def fit(*args, **kwargs):
        calls.append((len(args[2]), args[4], args[5], kwargs["learning_rates"]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(network, "fit", fit)
    model, history, summary = network.refit_model(
        X.iloc[:3], y.iloc[:3], X.iloc[3:], y.iloc[3:], selection_history=selection,
        hidden_dim=4, batch_size=5, device="cpu", verbose=False)
    assert calls == [(5, None, None, [.01, .005])]
    assert history.learning_rate.tolist() == [.01, .005]
    assert summary["epochs_run"] == 2
    assert summary["training_rows"] == 5
    assert "val_loss" not in history
    assert "best_epoch" not in summary
    assert not model.training
    assert model.feature_columns == ["a"]
    # Independent full-batch updates verify fresh initialization and final weights.
    torch.manual_seed(42)
    expected = network.NeuralNetwork(1, 4)
    optimizer = torch.optim.Adam(expected.parameters(), lr=.01, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(network.make_dataset(X, y), batch_size=5,
        shuffle=True, generator=torch.Generator().manual_seed(42))
    for rate in [.01, .005]:
        optimizer.param_groups[0]["lr"] = rate
        for features, labels in loader:
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(expected(features), labels).backward()
            optimizer.step()
    for actual, reference in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, reference)
    assert network.save_training_plots(history, tmp_path).exists()
    with pytest.raises(ValueError, match="history"):
        network.refit_model(X, y, X, y, selection_history=selection.iloc[:0])


@pytest.mark.parametrize("refit", [False, True])
def test_cli_refit_evaluation_order(monkeypatch, tmp_path, refit):
    import neural_network as network

    X = pd.DataFrame({"a": [-2., -1., 1., 2., -3., 3., -4., 4.]})
    y = (X.a > 0).astype(int)
    X_val, X_test = X.iloc[4:6], X.iloc[6:]
    monkeypatch.setattr(network, "run_pipeline", lambda: (
        X.iloc[:4], X_val, X_test, y.iloc[:4], y.iloc[4:6], y.iloc[6:], None))
    calls = []
    real_evaluate, real_refit = network.evaluate_model, network.refit_model

    def evaluate(model, features, labels):
        calls.append("test" if features is X_test else "validation")
        return real_evaluate(model, features, labels)

    def refit_model(*args, **kwargs):
        calls.append("refit")
        return real_refit(*args, **kwargs)

    monkeypatch.setattr(network, "evaluate_model", evaluate)
    monkeypatch.setattr(network, "refit_model", refit_model)
    network.main(["--epochs", "2", "--hidden-dim", "4", "--device", "cpu",
                  "--figures-dir", str(tmp_path)] + (["--refit"] if refit else []))
    assert calls == (["validation", "refit", "test"] if refit else ["validation", "test"])
    assert (tmp_path / "refit/neural_network_history.csv").exists() == refit
