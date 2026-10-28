import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
from evaluation import compute_metrics, plot_roc_curve
from plotting import save_figure
from tuning import results_table, tune_on_validation


@pytest.fixture
def splits():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = (X.a + 0.5 * X.b + rng.normal(scale=0.3, size=300) > 0).astype(int)
    return X.iloc[:200], X.iloc[200:250], X.iloc[250:], y.iloc[:200], y.iloc[200:250], y.iloc[250:]


def test_compute_metrics_matches_sklearn():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
    pred = (proba >= 0.5).astype(int)
    metrics = compute_metrics(y, proba)
    assert list(metrics) == ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert metrics["f1"] == pytest.approx(f1_score(y, pred))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, proba))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 2]]
    # Threshold is configurable
    assert compute_metrics(y, proba, threshold=0.3)["recall"] == 1.0


def test_compute_metrics_single_class_has_nan_auc():
    metrics = compute_metrics(pd.Series([1, 1, 1]), np.array([0.2, 0.7, 0.9]))
    assert np.isnan(metrics["roc_auc"])
    assert metrics["confusion_matrix"] == [[0, 0], [1, 2]]


def test_plot_roc_curve_supports_several_models(tmp_path):
    y = np.array([0, 1, 0, 1])
    path = plot_roc_curve(y, {"A": [0.1, 0.9, 0.2, 0.8], "B": [0.5, 0.5, 0.4, 0.6]},
                          tmp_path / "nested" / "roc.png")
    assert path.read_bytes().startswith(b"\x89PNG")


def test_save_figure_creates_folder_and_closes(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_figure(fig, tmp_path / "a" / "b.png")
    assert path.read_bytes().startswith(b"\x89PNG")
    assert not plt.fignum_exists(fig.number)


def test_tune_on_validation_scores_only_the_validation_split(splits):
    X_train, X_val, _, y_train, y_val, _ = splits
    grid = {"max_depth": [1, 3], "min_samples_leaf": [1, 5]}
    search = tune_on_validation(DecisionTreeClassifier(random_state=0), grid,
                                X_train, y_train, X_val, y_val, verbose=0)
    # The validation score must equal refitting on train only and scoring on val
    best = DecisionTreeClassifier(random_state=0, **search.best_params_).fit(X_train, y_train)
    expected = roc_auc_score(y_val, best.predict_proba(X_val)[:, 1])
    assert search.cv_results_["mean_test_roc_auc"][search.best_index_] == pytest.approx(expected)
    # The best estimator is refitted on train + validation
    assert search.best_estimator_.tree_.n_node_samples[0] == len(X_train) + len(X_val)

    table = results_table(search)
    assert len(table) == 4
    assert {"max_depth", "min_samples_leaf", "train_accuracy", "val_roc_auc", "fit_time_s"} <= set(table.columns)
    assert table.val_roc_auc.is_monotonic_decreasing


def test_seed_and_paths_are_shared():
    import data_pipeline
    import eda

    assert data_pipeline.RANDOM_STATE == eda.RANDOM_STATE == config.RANDOM_STATE
    assert eda.FIGURE_DIR == config.FIGURES_DIR / "eda"
    assert (config.PROJECT_ROOT / "src" / "config.py").exists()


def test_decision_tree_main_dry_run(splits, monkeypatch, tmp_path, capsys):
    import decision_tree

    monkeypatch.setattr(decision_tree, "run_pipeline", lambda: (*splits, None))
    monkeypatch.setattr(decision_tree, "FIGURE_DIR", tmp_path)
    monkeypatch.setattr(decision_tree, "PARAM_GRID", {"max_depth": [2, 4]})
    decision_tree.main()
    out = capsys.readouterr().out
    assert "Test metrics:" in out and "confusion_matrix" in out
    assert (tmp_path / "roc_curve.png").read_bytes().startswith(b"\x89PNG")


def test_neural_network_main_dry_run(splits, monkeypatch, tmp_path, capsys):
    import neural_network

    monkeypatch.setattr(neural_network, "run_pipeline", lambda: (*splits, None))
    monkeypatch.setattr(sys, "argv", ["neural_network.py", "--epochs", "2", "--device", "cpu",
                                      "--figures-dir", str(tmp_path)])
    neural_network.main()
    out = capsys.readouterr().out
    assert "Test metrics:" in out and "inference_seconds" in out
    assert (tmp_path / "neural_network_training.png").exists()


def test_logistic_regression_main_dry_run(splits, monkeypatch, tmp_path, capsys):
    import logistic_regression

    monkeypatch.setattr(logistic_regression, "run_pipeline", lambda: (*splits, None))
    monkeypatch.setattr(logistic_regression, "FIGURE_DIR", tmp_path)
    monkeypatch.setattr(logistic_regression, "PARAM_GRID", {"poly__degree": [1, 2], "clf__C": [1]})
    logistic_regression.main()
    out = capsys.readouterr().out
    assert "Linear vs polynomial" in out and "Test metrics:" in out
    assert (tmp_path / "roc_curve.png").read_bytes().startswith(b"\x89PNG")
