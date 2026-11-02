import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bagging_boosting
from bagging_boosting import (
    MeanWeightedLogisticRegression, polynomial_logreg, proba_at, select_size, staged_proba,
)


@pytest.fixture
def splits():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = (X.a * X.b + 0.5 * X.c + rng.normal(scale=0.3, size=300) > 0).astype(int)
    return X.iloc[:200], X.iloc[200:250], X.iloc[250:], y.iloc[:200], y.iloc[200:250], y.iloc[250:]


@pytest.mark.parametrize("model", [
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=5, random_state=0),
    BaggingClassifier(DecisionTreeClassifier(), n_estimators=5, random_state=0),
    polynomial_logreg(lambda clf: BaggingClassifier(clf, n_estimators=3, random_state=0)),
    DecisionTreeClassifier(max_depth=2),
])
def test_staged_proba_ends_at_predict_proba(splits, model):
    X_train, X_val, _, y_train, _, _ = splits
    model.fit(X_train, y_train)
    stages = list(staged_proba(model, X_val))
    assert stages[-1] == pytest.approx(model.predict_proba(X_val)[:, 1])
    assert proba_at(model, X_val, len(stages)) == pytest.approx(stages[-1])
    with pytest.raises(ValueError):
        proba_at(model, X_val, len(stages) + 1)


def test_mean_weighted_logreg_ignores_weight_scale(splits):
    X_train, _, _, y_train, _, _ = splits
    n = len(y_train)
    plain = LogisticRegression(C=0.01).fit(X_train, y_train)
    # AdaBoost's first round: uniform weights summing to 1
    weighted = MeanWeightedLogisticRegression(C=0.01).fit(X_train, y_train, sample_weight=np.full(n, 1 / n))
    assert weighted.coef_ == pytest.approx(plain.coef_, rel=1e-4)


def test_select_size_early_stops_only_boosting():
    curve = pd.DataFrame({"n_members": [1, 2, 3, 4], "val_roc_auc": [0.9, 0.7, 0.75, 0.74],
                          "val_average_precision": [0.7, 0.8, 0.75, 0.74]})
    assert select_size("boosting", curve) == 2
    assert select_size("bagging", curve) == 4
    assert select_size("single", curve.head(1)) == 1


def test_bagging_boosting_main_dry_run(splits, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bagging_boosting, "run_pipeline", lambda: (*splits, None))
    monkeypatch.setattr(bagging_boosting, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(bagging_boosting, "RESULT_DIR", tmp_path / "results")
    for name in ["TREE_ROUNDS", "TREE_MEMBERS", "LOGREG_ROUNDS", "LOGREG_MEMBERS"]:
        monkeypatch.setattr(bagging_boosting, name, 3)
    monkeypatch.setattr(bagging_boosting, "LOGREG_N_JOBS", 1)

    bagging_boosting.main([])
    out = capsys.readouterr().out
    assert out.count("Test metrics:") == 6 and "Summary" in out
    assert "Boosting round 3/3" in out
    table = pd.read_csv(tmp_path / "results" / "metrics.csv")
    assert list(table.model) == [m.name for m in bagging_boosting.build_models()]
    assert (table.n_used <= table.n_members).all()
    for name in ["validation_curves.png", "roc_curve_tree.png", "roc_curve_logreg.png",
                 "pr_curve_tree.png", "pr_curve_logreg.png"]:
        assert (tmp_path / "figures" / name).read_bytes().startswith(b"\x89PNG")

    # Rerunning a subset replaces only those rows
    bagging_boosting.main(["--models", "tree"])
    assert len(pd.read_csv(tmp_path / "results" / "metrics.csv")) == 6
