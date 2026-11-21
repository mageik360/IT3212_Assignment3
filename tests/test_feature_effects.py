import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import feature_effects as fe


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 2)), columns=["a", "b"])
    y = (X.a + X.a * X.b + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return X, y


def test_linear_model_effect_is_coefficient_times_change(data):
    X, y = data
    model = LogisticRegression().fit(X, y)
    table = fe.total_effects(model, X, lambda rows: rows, [("a", -1.0, 1.0, "")])
    expected = 2 * model.coef_[0][0]
    row = table.iloc[0]
    # Without interactions the change is the same for everyone
    assert row.mean_delta_logit == pytest.approx(expected)
    assert row.delta_logit_p10 == pytest.approx(row.delta_logit_p90)


def test_interaction_makes_effect_depend_on_other_features(data):
    X, y = data
    model = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                      ("clf", LogisticRegression())]).fit(X, y)
    groups = pd.Series(np.where(X.b > 0, "high_b", "low_b"), index=X.index)
    row = fe.total_effects(model, X, lambda rows: rows, [("a", -1.0, 1.0, "")], groups).iloc[0]
    # The a*b term makes a matter more when b is high
    assert row.mean_delta_logit_high_b > row.mean_delta_logit_low_b
    assert row.delta_logit_p10 < row.delta_logit_p90
    assert row.odds_ratio_of_mean == pytest.approx(np.exp(row.mean_delta_logit))


def test_input_rows_are_not_modified(data):
    X, y = data
    model = LogisticRegression().fit(X, y)
    before = X.copy()
    fe.total_effects(model, X, lambda rows: rows, [("a", 0.0, 1.0, "")])
    pd.testing.assert_frame_equal(X, before)
