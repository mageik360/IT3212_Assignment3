import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import paired_bootstrap as pb
from feature_selection import bootstrap_counts


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    base = y * 0.3 + rng.uniform(0, 0.7, 500)
    # Rounding creates ties, which both AUC implementations must treat alike
    return y, np.round(base, 2), np.round(base + rng.normal(0, 0.05, 500), 2)


def test_roc_auc_matches_sklearn_with_weights(scores):
    y, p, _ = scores
    counts = bootstrap_counts(len(y), 5)
    expected = [roc_auc_score(y, p, sample_weight=c) for c in counts]
    np.testing.assert_allclose(pb.bootstrap_roc_auc(y, p, counts), expected)


def test_threshold_metrics_match_point_metrics_on_full_sample(scores):
    y, p, _ = scores
    ones = np.ones((1, len(y)), dtype=np.uint16)
    boot = pb.bootstrap_threshold_metrics(y, p, ones)
    point = pb.point_metrics(y, p)
    for metric in ["accuracy", "precision", "recall", "false_positives"]:
        assert boot[metric][0] == pytest.approx(point[metric])


def test_paired_interval_is_narrower_and_brackets_estimate(scores):
    y, p, q = scores
    models, diffs = pb.paired_bootstrap(y, {"a": p, "b": q}, [("a", "b")], n_bootstrap=200)
    assert set(models.model) == {"a", "b"} and len(diffs) == len(pb.METRICS)
    for row in diffs.itertuples():
        assert row.ci_low <= row.difference <= row.ci_high
    ap = diffs.set_index("metric").loc["average_precision"]
    # The two scores are strongly correlated, so pairing removes most of the noise
    assert ap.ci_high - ap.ci_low < ap.unpaired_half_width


def test_main_writes_tables(scores, tmp_path):
    y, p, q = scores
    path = tmp_path / "probabilities.npz"
    np.savez(path, y_test=y, logreg_test=p, tree_test=q)
    pb.main(["--input", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "20"])
    diffs = pd.read_csv(tmp_path / "out" / "paired_differences.csv")
    assert set(zip(diffs.model_a, diffs.model_b)) == {("logreg", "tree")}
