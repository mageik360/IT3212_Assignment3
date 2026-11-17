import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_pipeline as dp
import feature_selection as fs
from feature_engineering import DERIVED_COLUMNS, SOURCE_COLUMNS
from preprocessing import fit_preprocessor, transform_features


def raw_frame(n=200, seed=0):
    """Raw records with the dataset's columns, in plausible units."""
    rng = np.random.default_rng(seed)
    columns = {c: rng.uniform(50, 150, n) for c in dp.CONTINUOUS_COLUMNS}
    columns.update({
        "height": rng.uniform(150, 190, n), "weight": rng.uniform(45, 100, n),
        "sight_left": rng.uniform(0.1, 1.5, n), "sight_right": rng.uniform(0.1, 1.5, n),
        "hear_left": rng.integers(1, 3, n), "hear_right": rng.integers(1, 3, n),
        "urine_protein": rng.integers(1, 7, n),
        "sex": rng.choice(["Female", "Male"], n),
        "SMK_stat_type_cd": rng.integers(1, 4, n),
    })
    return pd.DataFrame(columns)


def preprocess(raw, *, derived=True, placeholders=dp.PLACEHOLDERS):
    continuous = [c for c in dp.CONTINUOUS_COLUMNS if c in raw]
    state = fit_preprocessor(
        raw, placeholders=placeholders,
        mappings={"SMK_stat_type_cd": dp.ENCODINGS["SMK_stat_type_cd"]},
        categories=dp.CATEGORIES, continuous_columns=continuous,
        log_columns=[c for c in dp.LOG_COLUMNS if c in raw],
        target_name=dp.TARGET, target_mapping=dp.ENCODINGS[dp.TARGET],
        extract_derived_features=derived,
    )
    return transform_features(raw, state)


def test_groups_cover_the_processed_columns_exactly():
    processed = preprocess(raw_frame())
    assert processed.shape[1] == 29
    fs.check_groups(processed.columns)
    assert {"sex_Male", "SMK_stat_type_cd_not_anymore", "SMK_stat_type_cd_current"} <= set(processed)
    with pytest.raises(ValueError):
        fs.check_groups(processed.drop(columns="age").columns)
    with pytest.raises(ValueError):
        fs.check_groups([*processed.columns, "extra"])


def test_derived_sources_match_feature_engineering():
    assert set(fs.DERIVED_SOURCES) == set(DERIVED_COLUMNS)
    assert {c for s in fs.DERIVED_SOURCES.values() for c in s} == set(SOURCE_COLUMNS)


@pytest.mark.parametrize("group, derived", [
    ("body_size", ["bmi", "waist_to_height_ratio"]),
    ("liver", ["ast_alt_ratio"]),
    ("lipids", ["non_hdl_cholesterol"]),
    ("blood_pressure", ["pulse_pressure"]),
    ("age", []),
    ("smoking", []),
])
def test_derived_features_are_dropped_with_their_sources(group, derived):
    assert fs.dropped_columns(group) == fs.GROUPS[group] + derived
    assert fs.dropped_columns(fs.BASELINE) == []
    assert fs.dropped_columns("derived") == list(DERIVED_COLUMNS)


def test_dropping_processed_columns_equals_never_creating_them():
    # The ablation's premise: preprocessing is per column, so a smaller input
    # gives the same values for the columns that remain
    raw = raw_frame()
    raw.loc[::7, "sight_left"] = 9.9  # placeholder, so imputation is exercised
    full = preprocess(raw)
    drop = fs.GROUPS["sight"] + list(DERIVED_COLUMNS)
    without = preprocess(raw.drop(columns=fs.GROUPS["sight"]), derived=False,
                         placeholders={"waistline": [999]})
    pd.testing.assert_frame_equal(full.drop(columns=drop), without[full.drop(columns=drop).columns])


def test_bootstrap_ap_matches_sklearn_on_resampled_rows():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    # Rounding creates tied scores, which sklearn treats as one threshold
    proba = np.round(np.clip(0.3 * y + rng.uniform(0, 0.7, 300), 0, 1), 2)
    counts = fs.bootstrap_counts(len(y), 20, random_state=3)
    assert (counts.sum(axis=1) == len(y)).all()
    boot = fs.bootstrap_ap(y, proba, counts, chunk=7)
    for b in range(len(counts)):
        idx = np.repeat(np.arange(len(y)), counts[b])
        assert boot[b] == pytest.approx(average_precision_score(y[idx], proba[idx]))
    assert fs.bootstrap_ap(y, proba, np.ones((1, len(y)), dtype=np.uint16))[0] == \
        pytest.approx(average_precision_score(y, proba))


def test_bootstrap_counts_are_shared_and_seeded():
    a = fs.bootstrap_counts(100, 5)
    assert np.array_equal(a, fs.bootstrap_counts(100, 5))
    assert not np.array_equal(a, fs.bootstrap_counts(100, 5, random_state=7))


@pytest.fixture
def splits():
    rng = np.random.default_rng(42)
    columns = [c for cols in fs.GROUPS.values() for c in cols]
    X = pd.DataFrame(rng.normal(size=(300, len(columns))), columns=columns)
    y = (X.age + X.bmi + rng.normal(scale=0.5, size=300) > 0).astype(int)
    return X.iloc[:200], X.iloc[200:250], X.iloc[250:], y.iloc[:200], y.iloc[200:250], y.iloc[250:]


def test_feature_selection_main_dry_run(splits, monkeypatch, tmp_path, capsys):
    X_train, X_val, _, y_train, y_val, _ = splits
    # The test split is None, so any use of it fails
    monkeypatch.setattr(fs, "run_pipeline", lambda: (X_train, X_val, None, y_train, y_val, None, None))
    monkeypatch.setattr(fs, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(fs, "RESULT_DIR", tmp_path / "results")
    config = tmp_path / "selected_config.json"
    config.write_text(json.dumps({"hidden_dim": 4, "learning_rate": 1e-3,
                                  "weight_decay": 1e-4, "batch_size": 64}))
    monkeypatch.setattr(fs, "NN_CONFIG", config)

    fs.main(["--models", "tree", "logreg_linear", "--with-nn", "--n-bootstrap", "20",
             "--epochs", "2", "--device", "cpu"])
    out = capsys.readouterr().out
    assert "Δ validation AP" in out and "Figure saved" in out
    table = pd.read_csv(tmp_path / "results" / "ablation.csv")
    n_configs = len(fs.GROUPS) + 1
    assert len(table) == 2 * n_configs + 2
    assert list(table.model.unique()) == ["tree", "logreg_linear", "nn"]
    base = table[table.config == fs.BASELINE]
    assert (base.delta_ap == 0).all() and (base.ci_low == 0).all() and (base.ci_high == 0).all()
    assert (table.ci_low <= table.delta_ap + 1e-12).all() and (table.delta_ap <= table.ci_high + 1e-12).all()
    body = table[(table.model == "tree") & (table.config == "body_size")].iloc[0]
    assert body.dropped_columns.split(";") == ["height", "weight", "waistline",
                                               "bmi", "waist_to_height_ratio"]
    assert body.n_features == 29 - 5
    assert set(table[table.model == "nn"].config) == {fs.BASELINE, "derived"}
    assert (tmp_path / "figures" / "delta_ap.png").read_bytes().startswith(b"\x89PNG")

    # Rerunning a subset replaces only that model's rows
    fs.main(["--models", "tree", "--n-bootstrap", "20"])
    rerun = pd.read_csv(tmp_path / "results" / "ablation.csv")
    assert len(rerun) == len(table) and set(rerun.model) == {"tree", "logreg_linear", "nn"}
