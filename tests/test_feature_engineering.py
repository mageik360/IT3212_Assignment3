import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from feature_engineering import DERIVED_COLUMNS, extract_features
from preprocessing import fit_preprocessor, transform_features


def measurements():
    return pd.DataFrame({
        'height': [200., 160., 180.], 'weight': [100., 64., 81.],
        'waistline': [100., 80., 90.], 'SGOT_AST': [40., 30., 20.],
        'SGOT_ALT': [20., 30., 40.], 'tot_chole': [200., 180., 220.],
        'HDL_chole': [50., 60., 70.], 'SBP': [120., 130., 140.],
        'DBP': [80., 85., 90.],
    }, index=[7, 11, 19])


def test_formulas_preserve_original_data_and_index():
    raw = measurements()
    original = raw.copy(deep=True)
    result = extract_features(raw)
    assert result.loc[7, DERIVED_COLUMNS].tolist() == [25., .5, 2., 150., 40.]
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(result[raw.columns], raw)


def test_invalid_denominators_and_missing_sources():
    raw = measurements()
    raw.loc[7, 'height'] = 0
    raw.loc[11, 'SGOT_ALT'] = -1
    raw.loc[19, 'SGOT_AST'] = np.inf
    result = extract_features(raw)
    assert result.loc[7, ['bmi', 'waist_to_height_ratio']].isna().all()
    assert result.loc[[11, 19], 'ast_alt_ratio'].isna().all()
    with pytest.raises(ValueError, match='height'):
        extract_features(raw.drop(columns='height'))


def test_saved_state_uses_training_statistics_and_original_units(tmp_path):
    import joblib

    raw = measurements()
    kwargs = dict(placeholders={}, mappings={}, categories={},
                  continuous_columns=list(raw.columns),
                  log_columns=['SGOT_AST', 'SGOT_ALT'],
                  target_name='DRK_YN', target_mapping={'N': 0, 'Y': 1})
    state = fit_preprocessor(raw, extract_derived_features=True, **kwargs)
    path = tmp_path / 'state.joblib'
    joblib.dump(state, path)
    loaded = joblib.load(path)
    prediction = raw.loc[[7]].copy()
    prediction.loc[7, 'SGOT_AST'] = np.nan
    transformed = transform_features(prediction, loaded)
    assert np.isfinite(transformed[DERIVED_COLUMNS].to_numpy()).all()
    assert list(transformed.columns) == state.feature_order
    assert transformed.index.equals(prediction.index)
    mean, std = state.scaling_params['ast_alt_ratio']
    recovered = transformed.loc[7, 'ast_alt_ratio'] * std + mean
    assert recovered == pytest.approx(state.derived_impute_values['ast_alt_ratio'])
    train = transform_features(raw, loaded)
    assert train[DERIVED_COLUMNS].mean().to_numpy() == pytest.approx(np.zeros(5), abs=1e-10)
    # Ratios are computed before AST and ALT are log-transformed.
    lower, upper = state.clip_bounds['SGOT_AST']
    ast = raw.loc[7, 'SGOT_AST'].clip(lower, upper)
    lower, upper = state.clip_bounds['SGOT_ALT']
    alt = raw.loc[7, 'SGOT_ALT'].clip(lower, upper)
    assert train.loc[7, 'ast_alt_ratio'] * std + mean == pytest.approx(ast / alt)
    baseline = fit_preprocessor(raw, **kwargs)
    assert not set(DERIVED_COLUMNS) & set(transform_features(raw, baseline).columns)
