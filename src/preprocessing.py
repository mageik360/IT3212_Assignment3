from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from feature_engineering import DERIVED_COLUMNS, extract_features


@dataclass
class PreprocessingState:
    """Fitted parameters and schema; can be persisted with joblib.dump/load."""

    input_columns: list[str]
    feature_order: list[str]
    placeholders: dict[str, list]
    mappings: dict[str, dict]
    categories: dict[str, list]
    log_columns: list[str]
    clip_bounds: dict[str, tuple[float, float]]
    impute_values: dict[str, float]
    scaling_params: dict[str, tuple[float, float]]
    target_name: str
    target_mapping: dict
    extract_derived_features: bool = False
    derived_impute_values: dict[str, float] = field(default_factory=dict)


def _prepare_features(df, input_columns, placeholders, mappings, categories):
    _check_columns(df, input_columns)
    features = df.loc[:, input_columns].copy()
    handle_placeholders(features, placeholders)
    encode_columns(features, mappings)
    one_hot_encode(features, categories)
    return features


def fit_preprocessor(
    X_train: pd.DataFrame,
    *,
    placeholders: dict[str, list],
    mappings: dict[str, dict],
    categories: dict[str, list],
    continuous_columns: list[str],
    log_columns: list[str],
    target_name: str,
    target_mapping: dict,
    extract_derived_features: bool = False,
) -> PreprocessingState:
    """Fit on raw training predictors only, without modifying the input."""
    if target_name in X_train.columns:
        raise ValueError(f"Remove target '{target_name}' before fitting preprocessing.")
    features = _prepare_features(
        X_train, list(X_train.columns), placeholders, mappings, categories
    )
    clip_bounds = compute_clip_bounds(features, continuous_columns)
    clip_extreme_values(features, clip_bounds)
    impute_values = compute_impute_values(features, list(placeholders))
    impute_missing(features, impute_values)
    derived_impute_values = {}
    scaling_columns = list(continuous_columns)
    if extract_derived_features:
        features = extract_features(features)
        derived_impute_values = compute_impute_values(features, DERIVED_COLUMNS)
        if not np.isfinite(list(derived_impute_values.values())).all():
            raise ValueError("Cannot impute a derived feature with no valid training values.")
        impute_missing(features, derived_impute_values)
        scaling_columns.extend(DERIVED_COLUMNS)
    log_transform(features, log_columns)
    scaling_params = compute_scaling_params(features, scaling_columns)
    return PreprocessingState(
        input_columns=list(X_train.columns),
        feature_order=list(features.columns),
        placeholders=deepcopy(placeholders),
        mappings=deepcopy(mappings),
        categories=deepcopy(categories),
        log_columns=list(log_columns),
        clip_bounds=clip_bounds,
        impute_values=impute_values,
        scaling_params=scaling_params,
        target_name=target_name,
        target_mapping=deepcopy(target_mapping),
        extract_derived_features=extract_derived_features,
        derived_impute_values=derived_impute_values,
    )


def transform_features(
    raw_features: pd.DataFrame, state: PreprocessingState
) -> pd.DataFrame:
    """Transform raw records using saved state, without fitting or dropping rows.

    The target is not required. Extra columns are ignored; missing required
    predictors raise an error. Input data and its row indices are preserved.
    """
    features = _prepare_features(
        raw_features, state.input_columns, state.placeholders,
        state.mappings, state.categories,
    )
    clip_extreme_values(features, state.clip_bounds)
    impute_missing(features, state.impute_values)
    if state.extract_derived_features:
        features = extract_features(features)
        impute_missing(features, state.derived_impute_values)
    log_transform(features, state.log_columns)
    scale_columns(features, state.scaling_params)
    return features.loc[:, state.feature_order]


def _check_columns(df: pd.DataFrame, columns) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {sorted(missing)}")


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows from the DataFrame.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The same DataFrame, with duplicate rows removed.
    """
    df.drop_duplicates(inplace=True)
    return df


def handle_placeholders(df: pd.DataFrame, placeholders: dict[str, list]) -> pd.DataFrame:
    """
    Replace placeholder values with NaN, per column.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    placeholders (dict[str, list]): Placeholder values to replace, per column.

    Returns:
    pd.DataFrame: The same DataFrame, with placeholders replaced by NaN.
    """
    _check_columns(df, placeholders)
    for column, values in placeholders.items():
        # np.nan keeps the column float; pd.NA would turn it into object dtype
        df[column] = df[column].replace(values, np.nan)
    return df


def encode_columns(df: pd.DataFrame, mappings: dict[str, dict]) -> pd.DataFrame:
    """
    Encode columns with fixed value mappings, per column.

    The mappings learn nothing from the data, so this is safe to run on the
    full dataset before splitting.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    mappings (dict[str, dict]): Mapping from original to encoded value, per column.

    Returns:
    pd.DataFrame: The same DataFrame, with the columns encoded.

    Raises:
    ValueError: If a column contains values that are not in its mapping.
    """
    _check_columns(df, mappings)
    for column, mapping in mappings.items():
        unmapped = set(df[column].dropna().unique()) - set(mapping)
        if unmapped:
            raise ValueError(f"Column '{column}' has values not in mapping: {sorted(unmapped)}")
        df[column] = df[column].map(mapping)
    return df


def one_hot_encode(df: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """
    One-hot encode categorical columns, per column.

    Each column is replaced by one 0/1 column per category, named
    "<column>_<category>". The first category in each list is the reference
    category and gets no column (equivalent to drop_first=True), so a binary
    column becomes a single 0/1 column. The categories are fixed, so this is
    safe to run on the full dataset before splitting.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    categories (dict[str, list]): Expected categories per column, reference category first.

    Returns:
    pd.DataFrame: The same DataFrame, with the columns one-hot encoded.

    Raises:
    ValueError: If a column contains values (including NaN) that are not in its category list.
    """
    _check_columns(df, categories)
    for column, values in categories.items():
        unexpected = set(df[column].unique()) - set(values)
        if unexpected:
            raise ValueError(f"Column '{column}' has values not in categories: {sorted(unexpected, key=str)}")
        for value in values[1:]:
            df[f"{column}_{value}"] = (df[column] == value).astype(int)
        df.drop(columns=column, inplace=True)
    return df


def clip_extreme_values(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """
    Clip values to (lower, upper) bounds, per column.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    bounds (dict[str, tuple[float, float]]): (lower, upper) bounds per column.

    Returns:
    pd.DataFrame: The same DataFrame, with values clipped.
    """
    _check_columns(df, bounds)
    for column, (lower, upper) in bounds.items():
        df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def compute_clip_bounds(
    df: pd.DataFrame,
    columns: list[str],
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999
) -> dict[str, tuple[float, float]]:
    """
    Compute percentile-based clip bounds per column.

    Call this on the training set only, and reuse the bounds for the test set
    to avoid data leakage.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list[str]): Columns to compute bounds for.
    lower_quantile (float): Quantile used as the lower bound.
    upper_quantile (float): Quantile used as the upper bound.

    Returns:
    dict[str, tuple[float, float]]: (lower, upper) bounds per column.
    """
    _check_columns(df, columns)
    quantiles = df[columns].quantile([lower_quantile, upper_quantile])
    return {c: (quantiles.loc[lower_quantile, c], quantiles.loc[upper_quantile, c]) for c in columns}


def compute_impute_values(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """
    Compute the median per column, used to fill missing values.

    Call this on the training set only, and reuse the values for the
    validation and test sets to avoid data leakage.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list[str]): Columns to compute medians for.

    Returns:
    dict[str, float]: Median per column.
    """
    _check_columns(df, columns)
    return df[columns].median().to_dict()


def impute_missing(df: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    """
    Fill missing values with a fixed value, per column.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    values (dict[str, float]): Fill value per column, from compute_impute_values.

    Returns:
    pd.DataFrame: The same DataFrame, with missing values filled.
    """
    _check_columns(df, values)
    for column, value in values.items():
        df[column] = df[column].fillna(value)
    return df


def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Apply log(1 + x) to reduce right skew, per column.

    The transform learns nothing from the data, so this is safe to run on the
    full dataset before splitting. Run it after clipping, so extreme values
    are already handled.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list[str]): Columns to transform. Values must be greater than -1.

    Returns:
    pd.DataFrame: The same DataFrame, with the columns log-transformed.
    """
    _check_columns(df, columns)
    if (df[columns] <= -1).any().any():
        raise ValueError("log_transform requires all values to be greater than -1.")
    df[columns] = np.log1p(df[columns])
    return df


def compute_scaling_params(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    """
    Compute mean and standard deviation per column for standardization.

    Call this on the training set only, and reuse the parameters for the test
    set to avoid data leakage. NaN values are ignored.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list[str]): Columns to compute parameters for.

    Returns:
    dict[str, tuple[float, float]]: (mean, std) per column.
    """
    _check_columns(df, columns)
    return {c: (df[c].mean(), df[c].std()) for c in columns}


def scale_columns(df: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """
    Standardize columns to zero mean and unit variance: (x - mean) / std.

    Note: modifies the input DataFrame in place.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    params (dict[str, tuple[float, float]]): (mean, std) per column, from compute_scaling_params.

    Returns:
    pd.DataFrame: The same DataFrame, with the columns standardized.
    """
    _check_columns(df, params)
    for column, (mean, std) in params.items():
        # A constant column has std 0; only center it to avoid division by zero
        df[column] = (df[column] - mean) / (std if std > 0 else 1)
    return df

