import numpy as np
import pandas as pd


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

