"""Interpretable features calculated from cleaned measurements in original units."""

import numpy as np
import pandas as pd

DERIVED_COLUMNS = [
    "bmi", "waist_to_height_ratio", "ast_alt_ratio",
    "non_hdl_cholesterol", "pulse_pressure",
]
SOURCE_COLUMNS = [
    "height", "weight", "waistline", "SGOT_AST", "SGOT_ALT",
    "tot_chole", "HDL_chole", "SBP", "DBP",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with five derived features, preserving rows and originals.

    Call after placeholder handling/cleaning, but BEFORE logs or scaling.
    Height and waist must be in cm and weight in kg. No parameters are fitted.
    Missing/non-finite inputs and nonpositive ratio denominators produce NaN;
    the preprocessing pipeline fills these using training-set medians.
    """
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Feature extraction requires columns: {sorted(missing)}")
    result = df.copy()
    source = df[SOURCE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    height = source["height"].where(source["height"] > 0)
    alt = source["SGOT_ALT"].where(source["SGOT_ALT"] > 0)
    result["bmi"] = source["weight"] / (height / 100) ** 2
    result["waist_to_height_ratio"] = source["waistline"] / height
    result["ast_alt_ratio"] = source["SGOT_AST"] / alt
    result["non_hdl_cholesterol"] = source["tot_chole"] - source["HDL_chole"]
    result["pulse_pressure"] = source["SBP"] - source["DBP"]
    result[DERIVED_COLUMNS] = result[DERIVED_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    return result
