import pandas as pd
from sklearn.model_selection import train_test_split

from data import load_dataset
import preprocessing as pp

TARGET = "DRK_YN"
RANDOM_STATE = 42

# Placeholder codes that mean "missing", found during EDA
PLACEHOLDERS = {
    "waistline": [999],
    "sight_left": [9.9],
    "sight_right": [9.9],
}

ENCODINGS = {
    "DRK_YN": {"N": 0, "Y": 1},
    # Readable labels, used as one-hot column names below
    "SMK_stat_type_cd": {1: "never", 2: "not_anymore", 3: "current"},
}

# Categorical features to one-hot encode, reference category first
CATEGORIES = {
    "sex": ["Female", "Male"],
    "SMK_stat_type_cd": ["never", "not_anymore", "current"],
}

# Right-skewed lab values (skew > 1.5 after clipping)
LOG_COLUMNS = ["gamma_GTP", "SGOT_AST", "SGOT_ALT", "BLDS", "triglyceride", "serum_creatinine"]

# Continuous features. Excludes one-hot columns, ordinal codes (hear_*, urine_protein)
# and the target.
CONTINUOUS_COLUMNS = [
    "age", "height", "weight", "waistline", "sight_left", "sight_right",
    "SBP", "DBP", "BLDS", "tot_chole", "HDL_chole", "LDL_chole", "triglyceride",
    "hemoglobin", "serum_creatinine", "SGOT_AST", "SGOT_ALT", "gamma_GTP",
]


def run_pipeline(*, extract_derived_features: bool = True) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series,
    pp.PreprocessingState,
]:
    """
    Load the dataset and run all preprocessing steps.

    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test, state: The 80/10/10
    splits and fitted preprocessing state for transforming future raw records.
    X contains predictors (including smoking status); y contains encoded DRK_YN.
    Each X/y pair retains matching original row indices.
    Five derived features are included by default. Set extract_derived_features
    to False for a baseline comparison using the same splits.
    """
    dataset = load_dataset()

    # 1. Dataset-wide steps. These learn nothing from the data, so they are
    #    safe to run before splitting.
    pp.drop_duplicates(dataset)
    pp.encode_columns(dataset, {TARGET: ENCODINGS[TARGET]})

    # 2. Split into train, validation and test sets, stratified on the target.
    X = dataset.drop(columns=TARGET)
    y = dataset[TARGET].copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    # 3. Fit only on raw training predictors, then reuse exactly the same
    #    transformation for all splits and future records.
    state = pp.fit_preprocessor(
        X_train,
        placeholders=PLACEHOLDERS,
        mappings={column: mapping for column, mapping in ENCODINGS.items()
                  if column != TARGET},
        categories=CATEGORIES,
        continuous_columns=CONTINUOUS_COLUMNS,
        log_columns=LOG_COLUMNS,
        target_name=TARGET,
        target_mapping=ENCODINGS[TARGET],
        extract_derived_features=extract_derived_features,
    )
    X_train = pp.transform_features(X_train, state)
    X_val = pp.transform_features(X_val, state)
    X_test = pp.transform_features(X_test, state)

    return X_train, X_val, X_test, y_train, y_val, y_test, state


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, state = run_pipeline()
    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        print(f"{name}: {len(X):,} rows, {X.isna().sum().sum()} missing predictors, "
              f"{y.mean() * 100:.2f}% drinkers")

if __name__ == "__main__":
    main()
