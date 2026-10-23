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


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset and run all preprocessing steps.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test sets (80/10/10).
    """
    dataset = load_dataset()

    # 1. Dataset-wide steps. These learn nothing from the data, so they are
    #    safe to run before splitting.
    pp.drop_duplicates(dataset)
    pp.handle_placeholders(dataset, PLACEHOLDERS)
    pp.encode_columns(dataset, ENCODINGS)
    pp.one_hot_encode(dataset, CATEGORIES)

    # 2. Split into train, validation and test sets, stratified on the target.
    #    .copy() so the in-place steps below do not touch shared memory.
    train_set, temp_set = train_test_split(
        dataset, test_size=0.2, random_state=RANDOM_STATE, stratify=dataset[TARGET]
    )
    val_set, test_set = train_test_split(
        temp_set, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_set[TARGET]
    )
    train_set, val_set, test_set = train_set.copy(), val_set.copy(), test_set.copy()
    splits = (train_set, val_set, test_set)

    # 3. Learned steps. Parameters are computed on the training set only,
    #    then applied to all three sets.

    # 3a. Clip extreme values
    clip_bounds = pp.compute_clip_bounds(train_set, CONTINUOUS_COLUMNS)
    for split in splits:
        pp.clip_extreme_values(split, clip_bounds)

    # 3b. Impute missing values (from the placeholders) with the median
    impute_values = pp.compute_impute_values(train_set, list(PLACEHOLDERS))
    for split in splits:
        pp.impute_missing(split, impute_values)

    # 3c. Log transform skewed features (learns nothing, but must follow clipping)
    for split in splits:
        pp.log_transform(split, LOG_COLUMNS)

    # 3d. Standardize. Must come last, so the parameters describe the final data.
    scaling_params = pp.compute_scaling_params(train_set, CONTINUOUS_COLUMNS)
    for split in splits:
        pp.scale_columns(split, scaling_params)

    return train_set, val_set, test_set

def main():
    train_set, val_set, test_set = run_pipeline()
    for name, split in [("train", train_set), ("val", val_set), ("test", test_set)]:
        print(f"{name}: {len(split):,} rows, {split.isna().sum().sum()} missing, "
              f"{split[TARGET].mean() * 100:.2f}% drinkers")

if __name__ == "__main__":
    main()
