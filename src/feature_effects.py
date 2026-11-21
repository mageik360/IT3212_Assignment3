"""Total effect of each original feature on the polynomial logistic regression.

Run from the project root:
    python src/feature_effects.py

In the degree 2 model a feature enters through its linear term, its square and
its products with every other feature, and through any derived feature built
from it (e.g. weight through BMI). A single coefficient is therefore not the
effect of the feature. Here the effect is measured on the predictions instead:
every test person's raw value of one feature is set to a low and then a high
value, with all their other raw measurements kept, and the rows are passed
through the full preprocessing (derived features, clipping, logs, scaling) and
the model. The difference is the total change in prediction for that person.

Continuous features move from their training-set 25th to 75th percentile;
categorical and ordinal features between the categories listed in CONTRASTS.
The effect varies from person to person, since the interactions make it depend
on everything else about them, so the spread (10th-90th percentile) and the
means for women and men are reported as well.

The model is the tuned degree 2 logistic regression refitted on training and
validation, as in the Models section. The test labels are not used.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as pp
from bagging_boosting import TUNED_LOGREG
from config import RESULTS_DIR
from data import load_dataset
from data_pipeline import CONTINUOUS_COLUMNS, PLACEHOLDERS, TARGET, run_pipeline
from logistic_regression import build_model

RESULT_DIR = RESULTS_DIR / "feature_effects"

# (feature, low, high, description) for features without a quartile contrast
CONTRASTS = [
    ("sex", "Female", "Male", "female to male"),
    ("SMK_stat_type_cd", 1.0, 2.0, "never to former smoker"),
    ("SMK_stat_type_cd", 1.0, 3.0, "never to current smoker"),
    ("hear_left", 1.0, 2.0, "normal to abnormal"),
    ("hear_right", 1.0, 2.0, "normal to abnormal"),
    ("urine_protein", 1.0, 2.0, "level 1 to 2"),
]


def quartile_contrasts(raw_train: pd.DataFrame) -> list[tuple]:
    """25th to 75th percentile of each continuous feature, placeholders removed."""
    clean = raw_train[CONTINUOUS_COLUMNS].copy()
    pp.handle_placeholders(clean, {k: v for k, v in PLACEHOLDERS.items() if k in clean})
    rows = []
    for column in CONTINUOUS_COLUMNS:
        low, high = clean[column].quantile([0.25, 0.75])
        rows.append((column, float(low), float(high), f"{low:g} to {high:g}"))
    return rows


def total_effects(model, raw: pd.DataFrame, transform, contrasts, groups=None) -> pd.DataFrame:
    """Change in log-odds and probability when one raw feature moves from low to high.

    `transform` maps raw rows to model inputs. `groups` is an optional Series of
    labels (e.g. sex), for the mean change in log-odds within each group.
    """
    rows = []
    for feature, low, high, description in contrasts:
        logit = {}
        for name, value in [("low", low), ("high", high)]:
            changed = raw.copy()
            changed[feature] = value
            logit[name] = model.decision_function(transform(changed))
        delta = logit["high"] - logit["low"]
        delta_p = 1 / (1 + np.exp(-logit["high"])) - 1 / (1 + np.exp(-logit["low"]))
        row = {
            "feature": feature, "contrast": description, "low": low, "high": high,
            "mean_delta_logit": delta.mean(),
            "odds_ratio_of_mean": float(np.exp(delta.mean())),
            "delta_logit_p10": np.percentile(delta, 10),
            "delta_logit_p90": np.percentile(delta, 90),
            "share_positive": float(np.mean(delta > 0)),
            "mean_delta_probability": delta_p.mean(),
        }
        if groups is not None:
            for label, mask in groups.groupby(groups).groups.items():
                row[f"mean_delta_logit_{label}"] = delta[raw.index.get_indexer(mask)].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("Loading and preprocessing data...", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, state = run_pipeline()
    raw = load_dataset().drop(columns=TARGET)
    raw_train, raw_test = raw.loc[X_train.index], raw.loc[X_test.index]

    print("Fitting the tuned degree 2 logistic regression on train + validation...", flush=True)
    model = build_model().set_params(**TUNED_LOGREG)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    def transform(rows):
        return pp.transform_features(rows, state)

    # The unchanged rows must reproduce the model inputs exactly
    pd.testing.assert_frame_equal(transform(raw_test), X_test)

    contrasts = quartile_contrasts(raw_train) + CONTRASTS
    table = total_effects(model, raw_test, transform, contrasts, groups=raw_test["sex"])
    table = table.reindex(table.mean_delta_logit.abs().sort_values(ascending=False).index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "total_effects.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nTotal effect on the test set, ordered by size (log-odds; probability in points):")
    print(table.assign(mean_delta_probability=table.mean_delta_probability * 100)
          .round(3).to_string(index=False))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
