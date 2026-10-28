import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import FIGURES_DIR, RANDOM_STATE
from data_pipeline import run_pipeline
from evaluation import compute_metrics, plot_roc_curve
from tuning import results_table, tune_on_validation

FIGURE_DIR = FIGURES_DIR / "logistic_regression"

# degree 1 is plain (linear) logistic regression; degree 2 adds squares and
# pairwise interactions. They are scored on the validation split, so the
# linear-vs-polynomial choice is made by the data rather than assumed.
PARAM_GRID = {
    "poly__degree": [1, 2],
    "clf__C": [0.001, 0.01, 0.1, 1],
}

# Degree 2 expands 29 features to ~460, about 3 GB per copy of the training
# data, so only a few candidates are fitted in parallel
N_JOBS = 2


def build_model() -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        # Squares and products are not standardized, which would make the L2
        # penalty uneven across terms; copy=False avoids a second large matrix
        ("scale", StandardScaler(copy=False)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def best_per_degree(table: pd.DataFrame) -> pd.DataFrame:
    """Best validation candidate for each polynomial degree, for the linear-vs-polynomial comparison."""
    best = table.loc[table.groupby("degree")["val_roc_auc"].idxmax()]
    return best.assign(overfit_gap=best.train_roc_auc - best.val_roc_auc)


def main():
    print("Loading and preprocessing data...", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()

    search = tune_on_validation(
        build_model(), PARAM_GRID, X_train, y_train, X_val, y_val, n_jobs=N_JOBS,
    )
    table = results_table(search).rename(columns=lambda c: c.split("__")[-1])
    print("\nAll configurations (validation):")
    print(table.to_string(index=False))
    print("\nLinear vs polynomial (best C per degree):")
    print(best_per_degree(table)[
        ["degree", "C", "train_roc_auc", "val_roc_auc", "val_accuracy", "overfit_gap", "fit_time_s"]
    ].to_string(index=False))
    print("\nBest parameters:", search.best_params_)

    # Held-out test score, reported once for the refitted best model
    model = search.best_estimator_
    proba = model.predict_proba(X_test)[:, 1]
    print("\nTest metrics:")
    for metric, value in compute_metrics(y_test, proba).items():
        print(f"  {metric}: {round(value, 4) if isinstance(value, float) else value}")

    # Coefficients are per standard deviation of each (expanded) feature
    names = model["poly"].get_feature_names_out(X_train.columns)
    coefs = pd.Series(model["clf"].coef_[0], index=names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
    print("\nTop 10 coefficients (log-odds per SD, odds ratio):")
    print(pd.DataFrame({"coef": top, "odds_ratio": np.exp(top)}).round(3).to_string())

    path = plot_roc_curve(y_test, {"Logistic regression": proba}, FIGURE_DIR / "roc_curve.png")
    print(f"\nROC curve saved to {path}")


if __name__ == "__main__":
    main()
