import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from config import FIGURES_DIR, RANDOM_STATE
from data_pipeline import run_pipeline
from evaluation import compute_metrics, plot_roc_curve
from tuning import results_table, tune_on_validation

FIGURE_DIR = FIGURES_DIR / "decision_tree"

# Hyperparameter grid, evaluated on the validation split
PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [6, 12, 18, 24],
    "min_samples_leaf": [100, 200, 400, 800],
}


def main():
    print("Loading and preprocessing data...", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()

    search = tune_on_validation(
        DecisionTreeClassifier(random_state=RANDOM_STATE), PARAM_GRID,
        X_train, y_train, X_val, y_val,
    )
    print("\nTop 10 configurations (validation):")
    print(results_table(search).head(10).to_string(index=False))
    print("\nBest parameters:", search.best_params_)

    # Held-out test score, reported once for the refitted best model
    tree = search.best_estimator_
    print(f"Best tree: {tree.get_n_leaves()} leaves, depth {tree.get_depth()}")
    proba = tree.predict_proba(X_test)[:, 1]
    print("\nTest metrics:")
    for metric, value in compute_metrics(y_test, proba).items():
        print(f"  {metric}: {round(value, 4) if isinstance(value, float) else value}")

    importances = pd.Series(tree.feature_importances_, index=X_train.columns)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).to_string())

    path = plot_roc_curve(y_test, {"Decision tree": proba}, FIGURE_DIR / "roc_curve.png")
    print(f"\nROC curve saved to {path}")


if __name__ == "__main__":
    main()
