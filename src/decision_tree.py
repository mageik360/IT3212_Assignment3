from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier

from data_pipeline import RANDOM_STATE, run_pipeline

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures" / "decision_tree"

# Hyperparameter grid, evaluated on the validation split
PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [6, 12, 18, 24],
    "min_samples_leaf": [100, 200, 400, 800],
}
SCORING = ["roc_auc", "accuracy", "f1"]


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Return accuracy, F1 and ROC-AUC for a fitted binary classifier."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def tune_decision_tree(X_train, y_train, X_val, y_val) -> GridSearchCV:
    """
    Grid search over PARAM_GRID, scoring each candidate on the fixed validation
    split instead of k-fold CV. The best candidate (by ROC-AUC) is refitted on
    train + validation combined.
    """
    X_search = pd.concat([X_train, X_val])
    y_search = pd.concat([y_train, y_val])
    # -1 = always in training, 0 = the single validation fold
    split = PredefinedSplit(np.r_[np.full(len(X_train), -1), np.zeros(len(X_val))])

    search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        scoring=SCORING,
        refit="roc_auc",
        cv=split,
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
    )
    return search.fit(X_search, y_search)


def results_table(search: GridSearchCV) -> pd.DataFrame:
    """One row per candidate with train accuracy and validation scores, best first."""
    cv = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "criterion": cv["param_criterion"],
        "max_depth": cv["param_max_depth"],
        "min_samples_leaf": cv["param_min_samples_leaf"],
        "train_acc": cv["mean_train_accuracy"],
        **{f"val_{name}": cv[f"mean_test_{name}"] for name in SCORING},
        "fit_time_s": cv["mean_fit_time"],
    })
    return table.sort_values("val_roc_auc", ascending=False)


def plot_roc_curve(model, X: pd.DataFrame, y: pd.Series, name: str = "roc_curve.png") -> Path:
    """Save the ROC curve of a fitted binary classifier and return the file path."""
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"Decision tree (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier (AUC = 0.5)")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    print("Loading and preprocessing data...", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()

    search = tune_decision_tree(X_train, y_train, X_val, y_val)
    print("\nTop 10 configurations (validation):")
    print(results_table(search).head(10).to_string(index=False))
    print("\nBest parameters:", search.best_params_)

    # Held-out test score, reported once for the refitted best model
    tree = search.best_estimator_
    print("\nTest:", {name: round(score, 4) for name, score in evaluate(tree, X_test, y_test).items()})

    importances = pd.Series(tree.feature_importances_, index=X_train.columns)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).to_string())

    print(f"\nROC curve saved to {plot_roc_curve(tree, X_test, y_test)}")


if __name__ == "__main__":
    main()
