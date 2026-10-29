#boosting
import xgboost as xgb

#bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#rest
from config import FIGURES_DIR, RANDOM_STATE
from data_pipeline import run_pipeline
from evaluation import compute_metrics, plot_roc_curve
import pandas as pd

FIGURE_DIR = FIGURES_DIR / "bagging_boosting"

def evaluate(model, X: pd.DataFrame, y: pd.Series, name: str) -> None:
    """Print the shared evaluation metrics and save the ROC curve for a fitted binary classifier."""
    proba = model.predict_proba(X)[:, 1]
    print("\nTest metrics:")
    for metric, value in compute_metrics(y, proba).items():
        print(f"  {metric}: {round(value, 4) if isinstance(value, float) else value}")

    path = plot_roc_curve(y, {name: proba}, FIGURE_DIR / f"{name.lower()}_roc_curve.png")
    print(f"\nROC curve saved to {path}")


def boost():
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()

    model = xgb.XGBClassifier(
        eval_metric="auc",
        tree_method="hist",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    evaluate(model, X_test, y_test, "Boosting")

def bag():
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()
    
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, "Bagging")


def main():
    boost()
    bag()


if __name__ == "__main__":
    main()

