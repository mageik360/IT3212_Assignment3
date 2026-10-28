"""Hyperparameter search for sklearn estimators on the fixed validation split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit

SCORING = ["roc_auc", "accuracy", "f1"]


def tune_on_validation(
    estimator, param_grid: dict, X_train, y_train, X_val, y_val,
    *, scoring=SCORING, refit: str = "roc_auc", verbose: int = 3,
) -> GridSearchCV:
    """
    Grid search over param_grid, scoring each candidate on the fixed validation
    split instead of k-fold CV. The best candidate (by `refit`) is refitted on
    train + validation combined.
    """
    X_search = pd.concat([X_train, X_val])
    y_search = pd.concat([y_train, y_val])
    # -1 = always in training, 0 = the single validation fold
    split = PredefinedSplit(np.r_[np.full(len(X_train), -1), np.zeros(len(X_val))])

    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=split,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True,
    )
    return search.fit(X_search, y_search)


def results_table(search: GridSearchCV) -> pd.DataFrame:
    """One row per candidate with train and validation scores, best first."""
    cv = pd.DataFrame(search.cv_results_)
    scoring = list(search.scorer_)
    params = cv.filter(like="param_").rename(columns=lambda c: c.removeprefix("param_"))
    table = params.assign(
        **{f"train_{name}": cv[f"mean_train_{name}"] for name in scoring},
        **{f"val_{name}": cv[f"mean_test_{name}"] for name in scoring},
        fit_time_s=cv["mean_fit_time"],
    )
    return table.sort_values(f"val_{search.refit}", ascending=False)
