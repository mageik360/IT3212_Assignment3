"""Task 7: boosting and bagging on both classical base models.

Run from the project root:
    python src/bagging_boosting.py
    python src/bagging_boosting.py --models logreg logreg_bagging   # a subset

Each base model from task 5 (decision tree, degree 2 logistic regression) is
fitted three ways: once on its own, boosted with AdaBoost and bagged with
BaggingClassifier. Keeping the boosting and bagging algorithms fixed across the
two base models means any difference comes from the base model itself.

Boosting (AdaBoost, SAMME) fits members one after another. After each round the
weights of misclassified samples are increased, so the next member focuses on
the current errors, and each member's vote is weighted by its accuracy. It
mainly reduces bias, so the trees are weak learners (depth 3). The tuned logistic
regression is kept as is: it is already a strong, stable model, so little gain
is expected.

Bagging draws a bootstrap sample (n rows with replacement) for each member,
fits the members independently and averages their probabilities. It mainly
reduces variance, so the trees are grown deep (unstable, low bias). Every
member sees all features, which separates plain bagging from a random forest.

All models are fitted on the training split only, because the validation split
is needed to choose the number of boosting rounds: the round with the highest
validation average precision is used, which acts as early stopping. Bagging cannot overfit
by adding members, so all members are used, and its validation curve only shows
where the gain levels off. Test scores therefore differ slightly from task 5,
where the final models were refitted on train + validation.

The neural network is not ensembled. It is already a flexible, low-bias model
(little for boosting to remove), each fit is expensive, and the deep network
of task 9 covers that model family separately.
"""

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import FIGURES_DIR, RANDOM_STATE, RESULTS_DIR
from data_pipeline import run_pipeline
from evaluation import compute_metrics, plot_pr_curve, plot_roc_curve
from logistic_regression import N_JOBS as LOGREG_N_JOBS
from logistic_regression import build_model as build_logreg
from plotting import save_figure

FIGURE_DIR = FIGURES_DIR / "bagging_boosting"
RESULT_DIR = RESULTS_DIR / "bagging_boosting"

# Best configurations from task 5 (validation search, see the report)
TUNED_TREE = {"criterion": "entropy", "max_depth": 12, "min_samples_leaf": 100}
TUNED_LOGREG = {"poly__degree": 2, "clf__C": 0.01}

# Depth 3 trees are weak learners that can still model a few interactions
BOOSTED_TREE_DEPTH = 3
# Deep trees for bagging; a minimum leaf size keeps 100 trees in memory
BAGGED_TREE_MIN_LEAF = 10
# Shrinks each member's vote, which needs more rounds but generalizes better
BOOSTING_LEARNING_RATE = 0.5

# Ensemble sizes. A degree 2 logistic regression takes about two minutes per
# fit, so its ensembles are kept small
TREE_ROUNDS = 200
TREE_MEMBERS = 100
LOGREG_ROUNDS = 10
LOGREG_MEMBERS = 10

BASE_LABELS = {"tree": "Decision tree", "logreg": "Logistic regression"}


class MeanWeightedLogisticRegression(LogisticRegression):
    """Logistic regression that rescales sample weights to a mean of 1.

    AdaBoost normalizes the sample weights to sum to 1. The L2 penalty does not
    scale with the weights, so C would otherwise act about n times stronger and
    shrink every coefficient towards zero. Rescaling keeps the tuned C.
    """

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight) * len(sample_weight) / np.sum(sample_weight)
        return super().fit(X, y, sample_weight=sample_weight)


class ProgressAdaBoost(AdaBoostClassifier):
    """AdaBoost that prints the current round, since sklearn's has no verbose option."""

    def fit(self, X, y, sample_weight=None):
        self._start = perf_counter()
        super().fit(X, y, sample_weight=sample_weight)
        print()  # end the progress line, also when boosting stops early
        return self

    def _boost(self, iboost, X, y, sample_weight, random_state):
        print(f"\r  Boosting round {iboost + 1}/{self.n_estimators} "
              f"({perf_counter() - self._start:.0f} s)", end="", flush=True)
        return super()._boost(iboost, X, y, sample_weight, random_state)


@dataclass(frozen=True)
class Model:
    name: str
    base: str    # key of BASE_LABELS
    method: str  # "single", "boosting" or "bagging"
    build: Callable[[], BaseEstimator]

    @property
    def label(self) -> str:
        prefix = {"single": "", "boosting": "Boosted ", "bagging": "Bagged "}[self.method]
        base = BASE_LABELS[self.base]
        return prefix + (base.lower() if prefix else base)


@dataclass
class Result:
    model: Model
    curve: pd.DataFrame  # validation ROC AUC and average precision per ensemble size
    n_used: int
    test_proba: np.ndarray
    row: dict


def tuned_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(**TUNED_TREE, random_state=RANDOM_STATE)


def boosted_tree() -> AdaBoostClassifier:
    weak = DecisionTreeClassifier(criterion=TUNED_TREE["criterion"], max_depth=BOOSTED_TREE_DEPTH)
    return ProgressAdaBoost(weak, n_estimators=TREE_ROUNDS,
                            learning_rate=BOOSTING_LEARNING_RATE, random_state=RANDOM_STATE)


def bagged_tree() -> BaggingClassifier:
    deep = DecisionTreeClassifier(criterion=TUNED_TREE["criterion"], min_samples_leaf=BAGGED_TREE_MIN_LEAF)
    return BaggingClassifier(deep, n_estimators=TREE_MEMBERS, n_jobs=-1, random_state=RANDOM_STATE)


def polynomial_logreg(wrap: Callable = lambda clf: clf) -> Pipeline:
    """Task 5's tuned degree 2 pipeline, with the classifier optionally wrapped in an ensemble.

    The polynomial expansion and scaling stay outside the ensemble, so they are
    fitted once rather than once per member.
    """
    model = build_logreg().set_params(**TUNED_LOGREG)
    return model.set_params(clf=wrap(model["clf"]))


def boosted_logreg(clf: LogisticRegression) -> AdaBoostClassifier:
    base = MeanWeightedLogisticRegression(**clf.get_params())
    return ProgressAdaBoost(base, n_estimators=LOGREG_ROUNDS,
                            learning_rate=BOOSTING_LEARNING_RATE, random_state=RANDOM_STATE)


def bagged_logreg(clf: LogisticRegression) -> BaggingClassifier:
    # Bootstrap counts are passed as sample weights that sum to n, so C is unaffected
    return BaggingClassifier(clf, n_estimators=LOGREG_MEMBERS,
                             n_jobs=LOGREG_N_JOBS, random_state=RANDOM_STATE)


def build_models() -> list[Model]:
    """The six models: each base model on its own, boosted and bagged."""
    return [
        Model("tree", "tree", "single", tuned_tree),
        Model("tree_boosting", "tree", "boosting", boosted_tree),
        Model("tree_bagging", "tree", "bagging", bagged_tree),
        Model("logreg", "logreg", "single", polynomial_logreg),
        Model("logreg_boosting", "logreg", "boosting", lambda: polynomial_logreg(boosted_logreg)),
        Model("logreg_bagging", "logreg", "bagging", lambda: polynomial_logreg(bagged_logreg)),
    ]


def staged_proba(model, X):
    """Yield positive-class probabilities using the first 1, 2, ... ensemble members.

    A single model yields once. The last value equals model.predict_proba.
    """
    if isinstance(model, Pipeline):
        X, model = model[:-1].transform(X), model[-1]
    if isinstance(model, AdaBoostClassifier):
        for proba in model.staged_predict_proba(X):
            yield proba[:, 1]
    elif isinstance(model, BaggingClassifier):
        # Bagging averages member probabilities, so keep a running mean
        X, total = np.asarray(X), 0.0
        for k, (member, features) in enumerate(zip(model.estimators_, model.estimators_features_), 1):
            total = total + member.predict_proba(X[:, features])[:, 1]
            yield total / k
    else:
        yield model.predict_proba(X)[:, 1]


def proba_at(model, X, n_members: int) -> np.ndarray:
    """Probabilities of the ensemble truncated to its first n_members."""
    for k, proba in enumerate(staged_proba(model, X), 1):
        if k == n_members:
            return proba
    raise ValueError(f"Ensemble has fewer than {n_members} members")


def validation_curve(model, X_val, y_val) -> pd.DataFrame:
    scores = [(roc_auc_score(y_val, proba), average_precision_score(y_val, proba))
              for proba in staged_proba(model, X_val)]
    curve = pd.DataFrame(scores, columns=["val_roc_auc", "val_average_precision"])
    return curve.assign(n_members=range(1, len(curve) + 1))[["n_members", *curve.columns]]


def select_size(method: str, curve: pd.DataFrame) -> int:
    """Boosting: the round with the best validation average precision. Otherwise: all members."""
    if method == "boosting":
        return int(curve.loc[curve.val_average_precision.idxmax(), "n_members"])
    return int(curve.n_members.max())


def run(model: Model, X_train, y_train, X_val, y_val, X_test, y_test) -> Result:
    print(f"\n=== {model.label} ===", flush=True)
    estimator = model.build()
    start = perf_counter()
    estimator.fit(X_train, y_train)
    fit_time = perf_counter() - start

    curve = validation_curve(estimator, X_val, y_val)
    n_used = select_size(model.method, curve)
    val_auc = curve.val_roc_auc.iloc[n_used - 1]
    val_ap = curve.val_average_precision.iloc[n_used - 1]
    print(f"Fit time: {fit_time:.1f} s, members used: {n_used} of {len(curve)}, "
          f"validation ROC AUC: {val_auc:.4f}, validation average precision: {val_ap:.4f}")

    # Held-out test score, reported once at the selected ensemble size
    start = perf_counter()
    proba = proba_at(estimator, X_test, n_used)
    predict_time = perf_counter() - start
    metrics = compute_metrics(y_test, proba)
    print("Test metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {round(value, 4) if isinstance(value, float) else value}")

    (tn, fp), (fn, tp) = metrics.pop("confusion_matrix")
    row = {"model": model.name, "base": model.base, "method": model.method,
           "n_members": len(curve), "n_used": n_used, "val_roc_auc": val_auc,
           "val_average_precision": val_ap,
           **{f"test_{name}": value for name, value in metrics.items()},
           "tn": tn, "fp": fp, "fn": fn, "tp": tp,
           "fit_time_s": fit_time, "predict_time_s": predict_time}
    return Result(model, curve, n_used, proba, row)


def plot_validation_curves(results: list[Result], path):
    """One panel per base model: ensemble size against validation average precision."""
    bases = list(dict.fromkeys(r.model.base for r in results))
    fig, axes = plt.subplots(1, len(bases), figsize=(6 * len(bases), 4.5), squeeze=False)
    for ax, base in zip(axes[0], bases):
        for r in (r for r in results if r.model.base == base):
            if r.model.method == "single":
                ax.axhline(r.curve.val_average_precision.iloc[0], linestyle="--", color="gray", label=r.model.label)
                continue
            (line,) = ax.plot(r.curve.n_members, r.curve.val_average_precision, label=r.model.label)
            ax.plot(r.n_used, r.curve.val_average_precision.iloc[r.n_used - 1], "o", color=line.get_color())
        ax.set(title=BASE_LABELS[base], xlabel="Ensemble members", ylabel="Validation average precision")
        ax.legend(loc="lower right")
    return save_figure(fig, path)


def save_results(results: list[Result]) -> pd.DataFrame:
    """Write metrics and validation curves; rows of models not run this time are kept."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([r.row for r in results])
    path = RESULT_DIR / "metrics.csv"
    if path.exists():
        previous = pd.read_csv(path)
        table = pd.concat([previous[~previous.model.isin(table.model)], table], ignore_index=True)
    table.to_csv(path, index=False)
    for r in results:
        r.curve.to_csv(RESULT_DIR / f"curve_{r.model.name}.csv", index=False)
    return table


def parse_args(argv=None) -> argparse.Namespace:
    names = [model.name for model in build_models()]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=names, default=names,
                        help="Models to run (default: all six)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("Loading and preprocessing data...", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()

    results = [run(model, X_train, y_train, X_val, y_val, X_test, y_test)
               for model in build_models() if model.name in args.models]

    table = save_results(results)
    columns = ["model", "n_used", "val_average_precision", "val_roc_auc", "test_roc_auc",
               "test_accuracy", "test_precision", "test_recall", "fit_time_s"]
    print("\nSummary (all saved runs):")
    print(table[columns].round(4).to_string(index=False))

    path = plot_validation_curves(results, FIGURE_DIR / "validation_curves.png")
    print(f"\nValidation curves saved to {path}")
    for base in dict.fromkeys(r.model.base for r in results):
        probas = {r.model.label: r.test_proba for r in results if r.model.base == base}
        path = plot_roc_curve(y_test, probas, FIGURE_DIR / f"roc_curve_{base}.png")
        print(f"ROC curve saved to {path}")
        path = plot_pr_curve(y_test, probas, FIGURE_DIR / f"pr_curve_{base}.png")
        print(f"Precision-recall curve saved to {path}")


if __name__ == "__main__":
    main()
