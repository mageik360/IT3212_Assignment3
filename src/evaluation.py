"""Model-agnostic evaluation, so every model is scored on the same metrics.

Functions take predicted probabilities for the positive class rather than a
model, so sklearn estimators, torch networks and ensembles are all handled alike.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from plotting import save_figure


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Binary classification metrics; confusion matrix order is [0, 1].

    Accuracy measures overall correctness, precision/recall/F1 describe the
    positive class, and ROC AUC and average precision measure ranking across
    thresholds. Average precision is the area under the precision-recall curve
    and is the metric all models are selected on.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if both_classes else float("nan"),
        "average_precision": average_precision_score(y_true, proba) if both_classes else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


# Ranking metrics that are bootstrapped; both are threshold free
BOOTSTRAP_METRICS = {"roc_auc": roc_auc_score, "average_precision": average_precision_score}


def bootstrap_scores(y_true, probas: dict, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Score every model on the same bootstrap resamples of the test set.

    Each replicate draws n test rows with replacement and recomputes the metrics,
    so the spread of a metric over replicates estimates how much it would vary
    on another test set of the same size. All models share the resamples, which
    makes their differences paired. Returns one row per replicate and model.
    """
    y_true = np.asarray(y_true)
    probas = {name: np.asarray(proba) for name, proba in probas.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        for name, proba in probas.items():
            rows.append({"replicate": replicate, "model": name,
                         **{metric: score(y_true[idx], proba[idx])
                            for metric, score in BOOTSTRAP_METRICS.items()}})
    return pd.DataFrame(rows)


def percentile_ci(values, level: float = 0.95) -> dict:
    """Bootstrap standard error and percentile confidence interval."""
    values = np.asarray(values)
    alpha = (1 - level) / 2
    return {"se": values.std(ddof=1), "ci_low": np.quantile(values, alpha),
            "ci_high": np.quantile(values, 1 - alpha)}


def bootstrap_summary(scores: pd.DataFrame, reference: dict | None = None,
                      level: float = 0.95) -> pd.DataFrame:
    """Confidence intervals per model and metric, from bootstrap_scores output.

    reference maps a model to the model it is compared with. The difference
    model minus reference is computed on each shared resample, and its interval
    shows whether the gap is larger than test set noise (it excludes 0).
    """
    reference = reference or {}
    wide = scores.pivot(index="replicate", columns="model")
    rows = []
    for model in scores.model.unique():
        for metric in BOOTSTRAP_METRICS:
            row = {"model": model, "metric": metric, **percentile_ci(wide[metric][model], level)}
            other = reference.get(model)
            if other is not None and other != model and other in wide[metric]:
                diff = percentile_ci(wide[metric][model] - wide[metric][other], level)
                row.update({"reference": other, "diff_mean": (wide[metric][model] - wide[metric][other]).mean(),
                            **{f"diff_{key}": value for key, value in diff.items()}})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_roc_curve(y_true, probas: dict, path) -> Path:
    """Save ROC curves for one or more models, given as {label: probabilities}."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier (AUC = 0.5)")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    return save_figure(fig, path)


def plot_pr_curve(y_true, probas: dict, path) -> Path:
    """Save precision-recall curves for one or more models, given as {label: probabilities}."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, linewidth=2,
                label=f"{label} (AP = {average_precision_score(y_true, proba):.3f})")
    # A random classifier has precision equal to the share of positives
    baseline = np.mean(y_true)
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Random classifier (AP = {baseline:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.legend(loc="lower left")
    return save_figure(fig, path)
