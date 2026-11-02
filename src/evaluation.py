"""Model-agnostic evaluation, so every model is scored on the same metrics.

Functions take predicted probabilities for the positive class rather than a
model, so sklearn estimators, torch networks and ensembles are all handled alike.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
