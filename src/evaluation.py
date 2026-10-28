"""Model-agnostic evaluation, so every model is scored on the same metrics.

Functions take predicted probabilities for the positive class rather than a
model, so sklearn estimators, torch networks and ensembles are all handled alike.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from plotting import save_figure


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Binary classification metrics; confusion matrix order is [0, 1].

    Accuracy measures overall correctness, precision/recall/F1 describe the
    positive class, and ROC AUC measures ranking across thresholds.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else float("nan"),
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
