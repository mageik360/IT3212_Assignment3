"""Paired bootstrap of the test set: how certain are the differences between models?

Run from the project root, after `python src/practical_comparison.py --parts threshold`:
    python src/paired_bootstrap.py

Reuses the test probabilities saved by practical_comparison.py, so nothing is
retrained. Every model is scored on the same bootstrap resamples of the test
rows, and the 95% interval of each difference between two models is taken from
the per-resample differences. Because the models make correlated errors on the
same people, this interval is much narrower than one built from each model's
own interval.

The interval covers which people ended up in the test set, not training
randomness. The saved models are fitted on the training split only (the
networks with seed 42), so the point estimates differ slightly from the
refitted models in the report; the intervals describe the size of the
uncertainty around such differences.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_STATE, RESULTS_DIR
from evaluation import compute_metrics
from feature_selection import bootstrap_ap, bootstrap_counts

INPUT = RESULTS_DIR / "practical_comparison" / "probabilities.npz"
RESULT_DIR = RESULTS_DIR / "paired_bootstrap"
MODELS = ["logreg", "tree", "shallow_nn", "deep_nn"]
# (a, b): the difference is a - b
PAIRS = [("shallow_nn", "logreg"), ("deep_nn", "logreg"), ("deep_nn", "shallow_nn"),
         ("logreg", "tree"), ("shallow_nn", "tree")]
METRICS = ["average_precision", "roc_auc", "accuracy", "precision", "recall", "false_positives"]
N_BOOTSTRAP = 1000
THRESHOLD = 0.5


def bootstrap_roc_auc(y_true, proba, counts: np.ndarray, chunk: int = 50) -> np.ndarray:
    """ROC-AUC on each bootstrap resample, given as row counts.

    Trapezoidal area under the ROC curve with tied scores as one point, as in
    sklearn's roc_auc_score with the counts as sample weights.
    """
    order = np.argsort(-np.asarray(proba), kind="mergesort")
    scores = np.asarray(proba)[order]
    positive = np.asarray(y_true)[order].astype(float)
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    result = np.empty(len(counts))
    for start in range(0, len(counts), chunk):
        weights = counts[start:start + chunk][:, order].astype(float)
        tp = np.cumsum(weights * positive, axis=1)[:, ends]
        fp = np.cumsum(weights * (1 - positive), axis=1)[:, ends]
        tpr = np.hstack([np.zeros((len(tp), 1)), tp / tp[:, -1:]])
        fpr = np.hstack([np.zeros((len(fp), 1)), fp / fp[:, -1:]])
        result[start:start + chunk] = np.sum(np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2,
                                             axis=1)
    return result


def bootstrap_threshold_metrics(y_true, proba, counts: np.ndarray,
                                threshold: float = THRESHOLD) -> dict[str, np.ndarray]:
    """Accuracy, precision, recall and false positives at a threshold, per resample."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(proba) >= threshold
    weights = counts.astype(float)
    n = weights.sum(axis=1)
    tp = weights @ (pred & y)
    fp = weights @ (pred & ~y)
    tn = weights @ (~pred & ~y)
    positives = weights @ y
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"accuracy": (tp + tn) / n, "precision": tp / (tp + fp),
                "recall": tp / positives, "false_positives": fp}


def point_metrics(y_true, proba) -> dict[str, float]:
    m = compute_metrics(y_true, proba, THRESHOLD)
    (_, fp), _ = m["confusion_matrix"]
    return {**{k: m[k] for k in METRICS if k != "false_positives"}, "false_positives": fp}


def bootstrap_metrics(y_true, proba, counts) -> dict[str, np.ndarray]:
    return {"average_precision": bootstrap_ap(y_true, proba, counts),
            "roc_auc": bootstrap_roc_auc(y_true, proba, counts),
            **bootstrap_threshold_metrics(y_true, proba, counts)}


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(values[~np.isnan(values)], [2.5, 97.5]))


def paired_bootstrap(y_true, probas: dict, pairs, n_bootstrap: int,
                     random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model and per-difference 95% intervals over shared test resamples."""
    counts = bootstrap_counts(len(y_true), n_bootstrap, random_state)
    point = {name: point_metrics(y_true, p) for name, p in probas.items()}
    boot = {name: bootstrap_metrics(y_true, p, counts) for name, p in probas.items()}

    models = []
    for name in probas:
        for metric in METRICS:
            low, high = interval(boot[name][metric])
            models.append({"model": name, "metric": metric, "estimate": point[name][metric],
                           "ci_low": low, "ci_high": high})
    differences = []
    for a, b in pairs:
        for metric in METRICS:
            diff = boot[a][metric] - boot[b][metric]
            low, high = interval(diff)
            differences.append({
                "model_a": a, "model_b": b, "metric": metric,
                "difference": point[a][metric] - point[b][metric], "ci_low": low, "ci_high": high,
                "share_a_better": float(np.mean(diff < 0 if metric == "false_positives" else diff > 0)),
                "excludes_zero": bool(low > 0 or high < 0),
                # Half-width an unpaired interval would have, for comparison
                "unpaired_half_width": 1.96 * float(np.sqrt(np.nanvar(boot[a][metric])
                                                            + np.nanvar(boot[b][metric]))),
            })
    return pd.DataFrame(models), pd.DataFrame(differences)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args(argv)
    if args.n_bootstrap < 2:
        parser.error("need at least two bootstrap resamples")
    return args


def main(argv=None):
    args = parse_args(argv)
    saved = np.load(args.input)
    y_test = saved["y_test"]
    probas = {name: saved[f"{name}_test"] for name in MODELS if f"{name}_test" in saved}
    pairs = [(a, b) for a, b in PAIRS if a in probas and b in probas]
    print(f"Paired bootstrap over {len(y_test):,} test rows, {args.n_bootstrap} resamples, "
          f"models: {', '.join(probas)}", flush=True)

    models, differences = paired_bootstrap(y_test, probas, pairs, args.n_bootstrap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models.to_csv(args.output_dir / "model_intervals.csv", index=False)
    differences.to_csv(args.output_dir / "paired_differences.csv", index=False)

    print("\nPer model (95% CI):")
    for row in models.itertuples():
        print(f"  {row.model:<11} {row.metric:<18} {row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]")
    print("\nPaired differences a - b (95% CI; unpaired half-width for comparison):")
    for row in differences.itertuples():
        print(f"  {row.model_a:>10} - {row.model_b:<10} {row.metric:<18} {row.difference:+.4f} "
              f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]  unpaired ±{row.unpaired_half_width:.4f}"
              + ("  *" if row.excludes_zero else ""))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
