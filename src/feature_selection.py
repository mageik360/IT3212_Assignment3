"""Feature-group ablation: does each group of inputs earn its place?

Run from the project root:
    python src/feature_selection.py
    python src/feature_selection.py --models tree logreg_linear   # a subset
    python src/feature_selection.py --with-nn                      # adds the shallow network

A diagnostic, not a new model: the final pipeline keeps all 29 inputs and no
reported model is retrained. For each group of related inputs, the model is
refitted on the training split without that group, using fixed hyperparameters
already tuned in tasks 5 and 7, and scored on the validation split. The test
split is never used.

The pipeline runs once. Preprocessing is per column (clip bounds, medians, logs
and scaling are each fitted on one column, and derived features are computed
from raw sources before any of that), so dropping processed columns gives the
same values as never creating them, on exactly the same rows.

Dropping a group also drops the derived features computed from it, otherwise
its information returns through them (e.g. dropping body size drops BMI).

Uncertainty is a paired bootstrap over validation rows: every configuration and
model is scored on the same resamples, and the 95% interval of the difference
in average precision to the all-features baseline is reported. It covers
validation sampling only, not training randomness (the tree and network seeds
are fixed), so a narrow interval does not mean the model itself is stable.
"""

import argparse
import gc
import json
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from bagging_boosting import TUNED_LOGREG, TUNED_TREE
from config import FIGURES_DIR, RANDOM_STATE, RESULTS_DIR
from data_pipeline import CATEGORIES
from data_pipeline import run_pipeline
from evaluation import compute_metrics
from feature_engineering import DERIVED_COLUMNS
from logistic_regression import build_model as build_logreg
from plotting import save_figure

FIGURE_DIR = FIGURES_DIR / "feature_selection"
RESULT_DIR = RESULTS_DIR / "feature_selection"
NN_CONFIG = RESULTS_DIR / "neural_network" / "selected_config.json"

# Linear logistic regression (degree 1; C barely matters at this size, see task 5).
# It cannot build products or ratios itself, so it shows most directly whether
# the derived features add information beyond their sources.
LINEAR_LOGREG = {"poly__degree": 1, "clf__C": 1}

N_BOOTSTRAP = 500
BASELINE = "none"


def _one_hot(column: str) -> list[str]:
    # Same naming as preprocessing.one_hot_encode; the reference level has no column
    return [f"{column}_{value}" for value in CATEGORIES[column][1:]]


# Processed column names per group. Every processed column is in exactly one group
GROUPS = {
    "age": ["age"],
    "sex": _one_hot("sex"),
    "body_size": ["height", "weight", "waistline"],
    "sight": ["sight_left", "sight_right"],
    "hearing": ["hear_left", "hear_right"],
    "blood_pressure": ["SBP", "DBP"],
    "glucose": ["BLDS"],
    "lipids": ["tot_chole", "HDL_chole", "LDL_chole", "triglyceride"],
    "kidney_blood": ["hemoglobin", "urine_protein", "serum_creatinine"],
    "liver": ["SGOT_AST", "SGOT_ALT", "gamma_GTP"],
    "smoking": _one_hot("SMK_stat_type_cd"),
    "derived": list(DERIVED_COLUMNS),
}

# Sources of each derived feature, as in feature_engineering.extract_features
DERIVED_SOURCES = {
    "bmi": ["height", "weight"],
    "waist_to_height_ratio": ["waistline", "height"],
    "ast_alt_ratio": ["SGOT_AST", "SGOT_ALT"],
    "non_hdl_cholesterol": ["tot_chole", "HDL_chole"],
    "pulse_pressure": ["SBP", "DBP"],
}

MODEL_LABELS = {
    "logreg_linear": "Linear logistic regression",
    "logreg_poly": "Polynomial logistic regression",
    "tree": "Decision tree",
    "nn": "Shallow neural network",
}
SKLEARN_MODELS = ["logreg_linear", "logreg_poly", "tree"]
# The network only tests the derived features, the claim made about it in the report
NN_CONFIGS = [BASELINE, "derived"]


def dropped_columns(group: str) -> list[str]:
    """Columns removed for a group: its own, plus derived features built from them."""
    if group == BASELINE:
        return []
    own = GROUPS[group]
    derived = [name for name, sources in DERIVED_SOURCES.items()
               if set(sources) & set(own) and name not in own]
    return own + derived


def check_groups(columns) -> None:
    """Fail early if the groups do not match the processed columns one to one."""
    grouped = [c for cols in GROUPS.values() for c in cols]
    missing = sorted(set(grouped) - set(columns))
    ungrouped = sorted(set(columns) - set(grouped))
    if missing or ungrouped or len(grouped) != len(set(grouped)):
        raise ValueError(f"Groups do not match the features: missing {missing}, "
                         f"ungrouped {ungrouped}, or a column is in two groups")


def build_estimator(model: str):
    if model == "logreg_linear":
        return build_logreg().set_params(**LINEAR_LOGREG)
    if model == "logreg_poly":
        return build_logreg().set_params(**TUNED_LOGREG)
    if model == "tree":
        return DecisionTreeClassifier(**TUNED_TREE, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {model}")


def bootstrap_counts(n: int, n_resamples: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    """How often each row appears in each resample, shape (n_resamples, n).

    Drawn once and shared by every configuration and model, so differences are
    paired. Counts are equivalent to resampled indices but let the AP of every
    resample be computed from a single sort.
    """
    rng = np.random.default_rng(random_state)
    counts = np.empty((n_resamples, n), dtype=np.uint16)
    for b in range(n_resamples):
        counts[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return counts


def bootstrap_ap(y_true, proba, counts: np.ndarray, chunk: int = 50) -> np.ndarray:
    """Average precision on each bootstrap resample, given as row counts.

    Matches sklearn's average_precision_score with the counts as sample weights:
    tied scores form one threshold, and AP sums precision times the recall step.
    """
    order = np.argsort(-np.asarray(proba), kind="mergesort")
    scores = np.asarray(proba)[order]
    positive = np.asarray(y_true)[order].astype(float)
    # Last position of each run of tied scores
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    result = np.empty(len(counts))
    for start in range(0, len(counts), chunk):
        weights = counts[start:start + chunk][:, order].astype(float)
        tp = np.cumsum(weights * positive, axis=1)[:, ends]
        seen = np.cumsum(weights, axis=1)[:, ends]
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        recall = tp / tp[:, -1:]
        result[start:start + chunk] = np.sum(np.diff(recall, prepend=0, axis=1) * precision, axis=1)
    return result


def fit_sklearn(model: str, X_train, y_train, X_val):
    estimator = build_estimator(model)
    start = perf_counter()
    estimator.fit(X_train, y_train)
    fit_time = perf_counter() - start
    return estimator.predict_proba(X_val)[:, 1], fit_time


def fit_nn(X_train, y_train, X_val, y_val, *, epochs: int, device: str):
    """The tuned shallow network, early-stopped on validation as in task 5g."""
    import neural_network as nn_module

    config = json.loads(NN_CONFIG.read_text())
    start = perf_counter()
    model, _, summary = nn_module.train_model(
        X_train, y_train, X_val, y_val, hidden_dim=config["hidden_dim"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        batch_size=config["batch_size"], epochs=epochs, device=device, verbose=False,
    )
    fit_time = perf_counter() - start
    print(f"  Best epoch {summary['best_epoch']} of {summary['epochs_run']}")
    _, proba = nn_module.predict(model, nn_module.make_dataset(X_val, y_val))
    return proba, fit_time


def save_table(rows: list[dict], previous: pd.DataFrame | None) -> pd.DataFrame:
    """Write all rows so far; rows of models not run this time are kept."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if previous is not None:
        table = pd.concat([previous[~previous.model.isin(table.model)], table], ignore_index=True)
    table.to_csv(RESULT_DIR / "ablation.csv", index=False)
    return table


def plot_deltas(table: pd.DataFrame, path):
    """One panel per model: change in validation AP without each group, with 95% CI."""
    groups = [g for g in GROUPS if g in set(table.config)]
    models = [m for m in MODEL_LABELS if m in set(table.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 0.4 * len(groups) + 1.6),
                             sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        rows = table[(table.model == model) & (table.config != BASELINE)].set_index("config")
        rows = rows.reindex([g for g in groups if g in rows.index])
        y = [groups.index(g) for g in rows.index]
        ax.errorbar(rows.delta_ap, y,
                    xerr=[rows.delta_ap - rows.ci_low, rows.ci_high - rows.delta_ap],
                    fmt="o", color="#2a6fb0", ecolor="#7a9cc0", elinewidth=2, capsize=3, markersize=6)
        ax.axvline(0, color="gray", linewidth=1, linestyle="--")
        ax.grid(axis="x", alpha=0.25)
        ax.set(title=MODEL_LABELS[model], xlabel="Δ validation AP vs all features")
    axes[0][0].set(yticks=range(len(groups)), yticklabels=groups)
    axes[0][0].invert_yaxis()
    return save_figure(fig, path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=SKLEARN_MODELS,
                        help="Models to run (default: the three sklearn models)")
    parser.add_argument("--with-nn", action="store_true",
                        help="Also run the shallow network, derived features on/off only")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--epochs", type=int, default=50, help="Network epoch limit")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args(argv)
    if args.with_nn and "nn" not in args.models:
        args.models = [*args.models, "nn"]
    return args


def main(argv=None):
    args = parse_args(argv)
    print("Loading and preprocessing data...", flush=True)
    # The test split is discarded here, so it cannot be used by accident
    X_train, X_val, _, y_train, y_val, _, _ = run_pipeline()
    check_groups(X_train.columns)

    counts = bootstrap_counts(len(y_val), args.n_bootstrap)
    path = RESULT_DIR / "ablation.csv"
    previous = pd.read_csv(path) if path.exists() else None
    rows = []

    for model in args.models:
        configs = NN_CONFIGS if model == "nn" else [BASELINE, *GROUPS]
        base_ap = base_boot = None
        for config in configs:
            drop = dropped_columns(config)
            print(f"\n=== {MODEL_LABELS[model]}, dropped: {config} ({len(drop)} columns) ===",
                  flush=True)
            X_tr, X_va = X_train.drop(columns=drop), X_val.drop(columns=drop)
            if model == "nn":
                proba, fit_time = fit_nn(X_tr, y_train, X_va, y_val,
                                         epochs=args.epochs, device=args.device)
            else:
                proba, fit_time = fit_sklearn(model, X_tr, y_train, X_va)
            # Free the degree 2 expansion before the next fit
            del X_tr, X_va
            gc.collect()

            metrics = compute_metrics(y_val, proba)
            (tn, fp), (fn, tp) = metrics.pop("confusion_matrix")
            boot = bootstrap_ap(y_val, proba, counts)
            if config == BASELINE:
                base_ap, base_boot = metrics["average_precision"], boot
            delta = boot - base_boot
            row = {
                "model": model, "config": config, "dropped_columns": ";".join(drop),
                "n_features": X_train.shape[1] - len(drop),
                **{f"val_{name}": value for name, value in metrics.items()},
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "delta_ap": metrics["average_precision"] - base_ap,
                "ci_low": np.percentile(delta, 2.5), "ci_high": np.percentile(delta, 97.5),
                "n_bootstrap": len(counts), "fit_time_s": fit_time,
            }
            rows.append(row)
            print(f"Fit time {fit_time:.1f} s, validation AP {row['val_average_precision']:.4f}, "
                  f"Δ AP {row['delta_ap']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
            # Saved after every fit, so a crash keeps the finished runs
            table = save_table(rows, previous)

    print("\nΔ validation AP vs all features, 95% paired bootstrap CI (all saved runs):")
    summary = table.assign(
        delta=lambda t: t.apply(lambda r: f"{r.delta_ap:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                                axis=1))
    print(summary.pivot(index="config", columns="model", values="delta")
          .reindex([c for c in [BASELINE, *GROUPS] if c in set(table.config)])
          .fillna("").to_string())
    print("\nAll runs:")
    print(table[["model", "config", "n_features", "val_average_precision", "val_roc_auc",
                 "delta_ap", "ci_low", "ci_high", "fit_time_s"]].round(4).to_string(index=False))
    print(f"\nResults saved to {path}")
    print(f"Figure saved to {plot_deltas(table, FIGURE_DIR / 'delta_ap.png')}")


if __name__ == "__main__":
    main()
