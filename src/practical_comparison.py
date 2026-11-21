"""Task 10 revision: a validation-chosen decision threshold and repeated timing.

Run from the project root:
    python src/practical_comparison.py                        # both parts
    python src/practical_comparison.py --parts threshold      # part A only (minutes)
    python src/practical_comparison.py --parts timing --repeats 3

Added after the first submission, in response to the grader's comment on the
integrated comparison: "Compare precision and coverage at a validation-chosen
threshold, with complete repeated timing boundaries." Nothing is retuned; every
model uses the hyperparameters already selected in tasks 5 and 9 and is trained
on the training split only, so all four models are compared on the same basis.

Part A, threshold. The discussion recommends raising the decision threshold so
that participants are only placed in drinking groups when the model is
confident. Here the rule is made explicit and tested on data it was not chosen
on: for each target precision, the threshold is the lowest one whose
*validation* precision reaches the target. It is then frozen and applied to the
test set, which reports
    precision      share of the people placed in drinking groups who drink,
    recall         share of all drinkers who are placed in drinking groups,
    share placed   share of all participants placed in drinking groups,
with 95% bootstrap intervals over test rows. The resamples are shared by every
model and target, so the intervals are paired. The networks are loaded from the
seed-42 runs saved by `advanced_neural_network.py --mode compare` when they
exist (the runs reported in task 9), and trained with seed 42 otherwise.

Part B, timing. Each model is trained --repeats times and the mean and standard
deviation are reported. The timed boundaries are the same for every model:
    fit        starts with the preprocessed training DataFrame in memory and
               stops when the trained model is returned. Included: polynomial
               expansion and rescaling (logistic regression); building the
               network, copying the data to tensors and the device, every epoch
               with its validation pass, and restoring the best weights
               (networks). Excluded: reading the CSV and the shared
               preprocessing, which are identical for every model.
    inference  probabilities for the full test set from the test DataFrame,
               including tensor conversion and device transfer for the
               networks. One untimed warm-up call is made first, then
               --inference-repeats calls are timed and averaged.
GPU work is synchronized before every clock reading. The sklearn models are
deterministic, so their repeats only measure timing noise. The networks use
seeds 42, 43, ... as in task 9, so their epoch counts (early stopping), and
therefore their times, also vary between repeats. Hyperparameter search times
are not repeated here; they are single measurements in tasks 5 and 9.
Run everything on one machine; environment.json records which.
"""

import argparse
import gc
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier

from bagging_boosting import TUNED_LOGREG, TUNED_TREE
from config import RANDOM_STATE, RESULTS_DIR
from data_pipeline import run_pipeline
from logistic_regression import build_model as build_logreg

RESULT_DIR = RESULTS_DIR / "practical_comparison"
SHALLOW_CONFIG = RESULTS_DIR / "neural_network" / "selected_config.json"
SAVED_NN_DIR = RESULTS_DIR / "advanced_neural_network" / "compare"

MODELS = ["logreg", "tree", "shallow_nn", "deep_nn"]
MODEL_LABELS = {
    "logreg": "Logistic regression (degree 2)",
    "tree": "Decision tree",
    "shallow_nn": "Shallow neural network",
    "deep_nn": "Deep neural network (tuned)",
}
# Folder names used by advanced_neural_network.py --mode compare
SAVED_NN_NAMES = {"shallow_nn": "shallow", "deep_nn": "tuned_deep"}

TARGET_PRECISIONS = [0.80, 0.85, 0.90]
DEFAULT_THRESHOLD = 0.5
N_BOOTSTRAP = 1000
# A loaded network whose test AP differs more than this from its saved
# metrics.json is not the run reported in task 9
AP_TOLERANCE = 1e-4
# Same training settings as tasks 5g and 9
NN_EPOCHS = 50
NN_PATIENCE = 10


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def shallow_settings() -> dict:
    settings = {"hidden_dim": 256, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 1024}
    if SHALLOW_CONFIG.exists():
        settings.update(json.loads(SHALLOW_CONFIG.read_text()))
    return settings


def deep_config():
    from advanced_neural_network import ModelConfig

    path = SAVED_NN_DIR / "selected_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run advanced_neural_network.py --mode compare.")
    values = json.loads(path.read_text())
    return ModelConfig(**{**values, "hidden_dims": tuple(values["hidden_dims"])})


def fit_model(name: str, X_train, y_train, X_val, y_val, *, seed: int, device: str,
              epochs: int = NN_EPOCHS):
    """Train one model on the training split; returns (model, info)."""
    if name == "logreg":
        model = build_logreg().set_params(**TUNED_LOGREG).fit(X_train, y_train)
        return model, {}
    if name == "tree":
        model = DecisionTreeClassifier(**TUNED_TREE, random_state=RANDOM_STATE).fit(X_train, y_train)
        return model, {"leaves": model.get_n_leaves()}
    if name == "shallow_nn":
        from neural_network import train_model as train_shallow

        model, _, summary = train_shallow(
            X_train, y_train, X_val, y_val, **shallow_settings(), epochs=epochs,
            patience=NN_PATIENCE, device=device, random_state=seed, verbose=False,
        )
        return model, {"epochs_run": summary["epochs_run"], "device": summary["device"]}
    if name == "deep_nn":
        from advanced_neural_network import train_model as train_deep

        model, _, summary = train_deep(
            X_train, y_train, X_val, y_val, config=deep_config(), epochs=epochs,
            patience=NN_PATIENCE, device=device, random_state=seed, verbose=False,
        )
        return model, {"epochs_run": summary["epochs_run"], "device": summary["device"]}
    raise ValueError(f"Unknown model: {name}")


def load_saved_nn(name: str, n_features: int, device: str, seed: int = RANDOM_STATE):
    """The network saved by the task 9 comparison, or None if it is not on disk."""
    from advanced_neural_network import AdvancedNeuralNetwork, ModelConfig
    from neural_network import NeuralNetwork, select_device

    path = SAVED_NN_DIR / SAVED_NN_NAMES[name] / f"seed_{seed}" / "model.pt"
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu")
    config = saved["config"]
    if name == "shallow_nn":
        model = NeuralNetwork(n_features, config["hidden_dim"])
    else:
        fields = {k: v for k, v in config.items() if k in ModelConfig.__dataclass_fields__}
        model = AdvancedNeuralNetwork(
            n_features, ModelConfig(**{**fields, "hidden_dims": tuple(fields["hidden_dims"])}))
    model.load_state_dict(saved["state_dict"])
    model.feature_columns = saved["feature_columns"]
    return model.to(select_device(device)).eval()


def reported_test_ap(name: str, seed: int = RANDOM_STATE) -> float:
    """Test AP recorded next to a saved network, to check that model.pt matches it."""
    if name not in SAVED_NN_NAMES:
        return float("nan")
    path = SAVED_NN_DIR / SAVED_NN_NAMES[name] / f"seed_{seed}" / "metrics.json"
    if not path.exists():
        return float("nan")
    return float(json.loads(path.read_text())["test"]["average_precision"])


def predict_proba(model, X) -> np.ndarray:
    """Probability of being a drinker, for sklearn models and networks alike."""
    if isinstance(model, torch.nn.Module):
        from neural_network import make_dataset, predict

        if list(X.columns) != model.feature_columns:
            raise ValueError("Feature columns differ from those the network was trained on.")
        # Labels are only needed to build the dataset; they do not affect the output
        _, proba = predict(model, make_dataset(X, pd.Series(0, index=X.index)))
        return proba
    return model.predict_proba(X)[:, 1]


def synchronize(device: str | None):
    from neural_network import synchronize as sync

    if device:
        sync(torch.device(device))


# ---------------------------------------------------------------------------
# Part A: validation-chosen threshold
# ---------------------------------------------------------------------------

def threshold_for_precision(y_true, proba, target: float) -> float:
    """Lowest threshold whose precision on (y_true, proba) reaches the target.

    Predictions use proba >= threshold, as in evaluation.compute_metrics. The
    lowest such threshold places as many people as possible in drinking groups
    while meeting the target. NaN if no threshold reaches it.
    """
    precision, _, thresholds = precision_recall_curve(y_true, proba)
    reached = np.flatnonzero(precision[:-1] >= target)
    return float(thresholds[reached[0]]) if len(reached) else float("nan")


def rule_metrics(y_true, placed, weights=None) -> dict:
    """Precision, recall and share placed of a fixed rule, optionally weighted.

    `placed` marks the people the rule places in drinking groups. With bootstrap
    counts as weights, this gives the metrics on that resample.
    """
    y = np.asarray(y_true, dtype=float)
    placed = np.asarray(placed, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    tp = np.sum(w * placed * y)
    n_placed = np.sum(w * placed)
    return {
        "precision": tp / n_placed if n_placed > 0 else float("nan"),
        "recall": tp / np.sum(w * y),
        "share_placed": n_placed / np.sum(w),
    }


def threshold_rules(y_val, p_val, targets) -> list[tuple[str, float]]:
    """(rule name, threshold) pairs: the default 0.5 and one per target precision."""
    rules = [("default 0.5", DEFAULT_THRESHOLD)]
    rules += [(f"precision >= {t:.2f}", threshold_for_precision(y_val, p_val, t)) for t in targets]
    return rules


def evaluate_rules(y_val, y_test, probas: dict, targets, n_bootstrap: int,
                   random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """One row per model and rule; thresholds come from validation only."""
    y_val, y_test = np.asarray(y_val), np.asarray(y_test)
    rows, test_placed = [], []
    for name, (p_val, p_test) in probas.items():
        for rule, threshold in threshold_rules(y_val, p_val, targets):
            target = float(rule.split(">= ")[1]) if ">=" in rule else float("nan")
            row = {"model": name, "rule": rule, "target_precision": target, "threshold": threshold}
            if np.isnan(threshold):
                rows.append(row)
                test_placed.append(None)
                continue
            val = rule_metrics(y_val, p_val >= threshold)
            placed = p_test >= threshold
            test = rule_metrics(y_test, placed)
            row.update({f"val_{k}": v for k, v in val.items()})
            row.update({f"test_{k}": v for k, v in test.items()})
            row.update({
                "test_placed": int(placed.sum()),
                "test_true_positives": int((placed & (y_test == 1)).sum()),
                "test_false_positives": int((placed & (y_test == 0)).sum()),
                "test_false_negatives": int((~placed & (y_test == 1)).sum()),
            })
            rows.append(row)
            test_placed.append(placed)

    # Paired bootstrap over test rows: one set of resamples for every row
    rng = np.random.default_rng(random_state)
    samples = {i: [] for i, placed in enumerate(test_placed) if placed is not None}
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, len(y_test), len(y_test)), minlength=len(y_test))
        for i in samples:
            samples[i].append(rule_metrics(y_test, test_placed[i], counts))
    for i, values in samples.items():
        frame = pd.DataFrame(values)
        for metric in ["precision", "recall", "share_placed"]:
            # Precision is undefined on a resample where nobody is placed
            values = frame[metric].dropna()
            low, high = (np.percentile(values, [2.5, 97.5]) if len(values)
                         else (float("nan"), float("nan")))
            rows[i][f"test_{metric}_ci_low"] = low
            rows[i][f"test_{metric}_ci_high"] = high
    return pd.DataFrame(rows)


def threshold_probabilities(models, data, *, device: str, epochs: int) -> tuple[dict, list[dict]]:
    """Validation and test probabilities of each model, trained on the training split."""
    X_train, X_val, X_test, y_train, y_val, y_test = data
    probas, checks = {}, []
    for name in models:
        model = None
        if name in SAVED_NN_NAMES:
            model = load_saved_nn(name, X_train.shape[1], device)
            source = "saved seed-42 run (task 9)" if model is not None else None
        if model is None:
            print(f"  {MODEL_LABELS[name]}: training on the training split...", flush=True)
            model, _ = fit_model(name, X_train, y_train, X_val, y_val, seed=RANDOM_STATE,
                                 device=device, epochs=epochs)
            source = "trained now (seed 42)"
        p_val, p_test = predict_proba(model, X_val), predict_proba(model, X_test)
        probas[name] = (p_val, p_test)
        check = {"model": name, "source": source,
                 "val_average_precision": average_precision_score(y_val, p_val),
                 "test_average_precision": average_precision_score(y_test, p_test),
                 "reported_test_average_precision": reported_test_ap(name)}
        checks.append(check)
        print(f"  {MODEL_LABELS[name]}: {source}, validation AP "
              f"{check['val_average_precision']:.4f}, test AP "
              f"{check['test_average_precision']:.4f}", flush=True)
        reported = check["reported_test_average_precision"]
        if source.startswith("saved") and not np.isclose(
                reported, check["test_average_precision"], atol=AP_TOLERANCE):
            print(f"  WARNING: the saved metrics.json reports test AP {reported:.4f}. The "
                  "model.pt on disk is probably from another run; rerun "
                  "advanced_neural_network.py --mode compare or delete the saved model "
                  "to train it here.", flush=True)
        del model
        gc.collect()
    return probas, checks


def run_threshold_part(models, data, args, output_dir: Path) -> pd.DataFrame:
    print("\n=== Part A: threshold chosen on validation, evaluated on test ===", flush=True)
    *_, y_val, y_test = data
    probas, checks = threshold_probabilities(models, data, device=args.device, epochs=args.epochs)
    np.savez_compressed(
        output_dir / "probabilities.npz", y_val=np.asarray(y_val), y_test=np.asarray(y_test),
        **{f"{name}_{split}": p for name, (pv, pt) in probas.items()
           for split, p in [("val", pv), ("test", pt)]},
    )
    pd.DataFrame(checks).to_csv(output_dir / "model_check.csv", index=False)
    table = evaluate_rules(y_val, y_test, probas, args.targets, args.n_bootstrap)
    table.to_csv(output_dir / "threshold.csv", index=False)

    def fmt(row, metric):
        value = row.get(f"test_{metric}")
        if value is None or pd.isna(value):
            return "-"
        return (f"{value:.3f} [{row[f'test_{metric}_ci_low']:.3f}, "
                f"{row[f'test_{metric}_ci_high']:.3f}]")

    print("\nTest set, thresholds chosen on validation (95% bootstrap CI):")
    for _, row in table.iterrows():
        print(f"  {MODEL_LABELS[row.model]:<31} {row.rule:<17} t={row.threshold:.3f}  "
              f"precision {fmt(row, 'precision')}  recall {fmt(row, 'recall')}  "
              f"placed {fmt(row, 'share_placed')}  FP {row.get('test_false_positives', '-')}")
    return table


# ---------------------------------------------------------------------------
# Part B: repeated timing
# ---------------------------------------------------------------------------

def time_inference(model, X, device: str | None, repeats: int) -> float:
    """Mean seconds to predict X, after one untimed warm-up call."""
    predict_proba(model, X)
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = perf_counter()
        predict_proba(model, X)
        synchronize(device)
        times.append(perf_counter() - start)
    return float(np.mean(times))


def run_timing_part(models, data, args, output_dir: Path) -> pd.DataFrame:
    print(f"\n=== Part B: timing, {args.repeats} repeats per model ===", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data
    rows = []
    for name in models:
        for repeat in range(args.repeats):
            # Networks vary their seed as in task 9; the sklearn fits are deterministic
            seed = RANDOM_STATE + repeat if name in SAVED_NN_NAMES else RANDOM_STATE
            gc.collect()
            device = None
            if name in SAVED_NN_NAMES:
                from neural_network import select_device
                device = str(select_device(args.device))
                synchronize(device)
            start = perf_counter()
            model, info = fit_model(name, X_train, y_train, X_val, y_val, seed=seed,
                                    device=args.device, epochs=args.epochs)
            synchronize(device)
            fit_seconds = perf_counter() - start
            inference = time_inference(model, X_test, device, args.inference_repeats)
            test_ap = average_precision_score(y_test, predict_proba(model, X_test))
            rows.append({"model": name, "repeat": repeat, "seed": seed,
                         "fit_seconds": fit_seconds, "inference_seconds": inference,
                         "device": device or "cpu", "test_average_precision": test_ap, **info})
            print(f"  {MODEL_LABELS[name]}, repeat {repeat + 1}/{args.repeats}: fit "
                  f"{fit_seconds:.1f} s, inference {inference:.3f} s, test AP {test_ap:.4f}"
                  + (f", {info['epochs_run']} epochs" if "epochs_run" in info else ""), flush=True)
            del model
            # Saved after every fit, so an interrupted run keeps the finished ones
            pd.DataFrame(rows).to_csv(output_dir / "timing_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summary = runs.groupby("model", sort=False).agg(
        repeats=("fit_seconds", "size"),
        fit_seconds_mean=("fit_seconds", "mean"), fit_seconds_std=("fit_seconds", "std"),
        inference_seconds_mean=("inference_seconds", "mean"),
        inference_seconds_std=("inference_seconds", "std"),
        device=("device", "first"),
    ).reset_index()
    if "logreg" in set(summary.model):
        base = summary.set_index("model").fit_seconds_mean["logreg"]
        summary["fit_time_vs_logreg"] = summary.fit_seconds_mean / base
    summary.to_csv(output_dir / "timing_summary.csv", index=False)
    print("\nTiming summary (seconds, mean and sample standard deviation):")
    print(summary.round(3).to_string(index=False))
    return summary


def environment(args) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "torch": torch.__version__, "sklearn": sklearn.__version__,
        "cuda_gpu": gpu, "mps_available": torch.backends.mps.is_available(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "tree_settings": TUNED_TREE, "logreg_settings": TUNED_LOGREG,
        "shallow_settings": shallow_settings(),
        "deep_settings": asdict(deep_config()) if (SAVED_NN_DIR / "selected_config.json").exists()
        else None,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", choices=["threshold", "timing"],
                        default=["threshold", "timing"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--targets", nargs="+", type=float, default=TARGET_PRECISIONS,
                        help="Target precisions for the validation-chosen threshold")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--repeats", type=int, default=3, help="Fits per model for timing")
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=NN_EPOCHS, help="Network epoch limit")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args(argv)
    if min(args.repeats, args.inference_repeats, args.n_bootstrap, args.epochs) < 1:
        parser.error("repeats, inference repeats, bootstrap resamples and epochs must be positive")
    if not all(0 < t < 1 for t in args.targets):
        parser.error("target precisions must lie between 0 and 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading and preprocessing data (not timed)...", flush=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()
    data = (X_train, X_val, X_test, y_train, y_val, y_test)
    (args.output_dir / "environment.json").write_text(json.dumps(environment(args), indent=2) + "\n")
    if "threshold" in args.parts:
        run_threshold_part(args.models, data, args, args.output_dir)
    if "timing" in args.parts:
        run_timing_part(args.models, data, args, args.output_dir)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
