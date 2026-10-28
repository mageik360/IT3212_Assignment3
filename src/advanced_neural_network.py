"""Exercise 9: a regularized deep MLP with reproducible random search.

Examples (from the project root):
    python src/advanced_neural_network.py --epochs 50
    python src/advanced_neural_network.py --mode tune --trials 30
    python src/advanced_neural_network.py --mode compare --device mps
    python src/advanced_neural_network.py --hidden-dims 128 64 --no-batch-norm

Tuning justification: depth/width control interaction capacity; dropout limits
reliance on individual units; AdamW weight decay regularizes weights; learning
rate controls optimization stability; batch size trades gradient noise against
throughput; batch normalization tests whether hidden-state normalization helps.
Rates and decay are sampled logarithmically because their useful scales span
orders of magnitude. These ranges are experimental choices, not known optima.

All runs use the existing features and fixed data splits. Checkpoints minimize
validation BCE; configurations maximize validation ROC AUC (ties: lower BCE).
Test data never enters search. Architecture/training flags such as --dropout
and --batch-size apply only to train mode; tune/compare use the fixed search
space. --seed controls model/search randomness, not the fixed data split.
Compare uses this seed and the next two (42, 43, 44 by default).
Comparison repeats all three models with matching
seeds, without selecting the best seed. Seeded runs need not be bitwise identical
across devices. The shallow baseline retains its original Adam optimizer.
"""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline import RANDOM_STATE, run_pipeline
from neural_network import (
    evaluate_model,
    make_dataset,
    predict,
    save_training_plots,
    select_device,
    synchronize,
)
from neural_network import (
    train_model as train_baseline,
)


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    batch_norm: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024

    def __post_init__(self):
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")


DEFAULT_CONFIG = ModelConfig()


class AdvancedNeuralNetwork(nn.Module):
    """Dense hidden blocks followed by a single binary-classification logit."""

    def __init__(self, input_dim: int, config: ModelConfig = DEFAULT_CONFIG):
        super().__init__()
        layers = []
        for width in config.hidden_dims:
            layers.append(nn.Linear(input_dim, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.extend([nn.ReLU(), nn.Dropout(config.dropout)])
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    config=DEFAULT_CONFIG,
    epochs=100,
    patience=10,
    device="auto",
    random_state=RANDOM_STATE,
    verbose=True,
):
    """Return (best model, epoch history, summary), using validation only."""
    if min(epochs, patience) < 1:
        raise ValueError("epochs and patience must be positive.")
    if not X_train.columns.equals(X_val.columns):
        raise ValueError("Training and validation feature columns must match in order.")
    train_data, val_data = make_dataset(X_train, y_train), make_dataset(X_val, y_val)
    torch.manual_seed(random_state)
    device = select_device(device)
    model = AdvancedNeuralNetwork(X_train.shape[1], config).to(device)
    model.feature_columns = list(X_train.columns)
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    synchronize(device)
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        epoch_lr = optimizer.param_groups[0]["lr"]
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            # A singleton has no batch variance: use running statistics for BN
            # only, retaining dropout and gradients and never discarding a row.
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.train(len(labels) > 1)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        val_loss, _ = predict(model, val_data)
        train_loss = loss_sum / len(train_data)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Non-finite loss; check training settings.")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": epoch_lr,
            }
        )
        if verbose:
            print(f"Epoch {epoch:03d}: train={train_loss:.4f}, val={val_loss:.4f}")
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    synchronize(device)
    return (
        model,
        pd.DataFrame(history),
        {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "training_seconds": perf_counter() - started,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "device": str(device),
        },
    )


def sample_configs(trials=30, seed=RANDOM_STATE):
    """Include the proposed starting model, then seeded random candidates."""
    if trials < 1:
        raise ValueError("trials must be positive.")
    rng = np.random.default_rng(seed)
    widths = [(128, 64), (256, 128, 64), (256, 256, 128, 64)]
    configs = [ModelConfig()]
    for _ in range(trials - 1):
        configs.append(
            ModelConfig(
                hidden_dims=widths[int(rng.integers(len(widths)))],
                dropout=float(rng.choice([0, 0.1, 0.2, 0.3])),
                batch_norm=bool(rng.integers(2)),
                learning_rate=float(10 ** rng.uniform(-4, np.log10(3e-3))),
                weight_decay=float(10 ** rng.uniform(-6, -2)),
                batch_size=int(rng.choice([512, 1024, 2048])),
            )
        )
    return configs


def rank_trials(trials):
    return trials.sort_values(
        ["val_roc_auc", "val_loss", "trial"], ascending=[False, True, True]
    ).reset_index(drop=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_run(directory, model, history, summary, config, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_training_plots(history, directory)
    write_json(directory / "summary.json", {**summary, "seed": seed, "config": config})
    torch.save(
        {
            "model_type": type(model).__name__,
            "config": config,
            "feature_columns": model.feature_columns,
            "seed": seed,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        directory / "model.pt",
    )


def tune_model(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    trials=30,
    seed=RANDOM_STATE,
    output_dir=None,
    **training_options,
):
    """Search API deliberately accepts no test split; return config and table."""
    if y_val.nunique() != 2:
        raise ValueError("ROC AUC tuning requires both classes in validation.")
    records = []
    configs = sample_configs(trials, seed)
    for index, config in enumerate(configs):
        print(f"Trial {index + 1}/{trials}: {config}")
        model, history, summary = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            random_state=seed,
            **training_options,
        )
        metrics = evaluate_model(model, X_val, y_val)
        records.append(
            {
                "trial": index,
                **asdict(config),
                **summary,
                **{f"val_{key}": value for key, value in metrics.items()},
            }
        )
        if output_dir is not None:
            directory = Path(output_dir)
            save_run(
                directory / f"trial_{index:03d}",
                model,
                history,
                summary,
                asdict(config),
                seed,
            )
            rank_trials(pd.DataFrame(records)).to_csv(
                directory / "trials.csv", index=False
            )
    ranked = rank_trials(pd.DataFrame(records))
    selected = configs[int(ranked.iloc[0].trial)]
    if output_dir is not None:
        write_json(Path(output_dir) / "selected_config.json", asdict(selected))
    return selected, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["train", "tune", "compare"], default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "result/advanced_neural_network",
    )
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--batch-norm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    # Architecture flags apply only to single-run training; search is predefined.
    config = ModelConfig(
        tuple(args.hidden_dims),
        args.dropout,
        args.batch_norm,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
    )
    if min(args.epochs, args.patience, args.trials) < 1:
        parser.error("epochs, patience and trials must be positive")
    X_train, X_val, X_test, y_train, y_val, y_test, state = run_pipeline()
    directory = args.output_dir / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, directory / "preprocessing.joblib")
    write_json(
        directory / "run_settings.json", {**vars(args), "output_dir": str(directory)}
    )
    options = {"epochs": args.epochs, "patience": args.patience, "device": args.device}
    if args.mode in {"tune", "compare"}:
        config, _ = tune_model(
            X_train,
            y_train,
            X_val,
            y_val,
            trials=args.trials,
            seed=args.seed,
            output_dir=directory / "search",
            **options,
        )
    write_json(directory / "selected_config.json", asdict(config))
    variants = [("tuned_deep" if args.mode != "train" else "deep", config)]
    seeds = [args.seed]
    if args.mode == "compare":
        variants = [
            ("shallow", None),
            ("starting_deep", ModelConfig()),
            ("tuned_deep", config),
        ]
        seeds = [args.seed, args.seed + 1, args.seed + 2]
    results = []
    # Configuration selection is complete before any test-set evaluation.
    for name, variant in variants:
        for seed in seeds:
            trainer = train_baseline if variant is None else train_model
            settings = {} if variant is None else {"config": variant}
            model, history, summary = trainer(
                X_train, y_train, X_val, y_val, random_state=seed, **options, **settings
            )
            run_dir = directory / name / f"seed_{seed}"
            saved_config = (
                asdict(variant)
                if variant is not None
                else {
                    "hidden_dim": 128,
                    "batch_size": 1024,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "optimizer": "Adam",
                }
            )
            save_run(run_dir, model, history, summary, saved_config, seed)
            metrics = {
                split: evaluate_model(model, X, y)
                for split, X, y in [
                    ("validation", X_val, y_val),
                    ("test", X_test, y_test),
                ]
            }
            write_json(run_dir / "metrics.json", metrics)
            for split, values in metrics.items():
                results.append(
                    dict(model=name, seed=seed, split=split, **summary, **values)
                )
    frame = pd.DataFrame(results)
    frame.to_csv(directory / "metrics.csv", index=False)
    columns = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "training_seconds",
        "inference_seconds",
        "parameter_count",
    ]
    # Sample standard deviation is reported only for repeated-seed comparisons.
    aggregate = frame.groupby(["model", "split"])[columns].agg(
        ["mean", "std"] if len(seeds) > 1 else ["mean"]
    )
    aggregate.to_csv(directory / "metrics_summary.csv")
    print(aggregate.to_string())
    print(f"Results saved to {directory}")


if __name__ == "__main__":
    main()
