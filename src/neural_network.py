"""Task 5g: a baseline neural network for predicting DRK_YN (1 = drinker).

Run from the project root: python src/neural_network.py --epochs 50
Uses the existing preprocessing and engineered features without fitting another
scaler. A single hidden layer learns nonlinear feature interactions while keeping
this baseline small; the deeper network and hyperparameter search belong to task 9.
"""

import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import RANDOM_STATE, run_pipeline


def select_device(name: str = "auto") -> torch.device:
    """Prefer CUDA, then Apple Silicon MPS, falling back to CPU in auto mode."""
    available = {"cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available(), "cpu": True}
    if name == "auto":
        name = next(device for device, supported in available.items() if supported)
    if name not in available or not available[name]:
        raise ValueError(f"Requested device '{name}' is unavailable on this machine.")
    return torch.device(name)


def synchronize(device: torch.device):
    """Wait for asynchronous GPU work before recording wall-clock timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class NeuralNetwork(nn.Module):
    """One ReLU hidden layer and one binary-classification output logit.

    128 hidden units increase capacity while retaining a single hidden layer.
    Returning logits lets BCEWithLogitsLoss compute binary cross-entropy stably;
    sigmoid is applied only when probabilities are needed for evaluation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def make_dataset(X: pd.DataFrame, y: pd.Series) -> TensorDataset:
    """Convert processed data to float32, checking row alignment and labels."""
    if len(X) == 0 or not X.index.equals(y.index):
        raise ValueError("X and y must be nonempty and have matching row indices.")
    features = X.to_numpy(dtype=np.float32, copy=True)
    labels = y.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(features).all():
        raise ValueError("Predictors must be finite; run preprocessing first.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be encoded as 0 or 1.")
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


@torch.inference_mode()
def predict(model: NeuralNetwork, dataset: TensorDataset, batch_size: int = 4096):
    """Return mean cross-entropy and probabilities without tracking gradients."""
    model.eval()
    loss_sum = 0.0
    probabilities = []
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for features, labels in DataLoader(dataset, batch_size=batch_size):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss_sum += criterion(logits, labels).item()
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return loss_sum / len(dataset), np.concatenate(probabilities)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    hidden_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "auto",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    """Train on the selected device and restore the best validation epoch.

    Adam is a practical initial optimizer; mini-batches limit memory use on this
    large dataset. Small L2 regularization and early stopping limit overfitting.
    Defaults are starting choices, not tuned results. The test set is never used
    here. Returns (model, epoch history, training summary).
    Halve the learning rate after a validation-loss plateau; default early
    stopping patience leaves time for the reduced learning rate to take effect.
    """
    if min(hidden_dim, epochs, batch_size, patience) < 1:
        raise ValueError("hidden_dim, epochs, batch_size and patience must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    if not X_train.columns.equals(X_val.columns):
        raise ValueError("Training and validation feature columns must match in order.")
    train_data = make_dataset(X_train, y_train)
    val_data = make_dataset(X_val, y_val)
    torch.manual_seed(random_state)
    device = select_device(device)
    model = NeuralNetwork(X_train.shape[1], hidden_dim).to(device)
    if verbose:
        print(f"Device: {device}; hidden units: {hidden_dim}")
    # Keep the schema so evaluation cannot silently swap feature positions.
    model.feature_columns = list(X_train.columns)
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    synchronize(device)
    started = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        loss_sum = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        val_loss, _ = predict(model, val_data)
        history.append({
            "epoch": epoch, "train_loss": loss_sum / len(train_data),
            "val_loss": val_loss, "learning_rate": epoch_lr,
        })
        if verbose:
            print(f"Epoch {epoch:02d}: train loss={history[-1]['train_loss']:.4f}, "
                  f"validation loss={val_loss:.4f}, learning rate={epoch_lr:.2g}")
        if not np.isfinite(val_loss):
            raise RuntimeError("Validation loss became non-finite; check training settings.")
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    synchronize(device)
    return model, pd.DataFrame(history), {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": perf_counter() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }


def save_training_plots(history: pd.DataFrame, output_dir: Path) -> Path:
    """Save loss/LR curves and the underlying epoch history for later analysis.

    Training loss is averaged over batches during updates; validation loss is
    measured at the end of each epoch with fixed weights.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(fig)
    loss_ax, lr_ax = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    loss_ax.plot(history.epoch, history.train_loss, label="Training (batch average)")
    loss_ax.plot(history.epoch, history.val_loss, label="Validation")
    best = history.loc[history.val_loss.idxmin()]
    loss_ax.axvline(best.epoch, linestyle="--", color="gray",
                    label=f"Best validation epoch: {int(best.epoch)}")
    loss_ax.set(title="Neural network training", ylabel="Binary cross-entropy")
    loss_ax.legend()
    loss_ax.grid(alpha=0.25)
    lr_ax.step(history.epoch, history.learning_rate, where="post")
    lr_ax.set(xlabel="Epoch", ylabel="Learning rate", yscale="log")
    lr_ax.grid(alpha=0.25)
    path = output_dir / "neural_network_training.png"
    fig.savefig(path, dpi=160)
    history.to_csv(output_dir / "neural_network_history.csv", index=False)
    return path


def evaluate_model(model: NeuralNetwork, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate with a fixed 0.5 threshold; confusion matrix order is [0, 1].

    Accuracy measures overall correctness, precision/recall/F1 describe the
    drinker class, and ROC AUC measures ranking across thresholds. These support
    later comparisons with the classical models and ensembles.
    """
    if list(X.columns) != model.feature_columns:
        raise ValueError("Evaluation feature columns must match training order.")
    dataset = make_dataset(X, y)
    synchronize(next(model.parameters()).device)
    started = perf_counter()
    loss, probabilities = predict(model, dataset)
    inference_seconds = perf_counter() - started
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "loss": loss,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities) if y.nunique() == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "inference_seconds": inference_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--figures-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_pipeline()
    print(f"Training with {len(X_train):,} rows and {X_train.shape[1]} features.")
    model, history, summary = train_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, device=args.device,
    )
    print(f"\nTraining plots saved to: {save_training_plots(history, args.figures_dir)}")
    print("\nTraining summary:", summary)
    # Evaluate the held-out test set only after selecting the epoch on validation.
    for split, X, y in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        print(f"\n{split} metrics:")
        for metric, value in evaluate_model(model, X, y).items():
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
