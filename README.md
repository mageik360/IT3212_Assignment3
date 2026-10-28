# IT3212 Assignment 3

Binary classification of `DRK_YN` (drinker) on the smoking and drinking dataset.

## Setup

```
pip install -r requirements.txt   # Python 3.12+
cp .env.example .env              # set PATH_DATASET to the CSV
pytest
```

Run scripts from the project root, e.g. `python src/decision_tree.py`.

## Layout (`src/`)

| Module | Purpose |
|---|---|
| `config.py` | `PROJECT_ROOT`, `FIGURES_DIR`, `RESULTS_DIR`, `RANDOM_STATE` |
| `data.py` | Loads the raw CSV from `PATH_DATASET` |
| `data_pipeline.py` | `run_pipeline()` → `X_train, X_val, X_test, y_train, y_val, y_test, state` (stratified 80/10/10) |
| `preprocessing.py`, `feature_engineering.py` | Cleaning, encoding, scaling, derived features (fitted on train only) |
| `evaluation.py` | `compute_metrics(y, proba)`, `plot_roc_curve(y, {label: proba}, path)` |
| `tuning.py` | `tune_on_validation(estimator, grid, ...)`, `results_table(search)` for sklearn models |
| `plotting.py` | `save_figure(fig, path)` |
| `neural_network.py` | Baseline MLP, shared torch helpers and the shared `fit()` training loop |
| `advanced_neural_network.py` | Deep MLP with random search (`--mode train/tune/compare`) |
| `decision_tree.py` | Decision tree tuned on the validation split |
| `eda.py` | Exploratory analysis, see `EDA.md` |

`classical_models.py`, `ensembles.py`, `feature_selection.py` and `transfer_learning.py` are empty placeholders.

## Conventions for new models

- Get data from `run_pipeline()`; don't preprocess again.
- Select hyperparameters on the validation split only. Touch the test set once, for the final model.
- sklearn models: tune with `tuning.tune_on_validation`.
- Torch models: build the model and optimizer, then call `neural_network.fit`.
- Score every model with `evaluation.compute_metrics` so results are comparable.
- Take seeds and paths from `config`, and save figures with `plotting.save_figure` under `FIGURES_DIR / "<model>"`.
- Put tests in `tests/`, using small synthetic data (see `tests/test_shared.py` for dry-running a `main()`).
