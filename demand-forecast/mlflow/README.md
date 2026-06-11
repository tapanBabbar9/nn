# MLflow track

Experiment tracking and model registry for the SKU demand forecasting pipeline.

For the full methodology (data cleaning, features, evaluation), see [`../README.md`](../README.md).

## Setup

```bash
cd demand-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r mlflow/requirements.txt
cd mlflow
```

## Commands

```bash
python generate_data.py                    # create synthetic CSV
python train.py                            # train + log run
python train.py --generate                 # auto-generate data if missing
python train.py --horizon 30               # 30-day forecast horizon
python train.py --test-start 2024-01-01    # custom split date
python evaluate.py                         # log test metrics to same run
python register.py                         # register model
python register.py --promote staging       # promote if MAPE gate passes
python register.py --promote production
python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Always launch the UI with the **same Python** you used for training (`python3 -m mlflow`, not a global `mlflow` binary from another install).

## What gets logged

| Item | Location |
|------|----------|
| Hyperparameters | `n_estimators`, `forecast_horizon`, `lag_days`, `n_skus`, `git_sha`, … |
| Metrics | `train_mae`, `train_mape`, `test_mae`, `test_mape`, … |
| Artifacts | dataset CSV, `feature_importance.csv`, `training_meta.json` |
| Model | sklearn `RandomForestRegressor` at `model/` |

Last run ID: `outputs/last_mlflow_run.txt` (used by `evaluate.py` and `register.py`).

## Model registry

Registered model name: `SKUDemandRF`

| Stage | MAPE gate (test set) |
|-------|----------------------|
| Staging | ≤ 25% |
| Production | ≤ 15% |

Gates are defined in `config.py` (`MLflowConfig`). Override for testing:

```bash
python register.py --promote staging --force
```

## Troubleshooting

**`Can't locate revision identified by 'da6fb0208061'`** — the SQLite DB was created by a newer MLflow than the one launching the UI. Common when macOS has multiple Pythons (e.g. 3.9 system `mlflow` vs 3.13 where you trained).

```bash
# Use the interpreter that has your project deps
cd demand-forecast/mlflow
python3 -m mlflow --version          # should match the env where you ran train.py
python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Or use a venv so train and UI share one MLflow version:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To reset (loses experiments): `rm mlflow.db` and re-run `train.py`.

## Config reference

| Class | Purpose |
|-------|---------|
| `DataConfig` | CSV path, column names, test split date |
| `FeatureConfig` | Lag days, rolling windows, forecast horizon |
| `RFConfig` | `n_estimators`, `max_depth`, `min_samples_leaf` |
| `MLflowConfig` | tracking URI, experiment name, registry gates |
