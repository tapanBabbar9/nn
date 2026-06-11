#!/usr/bin/env python3
"""
Train a global Random Forest demand model with MLflow tracking.

One model across all SKUs — SKU is encoded as a feature. Time-based train/test
split (no shuffle). Engineered lag, rolling, calendar, and trend features.

Usage:
  python generate_data.py   # first time
  python train.py
  python train.py --horizon 7
  python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import DataConfig, FeatureConfig, MLflowConfig, RFConfig, TrainConfig
from features import (
    build_features,
    load_sales_csv,
    prepare_xy,
    train_test_split_by_date,
)
from metrics import compute_metrics, print_metrics
from mlflow_tracking import (
    log_dataset_artifact,
    log_feature_importance,
    log_forecast_metrics,
    log_sklearn_model,
    log_training_params,
    save_run_id,
    setup_mlflow,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train global RF SKU demand model with MLflow")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--horizon", type=int, default=None, help="Forecast horizon in days (t+N)")
    p.add_argument("--test-start", type=str, default=None, help="Test period start date")
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--generate", action="store_true", help="Generate synthetic data if missing")
    return p.parse_args()


def ensure_dataset(path: str, generate: bool) -> None:
    if Path(path).is_file():
        return
    if not generate:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python generate_data.py"
        )
    from generate_data import generate_synthetic_sales
    from config import SyntheticDataConfig

    cfg = SyntheticDataConfig()
    df = generate_synthetic_sales(cfg)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[info] Generated synthetic dataset at {path}")


def main():
    args = parse_args()
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    rf_cfg = RFConfig()
    train_cfg = TrainConfig()
    mlflow_cfg = setup_mlflow()

    if args.dataset:
        data_cfg.data_path = args.dataset
    if args.horizon:
        feat_cfg.forecast_horizon = args.horizon
    if args.test_start:
        data_cfg.test_start_date = args.test_start
    if args.n_estimators:
        rf_cfg.n_estimators = args.n_estimators

    ensure_dataset(data_cfg.data_path, args.generate)

    print("=" * 60)
    print("SKU Demand Forecast — Random Forest (MLflow)")
    print("=" * 60)
    print(f"  Dataset    : {data_cfg.data_path}")
    print(f"  Horizon    : t+{feat_cfg.forecast_horizon}")
    print(f"  Test from  : {data_cfg.test_start_date}")
    print(f"  RF trees   : {rf_cfg.n_estimators}")
    print(f"  Experiment : {mlflow_cfg.experiment_name}")
    print()

    raw = load_sales_csv(data_cfg.data_path, data_cfg)
    featured, artifacts = build_features(raw, data_cfg, feat_cfg)
    train_df, test_df = train_test_split_by_date(featured, data_cfg, artifacts.feature_columns)

    x_train, y_train, train_clean = prepare_xy(train_df, artifacts.feature_columns)
    x_test, y_test, test_clean = prepare_xy(test_df, artifacts.feature_columns)

    print(f"  SKUs       : {raw[data_cfg.sku_col].nunique()}")
    print(f"  Train rows : {len(train_clean):,}")
    print(f"  Test rows  : {len(test_clean):,}")
    print(f"  Features   : {len(artifacts.feature_columns)}")
    print()

    model = RandomForestRegressor(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        random_state=rf_cfg.random_state,
        n_jobs=rf_cfg.n_jobs,
    )

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=args.run_name) as run:
        log_training_params(
            n_estimators=rf_cfg.n_estimators,
            max_depth=rf_cfg.max_depth,
            min_samples_leaf=rf_cfg.min_samples_leaf,
            forecast_horizon=feat_cfg.forecast_horizon,
            lag_days=feat_cfg.lag_days,
            rolling_windows=feat_cfg.rolling_windows,
            train_rows=len(train_clean),
            test_rows=len(test_clean),
            n_skus=int(raw[data_cfg.sku_col].nunique()),
            test_start_date=data_cfg.test_start_date,
            dataset_path=data_cfg.data_path,
            seed=train_cfg.seed,
        )
        log_dataset_artifact(data_cfg.data_path)
        mlflow.set_tag("pipeline", "sku-demand-rf")
        mlflow.set_tag("stage", "training")

        print("Training Random Forest...")
        model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)

        log_forecast_metrics(train_metrics, prefix="train_")
        log_forecast_metrics(test_metrics, prefix="test_")
        log_feature_importance(model, artifacts.feature_columns, str(out_dir))
        log_sklearn_model(model)

        meta = {
            "feature_columns": artifacts.feature_columns,
            "sku_classes": (
                list(artifacts.sku_encoder.classes_)
                if artifacts.sku_encoder is not None
                else None
            ),
            "min_date": str(artifacts.min_date.date()),
            "forecast_horizon": feat_cfg.forecast_horizon,
            "test_start_date": data_cfg.test_start_date,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        meta_path = out_dir / "training_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(str(meta_path), artifact_path="metadata")

        save_run_id(run.info.run_id, mlflow_cfg.run_id_path)

        print_metrics(train_metrics, label="Train")
        print_metrics(test_metrics, label="Test")
        print(f"\nMLflow run id : {run.info.run_id}")
        print(f"Metadata      : {meta_path}")
        print("Next:")
        print("  python evaluate.py")
        print("  python register.py --promote staging")
        print("  python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
