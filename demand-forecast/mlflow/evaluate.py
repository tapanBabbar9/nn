#!/usr/bin/env python3
"""
Evaluate the trained RF model on the held-out test period and log to MLflow.

Usage:
  python evaluate.py
  python evaluate.py --run-id <mlflow_run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from config import DataConfig, EvalConfig, FeatureConfig, MLflowConfig, TrainConfig
from features import build_features, load_sales_csv, prepare_xy, train_test_split_by_date
from metrics import compute_metrics, print_metrics
from mlflow_tracking import load_run_id, log_forecast_metrics, setup_mlflow


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SKU demand RF model")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    eval_cfg = EvalConfig()
    train_cfg = TrainConfig()
    mlflow_cfg = setup_mlflow()

    if args.dataset:
        data_cfg.data_path = args.dataset
    if args.output:
        eval_cfg.results_path = args.output

    run_id = args.run_id or load_run_id(mlflow_cfg.run_id_path)
    if not run_id:
        print("No MLflow run id found. Run train.py first.")
        return

    meta_path = Path(train_cfg.output_dir) / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feat_cfg.forecast_horizon = meta.get("forecast_horizon", feat_cfg.forecast_horizon)
        data_cfg.test_start_date = meta.get("test_start_date", data_cfg.test_start_date)

    dataset_path = Path(data_cfg.data_path)
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("SKU Demand Forecast — Evaluation (MLflow)")
    print("=" * 60)
    print(f"  Dataset    : {data_cfg.data_path}")
    print(f"  Test from  : {data_cfg.test_start_date}")
    print(f"  MLflow run : {run_id}")
    print()

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    raw = load_sales_csv(data_cfg.data_path, data_cfg)
    featured, artifacts = build_features(raw, data_cfg, feat_cfg)
    _, test_df = train_test_split_by_date(featured, data_cfg, artifacts.feature_columns)
    x_test, y_test, test_clean = prepare_xy(test_df, artifacts.feature_columns)

    print(f"  Test rows  : {len(test_clean):,}")
    print("Running predictions...")
    y_pred = model.predict(x_test)
    test_metrics = compute_metrics(y_test, y_pred)
    print_metrics(test_metrics, label="Test")

    residuals = y_test - y_pred
    by_sku = (
        test_clean.assign(actual=y_test, predicted=y_pred, residual=residuals)
        .groupby(data_cfg.sku_col)
        .apply(
            lambda g: pd.Series(
                compute_metrics(g["actual"].to_numpy(), g["predicted"].to_numpy())
            ),
            include_groups=False,
        )
        .reset_index()
    )

    results = {
        "mlflow_run_id": run_id,
        "dataset_path": data_cfg.data_path,
        "test_start_date": data_cfg.test_start_date,
        "forecast_horizon": feat_cfg.forecast_horizon,
        "n_test_rows": len(test_clean),
        "metrics": test_metrics,
        "per_sku_metrics": by_sku.to_dict(orient="records"),
        "sample_predictions": test_clean.assign(
            actual=y_test,
            predicted=np.round(y_pred, 2),
            residual=np.round(residuals, 2),
        )
        .head(20)
        .to_dict(orient="records"),
    }

    out_path = Path(eval_cfg.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")

    with mlflow.start_run(run_id=run_id):
        log_forecast_metrics(test_metrics)
        mlflow.log_artifact(str(out_path), artifact_path="eval")
        mlflow.set_tag("stage", "evaluation")

    print(f"\nEval metrics logged to MLflow run {run_id}")
    print("Next: python register.py --promote staging")


if __name__ == "__main__":
    main()
