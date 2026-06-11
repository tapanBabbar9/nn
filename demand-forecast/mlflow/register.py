#!/usr/bin/env python3
"""
Register the trained RF model in MLflow Model Registry and promote stages.

Usage:
  python register.py
  python register.py --promote staging
  python register.py --promote production
"""

from __future__ import annotations

import argparse

from config import MLflowConfig
from mlflow_tracking import (
    get_run_metrics,
    load_run_id,
    promote_model_version,
    register_model_from_run,
    setup_mlflow,
)


def parse_args():
    p = argparse.ArgumentParser(description="Register and promote SKU demand RF model")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument(
        "--promote",
        choices=["staging", "production"],
        default=None,
    )
    p.add_argument("--force", action="store_true", help="Skip MAPE gates when promoting")
    return p.parse_args()


def main():
    args = parse_args()
    mlflow_cfg = setup_mlflow()

    run_id = args.run_id or load_run_id(mlflow_cfg.run_id_path)
    if not run_id:
        print("No MLflow run id found. Run train.py and evaluate.py first.")
        return

    print("=" * 60)
    print("MLflow Model Registry — SKU Demand RF")
    print("=" * 60)
    print(f"  Run id         : {run_id}")
    print(f"  Registered name: {mlflow_cfg.registered_model_name}")
    print()

    model_version = register_model_from_run(run_id, mlflow_cfg.registered_model_name)
    version = model_version.version
    print(f"Registered {mlflow_cfg.registered_model_name} version {version}")

    if not args.promote:
        print("\nRegistered. To promote:")
        print("  python register.py --promote staging")
        print("  python register.py --promote production")
        return

    metrics = get_run_metrics(run_id)
    mape = metrics.get("mape") or metrics.get("test_mape")

    if args.promote == "staging":
        if not args.force and mape is not None:
            if mape > mlflow_cfg.staging_max_mape * 100:
                print(
                    f"\nStaging gate failed: MAPE={mape:.2f}% "
                    f"> {mlflow_cfg.staging_max_mape * 100:.0f}%"
                )
                print("Run evaluate.py first, or use --force to override.")
                return
        promote_model_version(mlflow_cfg.registered_model_name, version, "Staging")
        print(f"\nVersion {version} promoted to Staging.")

    if args.promote == "production":
        if not args.force:
            if mape is None:
                print("\nProduction gate failed: eval MAPE missing on run.")
                print("Run evaluate.py first.")
                return
            if mape > mlflow_cfg.production_max_mape * 100:
                print(
                    f"\nProduction gate failed: MAPE={mape:.2f}% "
                    f"> {mlflow_cfg.production_max_mape * 100:.0f}%"
                )
                return
        promote_model_version(mlflow_cfg.registered_model_name, version, "Production")
        print(f"\nVersion {version} promoted to Production.")

    print("\nOpen the registry in MLflow UI:")
    print("  python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
