#!/usr/bin/env python3
"""
Register a trained adapter in the MLflow Model Registry and promote stages.

Usage:
  python register.py
  python register.py --promote staging
  python register.py --promote production
  python register.py --run-id <mlflow_run_id> --promote staging
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
    p = argparse.ArgumentParser(description="Register and promote LoRA adapter in MLflow")
    p.add_argument("--run-id", type=str, default=None, help="MLflow training/eval run id")
    p.add_argument(
        "--promote",
        choices=["staging", "production"],
        default=None,
        help="Promote registered version to Staging or Production if gates pass",
    )
    p.add_argument("--force", action="store_true", help="Skip metric gates when promoting")
    return p.parse_args()


def main():
    args = parse_args()
    mlflow_cfg = setup_mlflow()

    run_id = args.run_id or load_run_id(mlflow_cfg.run_id_path)
    if not run_id:
        print("No MLflow run id found. Run train.py and evaluate.py first.")
        return

    print("=" * 60)
    print("MLflow Model Registry")
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
    sentiment = metrics.get("lora_sentiment_accuracy")
    format_strict = metrics.get("lora_format_compliance_strict")

    if args.promote == "staging":
        if not args.force and sentiment is not None:
            if sentiment < mlflow_cfg.staging_min_sentiment_accuracy:
                print(
                    f"\nStaging gate failed: lora_sentiment_accuracy={sentiment:.4f} "
                    f"< {mlflow_cfg.staging_min_sentiment_accuracy}"
                )
                print("Run evaluate.py first, or use --force to override.")
                return
        promote_model_version(mlflow_cfg.registered_model_name, version, "Staging")
        print(f"\nVersion {version} promoted to Staging.")

    if args.promote == "production":
        if not args.force:
            if sentiment is None or format_strict is None:
                print("\nProduction gate failed: eval metrics missing on run.")
                print("Run evaluate.py first.")
                return
            if sentiment < mlflow_cfg.staging_min_sentiment_accuracy:
                print(
                    f"\nProduction gate failed: lora_sentiment_accuracy={sentiment:.4f} "
                    f"< {mlflow_cfg.staging_min_sentiment_accuracy}"
                )
                return
            if format_strict < mlflow_cfg.production_min_format_strict:
                print(
                    f"\nProduction gate failed: lora_format_compliance_strict={format_strict:.4f} "
                    f"< {mlflow_cfg.production_min_format_strict}"
                )
                return
        promote_model_version(mlflow_cfg.registered_model_name, version, "Production")
        print(f"\nVersion {version} promoted to Production.")

    print("\nOpen the registry in MLflow UI:")
    print("  mlflow ui")


if __name__ == "__main__":
    main()
