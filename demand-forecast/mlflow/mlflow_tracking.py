"""MLflow helpers for SKU demand forecasting."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor

from config import MLflowConfig


def setup_mlflow(cfg: MLflowConfig | None = None) -> MLflowConfig:
    cfg = cfg or MLflowConfig()
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    return cfg


def git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_training_params(
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    forecast_horizon: int,
    lag_days: list[int],
    rolling_windows: list[int],
    train_rows: int,
    test_rows: int,
    n_skus: int,
    test_start_date: str,
    dataset_path: str,
    seed: int,
) -> None:
    mlflow.log_params(
        {
            "model_type": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "forecast_horizon": forecast_horizon,
            "lag_days": ",".join(map(str, lag_days)),
            "rolling_windows": ",".join(map(str, rolling_windows)),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "n_skus": n_skus,
            "test_start_date": test_start_date,
            "dataset_path": dataset_path,
            "seed": seed,
            "architecture": "global_model_all_skus",
        }
    )
    sha = git_sha()
    if sha:
        mlflow.log_param("git_sha", sha)


def log_forecast_metrics(metrics: dict[str, float], *, prefix: str = "") -> None:
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else key
        mlflow.log_metric(name, value)


def log_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    output_dir: str,
) -> None:
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "feature_importance.csv"
    importance.to_csv(path, index=False)
    mlflow.log_artifact(str(path), artifact_path="analysis")

    top = importance.head(10)
    print("\nTop feature importances:")
    for _, row in top.iterrows():
        print(f"  {row['feature']:<24} {row['importance']:.4f}")


def log_dataset_artifact(dataset_path: str) -> None:
    path = Path(dataset_path)
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path="dataset")


def log_sklearn_model(
    model: RandomForestRegressor,
    registered_model_name: str | None = None,
) -> None:
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=registered_model_name,
    )


def save_run_id(run_id: str, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run_id, encoding="utf-8")


def load_run_id(path: str) -> str | None:
    p = Path(path)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return None


def register_model_from_run(run_id: str, registered_model_name: str):
    model_uri = f"runs:/{run_id}/model"
    return mlflow.register_model(model_uri, registered_model_name)


def promote_model_version(
    registered_model_name: str,
    version: int,
    stage: str,
) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )


def get_run_metrics(run_id: str) -> dict[str, float]:
    client = MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics)
