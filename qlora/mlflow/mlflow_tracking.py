"""MLflow helpers for experiment tracking and model registry."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

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
    model_id: str,
    mode: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    num_epochs: int,
    seed: int,
    dataset_path: str,
    num_examples: int,
    trainable_m: float,
) -> None:
    mlflow.log_params(
        {
            "base_model": model_id,
            "mode": mode,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "seed": seed,
            "dataset_path": dataset_path,
            "num_examples": num_examples,
            "trainable_params_m": round(trainable_m, 4),
        }
    )
    sha = git_sha()
    if sha:
        mlflow.log_param("git_sha", sha)


def log_dataset_artifact(dataset_path: str) -> None:
    path = Path(dataset_path)
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path="dataset")


def log_train_metrics_from_history(log_history: list[dict]) -> None:
    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry:
            mlflow.log_metric("train_loss", entry["loss"], step=step)
        if "learning_rate" in entry:
            mlflow.log_metric("learning_rate", entry["learning_rate"], step=step)
        if "epoch" in entry:
            mlflow.log_metric("epoch", entry["epoch"], step=step)


def log_adapter_artifacts(output_dir: str) -> None:
    mlflow.log_artifacts(output_dir, artifact_path="adapter")


def log_eval_metrics(lora_summary: dict, base_summary: dict) -> None:
    for key, value in lora_summary.items():
        if isinstance(value, (int, float)) and key != "examples" and key != "rouge_examples":
            mlflow.log_metric(f"lora_{key}", value)
    for key, value in base_summary.items():
        if isinstance(value, (int, float)) and key != "examples" and key != "rouge_examples":
            mlflow.log_metric(f"base_{key}", value)

    mlflow.log_metric(
        "sentiment_accuracy_delta",
        lora_summary["sentiment_accuracy"] - base_summary["sentiment_accuracy"],
    )
    mlflow.log_metric(
        "format_strict_delta",
        lora_summary["format_compliance_strict"] - base_summary["format_compliance_strict"],
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


def register_model_from_run(run_id: str, registered_model_name: str) -> mlflow.entities.model_registry.ModelVersion:
    model_uri = f"runs:/{run_id}/adapter"
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
