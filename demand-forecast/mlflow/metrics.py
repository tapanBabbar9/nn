"""Forecast evaluation metrics — MAE, RMSE, MAPE."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1.0) -> float:
    """Mean absolute percentage error; eps avoids division by zero on zero-demand days."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
    }


def print_metrics(metrics: dict[str, float], *, label: str = "Test") -> None:
    print(f"\n{label} metrics:")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MAPE : {metrics['mape']:.2f}%")
