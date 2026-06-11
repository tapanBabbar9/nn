"""Configuration for global SKU demand forecasting with Random Forest."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Six years of daily SKU-level sales (synthetic or real CSV)."""
    data_path: str = "../data/sku_daily_sales.csv"
    date_col: str = "date"
    sku_col: str = "sku"
    target_col: str = "units_sold"
    # Time-based split — train on history, test on the final year
    test_start_date: str = "2025-01-01"


@dataclass
class SyntheticDataConfig:
    """Defaults for generate_data.py — ~6 years × 50 SKUs."""
    n_skus: int = 50
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42


@dataclass
class FeatureConfig:
    """
    Time-series features for tabular forecasting.

    forecast_horizon: predict demand t+N days ahead (1 = next day).
    """
    lag_days: list[int] = field(default_factory=lambda: [1, 7, 14, 30])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 30])
    forecast_horizon: int = 1
    include_calendar: bool = True
    include_trend: bool = True
    encode_sku: bool = True


@dataclass
class RFConfig:
    """Random Forest hyperparameters."""
    n_estimators: int = 500
    max_depth: int = 20
    min_samples_leaf: int = 3
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class TrainConfig:
    output_dir: str = "outputs/rf-demand"
    seed: int = 42


@dataclass
class EvalConfig:
    results_path: str = "outputs/rf-demand/eval_results.json"


@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "sku-demand-forecast"
    registered_model_name: str = "SKUDemandRF"
    run_id_path: str = "outputs/last_mlflow_run.txt"
    # Promotion gates — lower MAPE is better
    staging_max_mape: float = 0.25
    production_max_mape: float = 0.15
