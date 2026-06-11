"""Feature engineering for global SKU demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import DataConfig, FeatureConfig


@dataclass
class FeatureArtifacts:
    """Fitted encoders and column metadata for inference."""
    feature_columns: list[str]
    sku_encoder: LabelEncoder | None
    min_date: pd.Timestamp


def load_sales_csv(path: str, cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[cfg.date_col])
    required = {cfg.date_col, cfg.sku_col, cfg.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    df = df.sort_values([cfg.sku_col, cfg.date_col]).reset_index(drop=True)
    return df


def build_features(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    *,
    sku_encoder: LabelEncoder | None = None,
    min_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, FeatureArtifacts]:
    """
    Build lag, rolling, calendar, trend, and SKU features per SKU group.

    Target is units_sold shifted backward by forecast_horizon (predict t+N).
    """
    out = df.copy()
    out[data_cfg.date_col] = pd.to_datetime(out[data_cfg.date_col])
    out = out.sort_values([data_cfg.sku_col, data_cfg.date_col]).reset_index(drop=True)

    if min_date is None:
        min_date = out[data_cfg.date_col].min()

    grouped = out.groupby(data_cfg.sku_col, group_keys=False)

    for lag in feat_cfg.lag_days:
        out[f"sales_lag_{lag}"] = grouped[data_cfg.target_col].shift(lag)

    for window in feat_cfg.rolling_windows:
        shifted = grouped[data_cfg.target_col].shift(1)
        out[f"rolling_mean_{window}"] = (
            shifted.groupby(out[data_cfg.sku_col]).transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )
        )
        out[f"rolling_std_{window}"] = (
            shifted.groupby(out[data_cfg.sku_col]).transform(
                lambda s: s.rolling(window, min_periods=window).std()
            )
        )

    if feat_cfg.include_calendar:
        dates = out[data_cfg.date_col]
        out["day_of_week"] = dates.dt.dayofweek
        out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        out["month"] = dates.dt.month
        out["quarter"] = dates.dt.quarter
        out["weekend_flag"] = (dates.dt.dayofweek >= 5).astype(int)

    if feat_cfg.include_trend:
        out["days_since_start"] = (out[data_cfg.date_col] - min_date).dt.days

    if feat_cfg.encode_sku:
        if sku_encoder is None:
            sku_encoder = LabelEncoder()
            out["sku_id"] = sku_encoder.fit_transform(out[data_cfg.sku_col])
        else:
            known = set(sku_encoder.classes_)
            out["sku_id"] = out[data_cfg.sku_col].map(
                lambda s: sku_encoder.transform([s])[0] if s in known else -1
            )

    horizon = feat_cfg.forecast_horizon
    out["target"] = grouped[data_cfg.target_col].shift(-horizon)

    feature_columns = _feature_column_names(feat_cfg)
    artifacts = FeatureArtifacts(
        feature_columns=feature_columns,
        sku_encoder=sku_encoder if feat_cfg.encode_sku else None,
        min_date=min_date,
    )
    return out, artifacts


def _feature_column_names(feat_cfg: FeatureConfig) -> list[str]:
    cols: list[str] = []
    cols.extend(f"sales_lag_{d}" for d in feat_cfg.lag_days)
    for window in feat_cfg.rolling_windows:
        cols.extend([f"rolling_mean_{window}", f"rolling_std_{window}"])
    if feat_cfg.include_calendar:
        cols.extend(["day_of_week", "week_of_year", "month", "quarter", "weekend_flag"])
    if feat_cfg.include_trend:
        cols.append("days_since_start")
    if feat_cfg.encode_sku:
        cols.append("sku_id")
    return cols


def train_test_split_by_date(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffle."""
    cutoff = pd.Timestamp(data_cfg.test_start_date)
    mask = df[data_cfg.date_col] < cutoff
    train_df = df.loc[mask].copy()
    test_df = df.loc[~mask].copy()
    return train_df, test_df


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Drop rows with NaN target or features (warm-up / horizon tail)."""
    cols = feature_columns + ["target"]
    clean = df.dropna(subset=cols)
    x = clean[feature_columns].to_numpy(dtype=float)
    y = clean["target"].to_numpy(dtype=float)
    return x, y, clean
