#!/usr/bin/env python3
"""
Generate synthetic daily SKU sales — ~6 years for interview demos.

Usage:
  python generate_data.py
  python generate_data.py --output ../data/sku_daily_sales.csv --n-skus 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DataConfig, SyntheticDataConfig


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic SKU daily sales CSV")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--n-skus", type=int, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def generate_synthetic_sales(cfg: SyntheticDataConfig) -> pd.DataFrame:
    """
    Realistic-ish panel: weekly seasonality, slow trend, SKU heterogeneity,
  occasional promotion spikes.
    """
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    n_days = len(dates)

    categories = ["grocery", "electronics", "apparel", "office", "home"]
    rows: list[dict] = []

    for i in range(cfg.n_skus):
        sku = f"SKU_{101 + i}"
        category = categories[i % len(categories)]
        base = rng.uniform(8, 120)
        trend = rng.uniform(-0.0005, 0.001)
        weekend_boost = 1.25 if category in ("grocery", "apparel") else 0.85
        promo_prob = 0.02

        for j, date in enumerate(dates):
            dow = date.dayofweek
            seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * dow / 7)
            if dow >= 5:
                seasonal *= weekend_boost
            monthly = 1.0 + 0.08 * np.sin(2 * np.pi * date.month / 12)
            promo = rng.random() < promo_prob
            promo_mult = rng.uniform(1.4, 2.2) if promo else 1.0
            noise = rng.normal(0, base * 0.08)

            units = max(0, base * (1 + trend * j) * seasonal * monthly * promo_mult + noise)
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "sku": sku,
                    "category": category,
                    "units_sold": int(round(units)),
                    "promotion": int(promo),
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    data_cfg = DataConfig()
    synth_cfg = SyntheticDataConfig()

    if args.output:
        data_cfg.data_path = args.output
    if args.n_skus:
        synth_cfg.n_skus = args.n_skus
    if args.start:
        synth_cfg.start_date = args.start
    if args.end:
        synth_cfg.end_date = args.end
    if args.seed is not None:
        synth_cfg.seed = args.seed

    print("=" * 60)
    print("Synthetic SKU daily sales")
    print("=" * 60)
    print(f"  SKUs   : {synth_cfg.n_skus}")
    print(f"  Range  : {synth_cfg.start_date} → {synth_cfg.end_date}")
    print(f"  Output : {data_cfg.data_path}")
    print()

    df = generate_synthetic_sales(synth_cfg)
    out = Path(data_cfg.data_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_days = df["date"].nunique()
    print(f"Wrote {len(df):,} rows ({synth_cfg.n_skus} SKUs × {n_days} days)")
    print(f"Saved to {out.resolve()}")
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()
