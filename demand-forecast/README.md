# SKU Demand Forecasting

Daily SKU-level demand forecasting using a **global Random Forest** model and **MLflow** for experiment tracking and model registry.

The pipeline trains one model across all SKUs (SKU encoded as a feature) rather than maintaining a separate model per product. This scales to thousands of SKUs while still capturing SKU-specific patterns.

## Dataset

Expected panel format:

| date | sku | units_sold |
|------|-----|------------|
| 2020-01-01 | SKU_101 | 25 |
| 2020-01-02 | SKU_101 | 28 |
| 2020-01-01 | SKU_102 | 12 |

Optional columns when available: `price`, `promotion`, `inventory`, `store`, `category`.

This repo ships with a synthetic generator (`mlflow/generate_data.py`) that produces ~6 years of daily history for 50 SKUs. Replace with your own CSV at `data/sku_daily_sales.csv`.

## Setup

```bash
cd demand-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r mlflow/requirements.txt
cd mlflow
```

## Run the pipeline

```bash
python generate_data.py          # synthetic data (skip if you have a CSV)
python train.py                  # train RF + log to MLflow
python evaluate.py               # evaluate on held-out period
python register.py --promote staging
python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Shortcut: `python train.py --generate` creates data automatically if the CSV is missing.

---

# Step 1: Data Cleaning

Raw supply chain data is rarely ready to model. Clean before feature engineering.

### Missing dates

Daily panels often have gaps — store closures, system outages, or sparse SKU history.

Reindex to a daily frequency per SKU:

```python
df = df.set_index("date").asfreq("D")
```

Fill missing demand depending on business rules:

```python
# No sale recorded → assume zero demand
df["units_sold"] = df["units_sold"].fillna(0)
```

Or interpolate for short gaps:

```python
df["units_sold"] = df["units_sold"].interpolate(method="linear", limit=3)
```

Apply cleaning **per SKU** so one product's calendar does not affect another.

### Outlier detection

Sudden spikes may be promotions, bulk orders, or data errors.

Use IQR:

```python
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

Or z-score for approximately normal series:

```python
z = (series - series.mean()) / series.std()
outliers = series[abs(z) > 3]
```

Example: normal demand ≈ 100 units/day, one day = 5,000 — likely a promotion or bad record. Cap, flag, or exclude based on domain knowledge.

### What this repo does

`features.py` loads and sorts the CSV but does not yet run gap-filling or outlier removal. Add a `clean.py` step (or preprocess in a notebook) before training on real data.

---

# Step 2: Feature Engineering

Most forecasting performance comes from features. Random Forest does not understand time directly — you must encode temporal structure as columns.

This repo implements the following in `mlflow/features.py`:

| Category | Features |
|----------|----------|
| Lag | `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_30` |
| Rolling | `rolling_mean_7`, `rolling_mean_30`, `rolling_std_7`, `rolling_std_30` |
| Calendar | `day_of_week`, `week_of_year`, `month`, `quarter`, `weekend_flag` |
| Trend | `days_since_start` |
| SKU | `sku_id` (label-encoded) |

### Lag features

Most important predictors for short-horizon demand.

```python
Demand_Lag_1
Demand_Lag_7
Demand_Lag_14
Demand_Lag_30
```

| Date | Demand | Lag_1 | Lag_7 |
|------|--------|-------|-------|
| Jan 8 | 120 | 110 | 100 |

Yesterday's and last week's demand are strong signals for today's forecast.

### Rolling statistics

Smooth short-term noise and capture trend:

```python
RollingMean_7
RollingMean_30
RollingStd_30
```

`rolling_mean_7` approximates the current weekly demand level.

### Calendar features

```python
DayOfWeek
Month
Quarter
WeekOfYear
IsWeekend
```

Demand often varies by weekday (e.g. grocery spikes on weekends; B2B dips on Sundays).

### Seasonal features

For yearly patterns, encode cyclical time:

```python
sin(2 * pi * day_of_year / 365)
cos(2 * pi * day_of_year / 365)
```

Or use categorical `month` and `week_of_year` (implemented here).

### Inventory features

When inventory data is available, add:

```python
CurrentInventory
Inventory_Lag_1
Inventory_Lag_7
DaysOfSupply = Inventory / AvgDemand
```

These connect demand forecasts to replenishment decisions.

### Stockout flag

Critical when inventory constrains observed sales:

```python
Stockout = 1 if Inventory <= threshold else 0
```

During stockouts, recorded demand understates true demand. Models that ignore this learn the wrong pattern. Include the flag as a feature or exclude stockout days from training.

### Forecast target

Default target: **next-day demand** (`t+1`). For longer replenishment cycles:

```bash
python train.py --horizon 7    # predict 7 days ahead
python train.py --horizon 30   # predict 30 days ahead
```

---

# Step 3: Train-Test Split

Do **not** shuffle time-series data:

```python
# Wrong for forecasting
train_test_split(X, y, shuffle=True)
```

Use a chronological split so the model never sees future data during training.

With ~6 years of history:

| Set | Period |
|-----|--------|
| Train | Years 1–5 (2020–2024) |
| Test | Year 6 (2025) |

```python
train = df[df["date"] < "2025-01-01"]
test  = df[df["date"] >= "2025-01-01"]
```

Override the cutoff:

```bash
python train.py --test-start 2024-01-01
```

---

# Step 4: Model

Global Random Forest regressor — one model, all SKUs:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

Hyperparameters live in `mlflow/config.py` (`RFConfig`). Train via:

```bash
python train.py
python train.py --n-estimators 500 --horizon 30
```

MLflow logs parameters, the fitted model, and training artifacts.

---

# Step 5: Evaluation

Report metrics on the **held-out future period** only.

### MAE (Mean Absolute Error)

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
```

Easy to explain: *"The forecast misses by 12 units on average."*

### RMSE (Root Mean Squared Error)

Penalizes large errors more than MAE. Useful when big misses are costly.

### MAPE (Mean Absolute Percentage Error)

Common in supply chain planning:

```python
MAPE = mean(abs(actual - pred) / actual) * 100
```

8% MAPE means the forecast is off by ~8% on average. MAPE is intuitive for business stakeholders but unstable when actual demand is near zero — use a small epsilon or MAE as a fallback.

```bash
python evaluate.py
```

Results are saved to `outputs/rf-demand/eval_results.json` and logged to MLflow.

---

# Step 6: Feature Importance

Inspect what drives predictions:

```python
rf.feature_importances_
```

Typical ranking for demand forecasting:

| Feature | Importance |
|---------|------------|
| Lag_7 | 30% |
| Lag_1 | 25% |
| RollingMean_30 | 15% |
| Month | 12% |
| DayOfWeek | 10% |

After training, importances are written to `outputs/rf-demand/feature_importance.csv` and logged as an MLflow artifact.

Interpretation: recent and weekly demand history, plus calendar seasonality, usually dominate. Inventory features rise in importance when stockouts are frequent.

---

# Step 7: Business Outcome

The model output feeds operational decisions:

| Use case | How forecasts are used |
|----------|------------------------|
| Replenishment | Trigger purchase orders when projected demand exceeds available inventory |
| Safety stock | Size buffers from forecast error (MAE / MAPE) and lead time |
| Allocation | Distribute inventory across stores or regions by SKU-level forecast |
| Promotion planning | Compare baseline forecast vs. expected lift during campaigns |

A practical rollout:

1. **Batch scoring** — run `evaluate.py` logic daily on the latest data to produce per-SKU forecasts.
2. **Monitor drift** — track MAPE over time in MLflow; retrain when error exceeds a threshold.
3. **Registry gates** — promote models to Staging / Production only when test MAPE meets targets (`config.py`).

```bash
python register.py --promote staging      # MAPE ≤ 25%
python register.py --promote production   # MAPE ≤ 15%
```

---

## Project layout

```
demand-forecast/
├── data/                    # sku_daily_sales.csv (generated or your own)
├── mlflow/
│   ├── config.py            # hyperparameters, split dates, MLflow settings
│   ├── features.py          # lag, rolling, calendar, SKU features
│   ├── generate_data.py     # synthetic dataset
│   ├── train.py             # train + MLflow logging
│   ├── evaluate.py          # test metrics + per-SKU breakdown
│   ├── register.py          # model registry promotion
│   └── mlflow_tracking.py   # MLflow helpers
└── README.md
```

## MLflow

See [`mlflow/README.md`](mlflow/README.md) for experiment tracking, registry, and promotion gate details.

Open the UI from `mlflow/` (same directory as `mlflow.db`):

```bash
python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `Can't locate revision identified by '...'` | MLflow version mismatch. The `mlflow` on your PATH may be a different Python than the one used to train. Use `python3 -m mlflow ui ...` from the same venv where you ran `pip install -r requirements.txt`. |
| UI shows no runs | Run commands from `demand-forecast/mlflow/` so the URI points at the correct `mlflow.db`. |
| Fresh start (deletes run history) | `rm mlflow.db` then re-run `train.py`. |

## Configuration

Edit `mlflow/config.py` to change:

- Lag windows and rolling periods (`FeatureConfig`)
- Random Forest hyperparameters (`RFConfig`)
- Train/test cutoff date (`DataConfig.test_start_date`)
- MLflow experiment name and MAPE promotion thresholds (`MLflowConfig`)
