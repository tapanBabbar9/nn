# Baseline — LoRA / QLoRA (no MLflow)

Local files only: `training_meta.json`, adapter weights under `outputs/`, eval JSON.

## Setup

From repo root:

```bash
cd qlora
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd baseline
```

## Usage

```bash
python stats.py
python train.py
python compare.py
python evaluate.py
```

Dataset paths point to shared `../data/` (run commands from this folder).

## Outputs

| Artifact | Location |
|----------|----------|
| Adapter | `outputs/lora-movie-reviews/` |
| Training metadata | `outputs/lora-movie-reviews/training_meta.json` |
| Eval results | `outputs/lora-movie-reviews/eval_results.json` |

Compare with the MLflow track in [`../mlflow/`](../mlflow/).
