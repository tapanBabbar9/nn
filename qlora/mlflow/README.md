# MLflow track — LoRA / QLoRA + Model Registry

Same training task as [`../baseline/`](../baseline/), with MLflow for:

- **Experiment tracking** — params, train loss, git SHA
- **Lineage** — dataset + adapter artifacts per run
- **Metrics** — eval logged to the same run
- **Model registry** — register versions, promote Staging → Production

## Setup

```bash
cd qlora
python -m venv .venv
source .venv/bin/activate
pip install -r mlflow/requirements.txt
cd mlflow
```

## Workflow

```bash
# 1. Train (creates MLflow run + logs dataset/adapter)
python train.py

# 2. Evaluate (logs metrics to the same run)
python evaluate.py

# 3. Register + promote
python register.py --promote staging
python register.py --promote production

# 4. UI (must point at the SQLite backend)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 — check **Experiments**, **Runs**, and **Models**.

## What gets logged

| Item | MLflow |
|------|--------|
| Params | `base_model`, `lora_r`, `learning_rate`, `seed`, `git_sha`, … |
| Dataset | `dataset/movie_reviews.jsonl` artifact |
| Train metrics | `train_loss`, `learning_rate`, `epoch` |
| Eval metrics | `lora_sentiment_accuracy`, `lora_rouge1`, deltas vs base |
| Model | `adapter/` artifacts → registered as `MovieReviewLoRA` |

Last run id is saved to `outputs/last_mlflow_run.txt` for evaluate/register.

## Promotion gates

Configured in `config.py` (`MLflowConfig`):

- **Staging** — `lora_sentiment_accuracy` ≥ threshold (default 0.5)
- **Production** — sentiment + `lora_format_compliance_strict` thresholds

Override gates for learning: `python register.py --promote staging --force`

## Interview talking points

- **Lineage** — run links dataset artifact, hyperparams, adapter, eval metrics
- **Reproducibility** — seed + params + dataset path + optional `git_sha`
- **Governance** — registry stages with metric gates before Production
- **Model registry** — versioned adapters, Staging/Production, rollback via UI
