# LoRA / QLoRA Fine-Tuning

Fine-tune a small instruct model on movie review sentiment using Hugging Face Transformers, PEFT, LoRA, and QLoRA.

Two **independent** implementations of the same task:

| | [`baseline/`](baseline/) | [`mlflow/`](mlflow/) |
|---|---|---|
| **Purpose** | Local training pipeline | Same pipeline + MLOps |
| **Metadata** | `training_meta.json` on disk | MLflow runs + artifacts |
| **Metrics** | Printed + `eval_results.json` | Logged to MLflow |
| **Model storage** | `outputs/lora-movie-reviews/` | Artifacts + Model Registry |
| **Governance** | Manual | Staging / Production promotion |

Shared dataset: [`data/`](data/) (both tracks use `../data/` from their folder).

## Setup

```bash
cd qlora
python -m venv .venv
source .venv/bin/activate

# Baseline only
pip install -r requirements.txt

# MLflow track (includes baseline deps)
pip install -r mlflow/requirements.txt
```

## Quick start

**Baseline** (no MLflow):

```bash
cd baseline
python train.py
python evaluate.py
```

**MLflow** (Day 4 MLOps):

```bash
cd mlflow
python train.py
python evaluate.py
python register.py --promote staging
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

See [`baseline/README.md`](baseline/README.md) and [`mlflow/README.md`](mlflow/README.md) for details.

## Dataset

Training: `data/movie_reviews.jsonl`  
Test: `data/movie_reviews_test.jsonl`

Each line:

```json
{
  "instruction": "Analyze this movie review.",
  "input": "Absolutely loved it! The cinematography was breathtaking.",
  "output": "SENTIMENT: positive\nSUMMARY: Enthusiastic praise for visuals."
}
```

## Base model

Default: `Qwen/Qwen2.5-3B-Instruct`. Override in either track:

```bash
python train.py --model-id Qwen/Qwen2.5-3B-Instruct
python train.py --model-id meta-llama/Llama-3.2-3B-Instruct
```

Llama models require accepting the license on Hugging Face and `huggingface-cli login`.

## Hardware

| Device | Mode | Notes |
|--------|------|-------|
| NVIDIA GPU (8GB+) | QLoRA | Default — 4-bit via `bitsandbytes` |
| Apple Silicon (MPS) | LoRA fp16/bf16 | No 4-bit on Mac; falls back automatically |
| CPU | LoRA | Slow; use `--epochs 1` for a smoke test |

## Background

**LoRA** adds low-rank matrices to frozen weights; only adapter weights are saved (~tens of MB).

**QLoRA** loads the base model in 4-bit and trains adapters on top — fits ~3B models on a single consumer GPU.

Default settings (`baseline/config.py` or `mlflow/config.py`):

```python
LoRAConfig.r = 8
LoRAConfig.lora_alpha = 16
QLoRAConfig.use_qlora = True
TrainConfig.learning_rate = 2e-4
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `bitsandbytes` error on Mac | Training falls back to LoRA without 4-bit |
| OOM on GPU | Lower batch size in `config.py` or use smaller `--rank` |
| Llama 403 | Accept license on HF and log in with CLI |
| Slow on CPU | `python train.py --epochs 1` |
| MLflow UI empty | Run from `mlflow/` and use `mlflow ui --backend-store-uri sqlite:///mlflow.db` |

## References

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PEFT docs](https://huggingface.co/docs/peft)
- [MLflow docs](https://mlflow.org/docs/latest/)
