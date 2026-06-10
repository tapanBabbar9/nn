# LoRA / QLoRA Fine-Tuning

Fine-tune a small instruct model on movie review sentiment using Hugging Face Transformers, PEFT, LoRA, and QLoRA.

## Setup

```bash
cd qlora
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Parameter and memory estimates (no GPU required)
python stats.py

# Train on the included dataset
python train.py

# Compare base model vs fine-tuned adapter
python compare.py

# Evaluate on the held-out test set
python evaluate.py
```

### Base model

Change the model in `config.py` or via CLI:

```bash
python train.py --model-id Qwen/Qwen2.5-3B-Instruct
python train.py --model-id meta-llama/Llama-3.2-3B-Instruct
python train.py --model-id microsoft/Phi-3-mini-4k-instruct
```

Llama models require accepting the license on Hugging Face and running `huggingface-cli login`.

### Hardware

| Device | Mode | Notes |
|--------|------|-------|
| NVIDIA GPU (8GB+) | QLoRA | Default — 4-bit via `bitsandbytes` |
| Apple Silicon (MPS) | LoRA fp16/bf16 | No 4-bit on Mac; falls back automatically |
| CPU | LoRA | Slow; use `--epochs 1` for a quick smoke test |

```bash
# Full-precision LoRA (more VRAM)
python train.py --no-qlora
```

## Dataset

Training data: `data/movie_reviews.jsonl`  
Test data: `data/movie_reviews_test.jsonl`

Each line is a JSON object with three fields:

```json
{
  "instruction": "Analyze this movie review.",
  "input": "Absolutely loved it! The cinematography was breathtaking.",
  "output": "SENTIMENT: positive\nSUMMARY: Enthusiastic praise for visuals."
}
```

Replace or extend these files with your own examples using the same schema. Keep test examples out of the training file.

## Project layout

```
qlora/
├── config.py              # Model, LoRA, QLoRA, and training settings
├── train.py               # Fine-tuning script
├── compare.py             # Base vs adapter inference
├── evaluate.py            # Accuracy, ROUGE, format checks
├── metrics.py             # Scoring helpers
├── stats.py               # Parameter and memory estimates
├── utils.py               # Tokenization and model loading
├── data/
│   ├── movie_reviews.jsonl
│   └── movie_reviews_test.jsonl
└── outputs/
    └── lora-movie-reviews/   # Saved adapter weights
```

## Background

**LoRA** adds low-rank matrices to frozen model weights instead of updating the full parameter set. Only the adapter weights are saved (typically tens of MB rather than a full model checkpoint).

**QLoRA** loads the base model in 4-bit precision and trains LoRA adapters on top. This reduces GPU memory enough to fine-tune a ~3B model on a single consumer GPU.

Default settings in `config.py`:

```python
LoRAConfig.r = 8
LoRAConfig.lora_alpha = 16
QLoRAConfig.use_qlora = True
TrainConfig.learning_rate = 2e-4
```

Run `python stats.py --rank 4 8 16 32` to compare parameter counts for different ranks.

After training, `compare.py` shows whether the adapter follows the `SENTIMENT:` / `SUMMARY:` output format better than the base model.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `bitsandbytes` error on Mac | Training falls back to LoRA without 4-bit |
| OOM on GPU | Lower batch size in `config.py` or use a smaller rank |
| Llama 403 | Accept the license on Hugging Face and log in with the CLI |
| Slow on CPU | `python train.py --epochs 1` to validate the pipeline |

## References

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PEFT docs](https://huggingface.co/docs/peft)
- [HF fine-tuning guide](https://huggingface.co/docs/transformers/training)
