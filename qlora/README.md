# LoRA / QLoRA Fine-Tuning Lab

A minimal, interview-ready project for learning **Hugging Face Transformers**, **PEFT**, **LoRA**, and **QLoRA** by fine-tuning a 3B instruct model on movie review sentiment.

## What you'll learn

| Concept | Where in this repo |
|---------|-------------------|
| Hugging Face `AutoModel`, `Trainer`, chat templates | `train.py`, `utils.py` |
| PEFT adapters (`LoraConfig`, `get_peft_model`) | `train.py` |
| LoRA (low-rank adapters on frozen weights) | `train.py`, `config.py` |
| QLoRA (4-bit base + LoRA training) | `train.py`, `config.py` |
| Base vs fine-tuned inference | `compare.py` |
| Quantitative eval (accuracy, ROUGE, format) | `evaluate.py` |
| Rank / memory math | `stats.py`, README below |

## Quick start

```bash
cd qlora
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# See LoRA parameter math (no GPU needed)
python stats.py

# Train tiny LoRA (~20 examples, few minutes on GPU)
python train.py

# Compare base model vs your adapter
python compare.py

# Quantitative eval on held-out test set
python evaluate.py
```

### Switch base model

Edit `config.py` or pass CLI flags:

```bash
# Qwen 3B (default)
python train.py --model-id Qwen/Qwen2.5-3B-Instruct

# Llama 3.2 3B
python train.py --model-id meta-llama/Llama-3.2-3B-Instruct

# Phi-3 mini
python train.py --model-id microsoft/Phi-3-mini-4k-instruct
```

> **Note:** Llama models require accepting the license on Hugging Face and `huggingface-cli login`.

### Hardware

| Device | Mode | Notes |
|--------|------|-------|
| NVIDIA GPU (8GB+) | QLoRA | Default — `bitsandbytes` 4-bit |
| Apple Silicon (MPS) | LoRA fp16/bf16 | Auto-fallback; no 4-bit on Mac |
| CPU | LoRA | Slow; use `--epochs 1` for smoke test |

```bash
# Force full-precision LoRA (more VRAM)
python train.py --no-qlora
```

## Dataset (placeholder — swap yours)

File: `data/movie_reviews.jsonl`

Each line is one training example:

```json
{
  "instruction": "Analyze this movie review.",
  "input": "Absolutely loved it! The cinematography was breathtaking.",
  "output": "SENTIMENT: positive\nSUMMARY: Enthusiastic praise for visuals."
}
```

**To use your own data:** replace or append lines in that file. Keep the same three keys. Aim for 50–500 examples for noticeable behavior change; 20 is enough to demo the pipeline.

**Test set:** `data/movie_reviews_test.jsonl` — held-out examples for `evaluate.py`. Never train on these; swap with your own labels.

## Project layout

```
qlora/
├── config.py              # Model, LoRA, QLoRA, training settings
├── train.py               # Fine-tune with PEFT
├── compare.py             # Base vs LoRA side-by-side
├── evaluate.py            # Sentiment accuracy, ROUGE, format compliance
├── metrics.py             # Parsing + scoring helpers
├── stats.py               # Parameter & memory estimates
├── utils.py               # Tokenization, model loading
├── data/
│   └── movie_reviews.jsonl
│   └── movie_reviews_test.jsonl
└── outputs/
    └── lora-movie-reviews/   # Saved adapter (~10–30 MB)
```

---

## Interview cheat sheet

Use this section to confidently discuss adapters, rank, memory, and inference.

### What is LoRA?

**LoRA (Low-Rank Adaptation)** freezes the pretrained weight matrix \(W\) and learns a small update \(\Delta W = BA\), where:

- \(B \in \mathbb{R}^{d \times r}\), \(A \in \mathbb{R}^{r \times k}\)
- **rank** \(r \ll \min(d, k)\) (this project defaults to **r=8**)

Forward pass becomes: \(h = Wx + BAx\) (with scaling \(\alpha/r\)).

**Why it works:** Fine-tuning updates often lie in a low-dimensional subspace. LoRA captures that subspace with far fewer parameters.

### What are adapters (PEFT)?

In **PEFT (Parameter-Efficient Fine-Tuning)**, you attach small trainable modules to a frozen base model:

- **LoRA adapters** — low-rank matrices on linear layers (`q_proj`, `v_proj`, MLP layers, etc.)
- Only adapter weights are saved (`adapter_model.safetensors`, ~10–50 MB)
- Multiple adapters can target one base model (swap per task)

This repo saves adapters to `outputs/lora-movie-reviews/`, not a full 6 GB checkpoint.

### What is QLoRA?

**QLoRA** = **Q**uantized base model (typically **4-bit NF4**) + **LoRA** training.

- Base weights stored in 4-bit; computations often in bf16
- **`prepare_model_for_kbit_training`** enables stable gradients through quantized layers
- Lets you fine-tune a 3B model on a **single consumer GPU (~8 GB)**

LoRA without quantization needs ~6 GB just for fp16 weights; QLoRA cuts base memory ~4×.

### Rank — what to say in an interview

| Rank (r) | Trainable params | Capacity | Risk |
|----------|------------------|----------|------|
| 4 | Smallest | May underfit complex tasks | Fast, tiny adapter |
| 8 | **Default here** | Good for format/style tasks | Sweet spot for demos |
| 16–32 | Larger | Better for domain knowledge | Diminishing returns |
| 64+ | Approaching full FT | Heavy tasks | Less "parameter efficient" |

**Rule of thumb:** Start with r=8, alpha=16 (scale=2). Double r if underfitting; halve if overfitting on tiny data.

Run `python stats.py --rank 4 8 16 32` to see exact numbers for your model.

### Memory savings

For a **3B model** (order of magnitude):

| Method | VRAM (train) | What's trainable |
|--------|--------------|------------------|
| Full fine-tune fp16 | ~18–24 GB | All ~3B params + optimizer |
| LoRA fp16 | ~8–12 GB | ~0.01–0.1% of params |
| **QLoRA 4-bit** | **~6–8 GB** | Adapters only, base frozen in 4-bit |

**Optimizer states** (Adam) store 2× momentum buffers per trainable param — that's why full FT is so expensive and LoRA is not.

**Disk:** Full model ~6 GB vs LoRA adapter ~15–30 MB.

### Inference impact

Three deployment patterns:

1. **Merged weights** (`merge_and_unload()`) — adapter baked into \(W\). **Zero latency overhead.** Same memory as base model.

2. **Unmerged adapters** (default PEFT) — extra \(BAx\) per adapted layer. **~5–15% slower**, slightly more memory for adapter weights.

3. **Multi-adapter** — load base once, hot-swap LoRA for different tasks (customer support vs coding vs sentiment).

For production at scale, teams usually **merge** after training. For experimentation, **unmerged** is fine.

**Quantization at inference:** QLoRA training doesn't require 4-bit at inference — you can merge adapters into fp16/bf16 or export to GGUF/ONNX separately.

---

## Expected compare.py output

After training, the **LoRA model** should follow your training format:

```
SENTIMENT: negative
SUMMARY: Criticism of predictable plot and weak acting.
```

The **base model** may ramble, use different labels, or ignore the `SENTIMENT:` schema — that's the point of the demo.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `bitsandbytes` error on Mac | Expected — training auto-uses LoRA without 4-bit |
| OOM on GPU | `--no-qlora` won't help; reduce batch size in `config.py` or use smaller rank |
| Llama 403 | Accept license on HF + `huggingface-cli login` |
| Slow on CPU | `python train.py --epochs 1` for pipeline validation only |

## Key hyperparameters (`config.py`)

```python
LoRAConfig.r = 8              # rank
LoRAConfig.lora_alpha = 16     # scaling (effective scale = alpha/r = 2)
LoRAConfig.target_modules      # which layers get adapters
QLoRAConfig.use_qlora = True    # 4-bit on CUDA
TrainConfig.learning_rate = 2e-4  # typical for LoRA
```

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PEFT docs](https://huggingface.co/docs/peft)
- [HF fine-tuning guide](https://huggingface.co/docs/transformers/training)
