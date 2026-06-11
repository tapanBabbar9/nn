#!/usr/bin/env python3
"""
Print LoRA parameter math and memory estimates for interview prep.

Usage:
  python stats.py
  python stats.py --rank 8 16 32
  python stats.py --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse

from transformers import AutoConfig

from config import LoRAConfig, ModelConfig
from utils import estimate_adapter_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=ModelConfig().model_id)
    p.add_argument("--rank", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--target-modules", type=int, default=7, help="Attention + MLP linears per layer")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    hidden = cfg.hidden_size
    layers = cfg.num_hidden_layers
    total_params_b = getattr(cfg, "num_parameters", None)
    if total_params_b is None:
        # rough estimate for 3B decoder-only
        total_params_b = (12 * layers * hidden * hidden) / 1e9

    print("=" * 60)
    print(f"LoRA Stats — {args.model}")
    print("=" * 60)
    print(f"  hidden_size      : {hidden}")
    print(f"  num_layers       : {layers}")
    print(f"  ~base params     : {total_params_b:.2f}B")
    print()

    print(f"{'Rank':>6} {'Adapter Params':>16} {'% of Base':>12} {'Adapter MB (fp16)':>18}")
    print("-" * 56)

    for r in args.rank:
        n = estimate_adapter_params(hidden, layers, r, args.target_modules)
        pct = 100 * n / (total_params_b * 1e9)
        mb = n * 2 / 1e6  # fp16 bytes
        print(f"{r:>6} {n/1e6:>14.2f}M {pct:>11.4f}% {mb:>16.1f} MB")

    print()
    print("Memory talking points (3B model, order-of-magnitude):")
    print("  Full fine-tune fp16 : ~6 GB weights + ~12 GB optimizer states ≈ 18+ GB VRAM")
    print("  QLoRA 4-bit base    : ~2 GB base + ~0.01–0.05 GB adapters ≈ fits 8 GB GPU")
    print("  Inference (merged)  : Same latency as base — adapters fused into weights")
    print("  Inference (unmerged): Small overhead — extra low-rank matmuls per layer")
    print()
    lora = LoRAConfig()
    print("Default config in this project:")
    print(f"  r={lora.r}, alpha={lora.lora_alpha}, scale={lora.lora_alpha/lora.r}")
    print(f"  target_modules={lora.target_modules}")


if __name__ == "__main__":
    main()
