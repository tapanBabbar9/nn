#!/usr/bin/env python3
"""
Compare base model vs LoRA-tuned model on the same prompts.

Usage:
  python compare.py
  python compare.py --adapter outputs/lora-movie-reviews
  python compare.py --prompt "Review: Mediocre at best."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CompareConfig, ModelConfig, QLoRAConfig
from utils import build_chat_prompt, count_parameters, generate_text, get_device, load_base_model, load_tokenizer


def _free_device_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description="Base vs LoRA comparison")
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None, help="Single review to test")
    p.add_argument("--no-qlora", action="store_true", help="Load base without 4-bit")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig()
    compare_cfg = CompareConfig()
    qlora_cfg = QLoRAConfig()

    if args.model_id:
        model_cfg.model_id = args.model_id
    if args.adapter:
        compare_cfg.lora_adapter_path = args.adapter
    if args.no_qlora:
        qlora_cfg.use_qlora = False

    device = get_device()
    if qlora_cfg.use_qlora and device != "cuda":
        qlora_cfg.use_qlora = False

    adapter_path = Path(compare_cfg.lora_adapter_path)
    if not adapter_path.exists():
        print(f"Adapter not found at {adapter_path}")
        print("Train first:  python train.py")
        return

    meta_path = adapter_path / "training_meta.json"
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print("Training metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    # Use the same base model the adapter was trained on (unless overridden)
    if not args.model_id and meta.get("base_model"):
        model_cfg.model_id = meta["base_model"]

    prompts = [args.prompt] if args.prompt else compare_cfg.test_prompts

    print("=" * 60)
    print("BASE MODEL vs LoRA ADAPTER")
    print("=" * 60)
    print(f"Base   : {model_cfg.model_id}")
    print(f"Adapter: {compare_cfg.lora_adapter_path}")
    print(f"Device : {device}")
    print()

    tokenizer = load_tokenizer(model_cfg)

    print("Loading model (single copy — base first, then attach adapter)...")
    model = load_base_model(model_cfg, qlora_cfg, for_training=False)
    base_stats = count_parameters(model)

    chat_prompts = []
    for review in prompts:
        user_msg = review if review.startswith("Review:") else f"Analyze this movie review.\n\n{review}"
        chat_prompts.append(build_chat_prompt(compare_cfg.system_prompt, user_msg, tokenizer))

    base_outputs = []
    for i, chat_prompt in enumerate(chat_prompts, 1):
        print(f"  Base inference {i}/{len(chat_prompts)}...")
        base_outputs.append(
            generate_text(
                model, tokenizer, chat_prompt,
                max_new_tokens=compare_cfg.max_new_tokens,
                temperature=compare_cfg.temperature,
            )
        )

    model = PeftModel.from_pretrained(model, compare_cfg.lora_adapter_path)
    lora_stats = count_parameters(model)
    _free_device_memory(device)

    print(f"\nParameter summary:")
    print(f"  Base (frozen)     : {base_stats['total_m']:.1f}M total, {base_stats['trainable_m']:.2f}M trainable")
    print(f"  Base + LoRA       : {lora_stats['total_m']:.1f}M total, {lora_stats['trainable_m']:.2f}M trainable")
    if meta.get("parameter_stats", {}).get("trainable_m"):
        print(f"  Adapter (trained) : ~{meta['parameter_stats']['trainable_m']:.2f}M trainable params")
    print()

    for i, review in enumerate(prompts, 1):
        print("-" * 60)
        print(f"Prompt {i}: {review[:80]}{'...' if len(review) > 80 else ''}")
        print("-" * 60)

        print(f"  LoRA inference {i}/{len(prompts)}...")
        lora_out = generate_text(
            model, tokenizer, chat_prompts[i - 1],
            max_new_tokens=compare_cfg.max_new_tokens,
            temperature=compare_cfg.temperature,
        )

        print("\n[BASE MODEL]")
        print(base_outputs[i - 1])
        print("\n[LoRA MODEL]")
        print(lora_out)
        print()

    print("=" * 60)
    print("Interview notes:")
    print("  • Base model: general-purpose, may ignore your output format.")
    print("  • LoRA model: follows SENTIMENT + SUMMARY format from training.")
    print("  • At inference: LoRA merges into weights OR runs as side adapters.")
    print("  • Adapter disk size: typically 10–50 MB vs multi-GB base checkpoint.")
    print("=" * 60)


if __name__ == "__main__":
    main()
