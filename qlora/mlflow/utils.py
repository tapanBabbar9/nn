"""Shared helpers for loading models, datasets, and formatting prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import LoRAConfig, ModelConfig, QLoRAConfig


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl_dataset(path: str) -> Dataset:
    """Load instruction-tuning examples from JSONL."""
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example: dict[str, str], tokenizer) -> dict[str, Any]:
    """
    Build chat-formatted text and mask prompt tokens so loss is only on the response.
    Works with models that support apply_chat_template (Qwen, Llama 3.2, Phi-3).
    """
    messages = [
        {"role": "user", "content": f"{example['instruction']}\n\nReview: {example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Prompt-only (no assistant reply) for masking
    prompt_messages = [messages[0]]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, truncation=True, max_length=512)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=512)["input_ids"]

    labels = full_ids.copy()
    # -100 = ignored in CrossEntropyLoss
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

    return {"input_ids": full_ids, "labels": labels}


def load_tokenizer(model_cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_id,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_bnb_config(qlora_cfg: QLoRAConfig) -> BitsAndBytesConfig | None:
    if not qlora_cfg.use_qlora:
        return None
    dtype = torch.bfloat16 if qlora_cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.load_in_4bit,
        bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=qlora_cfg.use_double_quant,
    )


def load_base_model(
    model_cfg: ModelConfig,
    qlora_cfg: QLoRAConfig,
    *,
    for_training: bool = True,
):
    """Load base (optionally 4-bit) causal LM."""
    device = get_device()
    bnb_config = build_bnb_config(qlora_cfg) if qlora_cfg.use_qlora and device == "cuda" else None

    kwargs: dict[str, Any] = {
        "trust_remote_code": model_cfg.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    elif device == "mps":
        # device_map="auto" can offload layers to disk/meta, which breaks LoRA backward on MPS
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_id, **kwargs)

    if device == "mps" and bnb_config is None:
        model = model.to(device)

    if for_training:
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    return model


def count_parameters(model) -> dict[str, float]:
    """Return total vs trainable param counts (in millions)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_m": total / 1e6,
        "trainable_m": trainable / 1e6,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def build_chat_prompt(system: str, user: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def estimate_adapter_params(
    hidden_size: int,
    num_layers: int,
    rank: int,
    target_module_count: int = 7,
) -> int:
    """
    Rough LoRA param count per layer per module: 2 * hidden * rank.
    Interview talking point: trainable params scale with r, not model size.
    """
    per_layer = target_module_count * 2 * hidden_size * rank
    return per_layer * num_layers


def model_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device(model))
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
