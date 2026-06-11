#!/usr/bin/env python3
"""
Evaluate base vs LoRA and log metrics to MLflow.

Usage:
  python evaluate.py
  python evaluate.py --run-id <mlflow_run_id>
  python evaluate.py --adapter outputs/lora-movie-reviews
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
from peft import PeftModel
from rouge_score import rouge_scorer

from config import EvalConfig, MLflowConfig, ModelConfig, QLoRAConfig
from metrics import EvalScores, parse_reference, print_comparison, score_example
from mlflow_tracking import load_run_id, log_eval_metrics, setup_mlflow
from utils import build_chat_prompt, generate_text, get_device, load_base_model, load_jsonl_dataset, load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate base vs LoRA with MLflow metrics")
    p.add_argument("--test", type=str, default=None, help="Path to test JSONL")
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="Where to save JSON results")
    p.add_argument("--run-id", type=str, default=None, help="MLflow run to attach eval metrics")
    p.add_argument("--no-qlora", action="store_true")
    return p.parse_args()


def _free_device_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def run_model_on_test(
    model,
    tokenizer,
    examples,
    system_prompt: str,
    scorer,
    *,
    max_new_tokens: int,
    temperature: float,
    label: str,
) -> EvalScores:
    scores = EvalScores()
    for i, ex in enumerate(examples, 1):
        gold_sentiment, gold_summary = parse_reference(ex["output"])
        user_msg = f"{ex['instruction']}\n\nReview: {ex['input']}"
        chat_prompt = build_chat_prompt(system_prompt, user_msg, tokenizer)

        print(f"  [{label}] {i}/{len(examples)}...", end="\r")
        prediction = generate_text(
            model, tokenizer, chat_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        scores.add(**score_example(prediction, gold_sentiment, gold_summary, scorer))
    print()
    return scores


def main():
    args = parse_args()
    model_cfg = ModelConfig()
    eval_cfg = EvalConfig()
    qlora_cfg = QLoRAConfig()
    mlflow_cfg = setup_mlflow()

    if args.test:
        eval_cfg.test_path = args.test
    if args.adapter:
        eval_cfg.adapter_path = args.adapter
    if args.output:
        eval_cfg.results_path = args.output
    if args.model_id:
        model_cfg.model_id = args.model_id
    if args.no_qlora:
        qlora_cfg.use_qlora = False

    run_id = args.run_id or load_run_id(mlflow_cfg.run_id_path)
    if not run_id:
        print("No MLflow run id found. Train first: python train.py")
        print("Or pass: python evaluate.py --run-id <run_id>")
        return

    device = get_device()
    if qlora_cfg.use_qlora and device != "cuda":
        qlora_cfg.use_qlora = False

    adapter_path = Path(eval_cfg.adapter_path)
    if not adapter_path.exists():
        print(f"Adapter not found at {adapter_path}. Train first: python train.py")
        return

    meta_path = adapter_path / "training_meta.json"
    if meta_path.exists() and not args.model_id:
        with open(meta_path) as f:
            meta = json.load(f)
        model_cfg.model_id = meta.get("base_model", model_cfg.model_id)

    test_path = Path(eval_cfg.test_path)
    if not test_path.exists():
        print(f"Test set not found at {test_path}")
        return

    examples = load_jsonl_dataset(str(test_path))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    print("=" * 60)
    print("QUANTITATIVE EVAL — Base vs LoRA (MLflow)")
    print("=" * 60)
    print(f"  Base model : {model_cfg.model_id}")
    print(f"  Adapter    : {eval_cfg.adapter_path}")
    print(f"  Test set   : {eval_cfg.test_path} ({len(examples)} examples)")
    print(f"  MLflow run : {run_id}")
    print(f"  Device     : {device}")
    print()

    tokenizer = load_tokenizer(model_cfg)
    print("Loading model...")
    model = load_base_model(model_cfg, qlora_cfg, for_training=False)

    print("Running base model on test set...")
    base_scores = run_model_on_test(
        model, tokenizer, examples, eval_cfg.system_prompt, scorer,
        max_new_tokens=eval_cfg.max_new_tokens,
        temperature=eval_cfg.temperature,
        label="base",
    )

    model = PeftModel.from_pretrained(model, eval_cfg.adapter_path)
    _free_device_memory(device)

    print("Running LoRA model on test set...")
    lora_scores = run_model_on_test(
        model, tokenizer, examples, eval_cfg.system_prompt, scorer,
        max_new_tokens=eval_cfg.max_new_tokens,
        temperature=eval_cfg.temperature,
        label="lora",
    )

    base_summary = base_scores.summary_dict("base")
    lora_summary = lora_scores.summary_dict("lora")

    print_comparison(base_summary, lora_summary)

    results = {
        "base_model": model_cfg.model_id,
        "adapter": eval_cfg.adapter_path,
        "test_path": eval_cfg.test_path,
        "mlflow_run_id": run_id,
        "n_examples": len(examples),
        "base": base_summary,
        "lora": lora_summary,
        "per_example": {
            "base": base_scores.per_example,
            "lora": lora_scores.per_example,
        },
    }

    out_path = Path(eval_cfg.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    with mlflow.start_run(run_id=run_id):
        log_eval_metrics(lora_summary, base_summary)
        mlflow.log_artifact(str(out_path), artifact_path="eval")
        mlflow.log_artifact(str(test_path), artifact_path="dataset")
        mlflow.set_tag("stage", "evaluation")

    print(f"\nEval metrics logged to MLflow run {run_id}")
    print("Next: python register.py --promote staging")


if __name__ == "__main__":
    main()
