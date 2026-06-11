#!/usr/bin/env python3
"""
Train a LoRA / QLoRA adapter with MLflow experiment tracking.

Same training logic as ../baseline/train.py, plus:
  - experiment + run logging
  - params, dataset artifact, train metrics
  - adapter artifacts for model registry

Usage:
  python train.py
  python train.py --rank 16
  mlflow ui   # from this directory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from config import LoRAConfig, MLflowConfig, ModelConfig, QLoRAConfig, TrainConfig
from mlflow_tracking import (
    log_adapter_artifacts,
    log_dataset_artifact,
    log_train_metrics_from_history,
    log_training_params,
    save_run_id,
    setup_mlflow,
)
from utils import (
    count_parameters,
    estimate_adapter_params,
    format_example,
    get_device,
    load_base_model,
    load_jsonl_dataset,
    load_tokenizer,
)


def parse_args():
    p = argparse.ArgumentParser(description="LoRA / QLoRA fine-tuning with MLflow")
    p.add_argument("--model-id", type=str, default=None, help="Override base model HF id")
    p.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--rank", type=int, default=None, help="LoRA rank (r)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--no-qlora", action="store_true", help="Disable 4-bit QLoRA")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig()
    lora_cfg = LoRAConfig()
    qlora_cfg = QLoRAConfig()
    train_cfg = TrainConfig()
    mlflow_cfg = setup_mlflow()

    if args.model_id:
        model_cfg.model_id = args.model_id
    if args.dataset:
        train_cfg.dataset_path = args.dataset
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.rank:
        lora_cfg.r = args.rank
    if args.epochs:
        train_cfg.num_train_epochs = args.epochs
    if args.no_qlora:
        qlora_cfg.use_qlora = False
    if args.lr:
        train_cfg.learning_rate = args.lr

    device = get_device()
    if qlora_cfg.use_qlora and device != "cuda":
        print(f"[info] QLoRA requires CUDA; running LoRA in {device} instead.")
        qlora_cfg.use_qlora = False

    print("=" * 60)
    print("LoRA Fine-Tuning — Movie Reviews (MLflow)")
    print("=" * 60)
    print(f"  Base model : {model_cfg.model_id}")
    print(f"  Mode       : {'QLoRA (4-bit)' if qlora_cfg.use_qlora else 'LoRA (fp16/bf16)'}")
    print(f"  Device     : {device}")
    print(f"  LoRA rank  : {lora_cfg.r}  (alpha={lora_cfg.lora_alpha})")
    print(f"  Dataset    : {train_cfg.dataset_path}")
    print(f"  Output     : {train_cfg.output_dir}")
    print(f"  Experiment : {mlflow_cfg.experiment_name}")
    print()

    Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_cfg)
    raw_ds = load_jsonl_dataset(train_cfg.dataset_path)
    print(f"Loaded {len(raw_ds)} training examples")

    tokenized = raw_ds.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=raw_ds.column_names,
    )

    model = load_base_model(model_cfg, qlora_cfg, for_training=True)

    if qlora_cfg.use_qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=lora_cfg.target_modules,
    )
    model = get_peft_model(model, peft_config)

    stats = count_parameters(model)
    est = estimate_adapter_params(
        hidden_size=model.config.hidden_size,
        num_layers=model.config.num_hidden_layers,
        rank=lora_cfg.r,
    )
    print(f"Total params     : {stats['total_m']:.1f}M")
    print(f"Trainable params : {stats['trainable_m']:.2f}M ({stats['trainable_pct']:.2f}%)")
    print(f"Estimated LoRA   : ~{est / 1e6:.2f}M (theoretical)")
    print()
    model.print_trainable_parameters()

    meta = {
        "base_model": model_cfg.model_id,
        "mode": "qlora" if qlora_cfg.use_qlora else "lora",
        "lora_r": lora_cfg.r,
        "lora_alpha": lora_cfg.lora_alpha,
        "target_modules": lora_cfg.target_modules,
        "dataset": train_cfg.dataset_path,
        "num_examples": len(raw_ds),
        "parameter_stats": stats,
    }
    with open(Path(train_cfg.output_dir) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    batch_size = train_cfg.per_device_train_batch_size
    if device == "mps" and batch_size > 1:
        print("[info] MPS: using batch size 1 to reduce memory pressure.")
        batch_size = 1

    use_bf16 = train_cfg.bf16 and device == "cuda"
    use_fp16 = device in ("cuda", "mps") and not use_bf16

    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        dataloader_pin_memory=device == "cuda",
        report_to="none",
        seed=train_cfg.seed,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    with mlflow.start_run(run_name=args.run_name) as run:
        log_training_params(
            model_id=model_cfg.model_id,
            mode="qlora" if qlora_cfg.use_qlora else "lora",
            lora_r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            learning_rate=train_cfg.learning_rate,
            num_epochs=train_cfg.num_train_epochs,
            seed=train_cfg.seed,
            dataset_path=train_cfg.dataset_path,
            num_examples=len(raw_ds),
            trainable_m=stats["trainable_m"],
        )
        log_dataset_artifact(train_cfg.dataset_path)
        mlflow.set_tag("pipeline", "qlora-mlflow")
        mlflow.set_tag("stage", "training")

        print("Starting training...")
        trainer.train()
        log_train_metrics_from_history(trainer.state.log_history)

        model.save_pretrained(train_cfg.output_dir)
        tokenizer.save_pretrained(train_cfg.output_dir)
        log_adapter_artifacts(train_cfg.output_dir)
        save_run_id(run.info.run_id, mlflow_cfg.run_id_path)

        print(f"\nMLflow run id : {run.info.run_id}")
        print(f"Adapter saved : {train_cfg.output_dir}")
        print("Next:")
        print("  python evaluate.py")
        print("  python register.py --promote staging")
        print("  mlflow ui")


if __name__ == "__main__":
    main()
