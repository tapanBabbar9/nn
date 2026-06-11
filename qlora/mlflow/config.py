"""
Configuration for LoRA / QLoRA fine-tuning on movie reviews.

Swap MODEL_ID to try different 3B-class models:
  - Qwen2.5-3B-Instruct  (default, strong instruction following)
  - meta-llama/Llama-3.2-3B-Instruct
  - microsoft/Phi-3-mini-4k-instruct
"""

from dataclasses import dataclass, field
from typing import Optional


SYSTEM_PROMPT = (
    "You are a movie review analyst. Given a review, respond with "
    "SENTIMENT: positive|negative|mixed and a one-sentence summary."
)


@dataclass
class ModelConfig:
    # --- pick your base model ---
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    # model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    # model_id: str = "microsoft/Phi-3-mini-4k-instruct"

    max_seq_length: int = 512
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """
    LoRA hyperparameters — the ones interviewers ask about:

    r (rank):     Low-rank dimension of adapter matrices. Higher = more capacity,
                  more trainable params, slightly slower inference.
    lora_alpha:   Scaling factor; effective scale ≈ alpha / r.
    target_modules: Which linear layers get adapters (attention + MLP is standard).
    """
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class QLoRAConfig:
    """4-bit NF4 quantization — trains adapters on frozen quantized weights."""
    use_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_double_quant: bool = True  # quantize the quantization constants


@dataclass
class TrainConfig:
    dataset_path: str = "../data/movie_reviews.jsonl"
    output_dir: str = "outputs/lora-movie-reviews"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_strategy: str = "epoch"
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass
class CompareConfig:
    lora_adapter_path: str = "outputs/lora-movie-reviews"
    test_prompts: list = field(
        default_factory=lambda: [
            "Review: The plot was predictable and the acting felt wooden throughout.",
            "Review: A stunning visual masterpiece with heartfelt performances.",
            "Review: I walked out halfway — boring and poorly edited.",
        ]
    )
    system_prompt: str = SYSTEM_PROMPT
    max_new_tokens: int = 80
    temperature: float = 0.1


@dataclass
class EvalConfig:
    test_path: str = "../data/movie_reviews_test.jsonl"
    adapter_path: str = "outputs/lora-movie-reviews"
    results_path: str = "outputs/lora-movie-reviews/eval_results.json"
    system_prompt: str = SYSTEM_PROMPT
    max_new_tokens: int = 80
    temperature: float = 0.0  # greedy for reproducible metrics


@dataclass
class MLflowConfig:
    """MLflow experiment tracking and model registry settings."""
    # MLflow 3.x: use SQLite backend (filesystem ./mlruns is deprecated)
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "qlora-movie-reviews"
    registered_model_name: str = "MovieReviewLoRA"
    run_id_path: str = "outputs/last_mlflow_run.txt"
    # Minimum LoRA sentiment accuracy to allow Staging promotion
    staging_min_sentiment_accuracy: float = 0.5
    # Minimum LoRA strict format compliance for Production
    production_min_format_strict: float = 0.5
