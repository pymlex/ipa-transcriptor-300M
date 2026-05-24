from pydantic import BaseModel


class TrainConfig(BaseModel):
    seed: int = 42
    data_dir: str = "data/processed"
    run_dir: str = "runs"
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    model_name_or_path: str = "google/byt5-small"
    source_prefix: str = "ipa: "
    max_source_length: int = 36
    max_target_length: int = 56
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    num_workers: int = 2
    epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.0
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    logging_steps: int = 25
    eval_steps: int = 500
    save_steps: int = 500
    beam_size: int = 4
    max_new_tokens: int = 56
    generation_num_beams: int = 4
    benchmark_after_train: bool = False
    benchmark_max_samples: int | None = None


class BenchmarkResult(BaseModel):
    split: str
    n_samples: int
    loss: float
    perplexity: float
    token_accuracy: float
    exact_match: float
    char_accuracy: float
    cer: float
    bleu: float
