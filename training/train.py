import inspect
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from data.hf_dataset import build_hf_dataset, preprocess_function
from evaluation.benchmark import run_benchmark
from schemas import TrainConfig
from utils.logging_utils import MetricsLogger
from utils.seed import set_seed


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, logger: MetricsLogger) -> None:
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is None:
            return
        epoch = float(state.epoch) if state.epoch is not None else 0.0
        if not math.isfinite(epoch) or epoch > args.num_train_epochs + 1.0:
            return
        record = dict(logs)
        if "loss" in record and not math.isfinite(record["loss"]):
            return
        if "loss" in record and record["loss"] > 50.0 and "eval_loss" not in record:
            return
        if "eval_loss" in record:
            record["phase"] = "eval"
            record["learning_rate"] = record.get("learning_rate", 0.0)
            if record["learning_rate"] > 1.0:
                record.pop("learning_rate", None)
        else:
            record["phase"] = "train"
        record["step"] = int(state.global_step)
        record["epoch"] = epoch
        self.logger.log(record)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_trainer(
    model: AutoModelForSeq2SeqLM,
    training_args: Seq2SeqTrainingArguments,
    tokenised,
    data_collator: DataCollatorForSeq2Seq,
    tokenizer: AutoTokenizer,
    callbacks: list[TrainerCallback],
) -> Seq2SeqTrainer:
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenised["train"],
        "eval_dataset": tokenised["validation"],
        "data_collator": data_collator,
        "callbacks": callbacks,
    }
    init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Seq2SeqTrainer(**trainer_kwargs)


def warmup_steps_from_ratio(
    num_train_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    warmup_ratio: float,
) -> int:
    steps_per_epoch = num_train_samples // (batch_size * gradient_accumulation_steps)
    total_steps = max(steps_per_epoch * num_epochs, 1)
    return max(int(total_steps * warmup_ratio), 1)


def resolve_mixed_precision(config: TrainConfig) -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    if config.bf16 and torch.cuda.is_bf16_supported():
        return False, True
    if config.fp16:
        return True, False
    return False, False


def run_training(config: TrainConfig, run_dir: Path) -> Path:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    set_seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_logger = MetricsLogger(run_dir)
    dataset = build_hf_dataset(config.data_dir, config.source_prefix)
    model_config = AutoConfig.from_pretrained(config.model_name_or_path)
    model_config.tie_word_embeddings = False
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        token=hf_token,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name_or_path,
        config=model_config,
        token=hf_token,
    )
    tokenised = dataset.map(
        lambda batch: preprocess_function(
            batch,
            tokenizer,
            config.max_source_length,
            config.max_target_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="tokenise",
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    use_fp16, use_bf16 = resolve_mixed_precision(config)
    warmup_steps = warmup_steps_from_ratio(
        len(tokenised["train"]),
        config.batch_size,
        config.gradient_accumulation_steps,
        config.epochs,
        config.warmup_ratio,
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=config.num_workers,
        disable_tqdm=False,
        report_to=[],
        seed=config.seed,
    )
    trainer = build_trainer(
        model,
        training_args,
        tokenised,
        data_collator,
        tokenizer,
        callbacks=[MetricsLoggerCallback(metrics_logger)],
    )
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print(
        f"precision={precision} train={len(tokenised['train'])} "
        f"val={len(tokenised['validation'])} warmup_steps={warmup_steps}",
        flush=True,
    )
    trainer.train()
    print("Training finished. Saving best checkpoint to runs/.../best", flush=True)
    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name_or_path": config.model_name_or_path,
        "source_prefix": config.source_prefix,
        "num_parameters": count_parameters(model),
        "config": config.model_dump(),
        "warmup_steps": warmup_steps,
        "precision": precision,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Checkpoint saved: {best_dir}", flush=True)
    if config.benchmark_after_train:
        print(
            "Running post-train benchmark with beam search on val and test. "
            "This can take 20–40 minutes. Use a separate evaluate cell instead.",
            flush=True,
        )
        run_benchmark(
            str(best_dir),
            config,
            run_dir / "benchmark.json",
            max_samples=config.benchmark_max_samples,
        )
        print(f"Benchmark written: {run_dir / 'benchmark.json'}", flush=True)
    else:
        print(
            "Skipping post-train benchmark. Run: "
            f"CHECKPOINT={best_dir} bash scripts/evaluate.sh",
            flush=True,
        )
    return best_dir
