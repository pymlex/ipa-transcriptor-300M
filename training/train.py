import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import (
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
        record = dict(logs)
        record["step"] = int(state.global_step)
        record["epoch"] = float(state.epoch) if state.epoch is not None else 0.0
        self.logger.log(record)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_training(config: TrainConfig, run_dir: Path) -> Path:
    set_seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_logger = MetricsLogger(run_dir)
    dataset = build_hf_dataset(config.data_dir, config.source_prefix)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
    tokenised = dataset.map(
        lambda batch: preprocess_function(
            batch,
            tokenizer,
            config.max_source_length,
            config.max_target_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    use_fp16 = config.fp16 and torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        label_smoothing_factor=config.label_smoothing,
        logging_steps=config.logging_steps,
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
        dataloader_num_workers=config.num_workers,
        report_to=[],
        seed=config.seed,
        logging_dir=str(run_dir / "tb"),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[MetricsLoggerCallback(metrics_logger)],
    )
    trainer.train()
    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name_or_path": config.model_name_or_path,
        "source_prefix": config.source_prefix,
        "num_parameters": count_parameters(model),
        "config": config.model_dump(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    run_benchmark(
        str(best_dir),
        config,
        run_dir / "benchmark.json",
    )
    return best_dir
