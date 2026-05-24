import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data.hf_dataset import load_split_frames
from evaluation.metrics import (
    character_error_rate,
    char_accuracy,
    corpus_bleu,
    exact_match,
    perplexity_from_loss,
    token_accuracy,
)
from schemas import BenchmarkResult, TrainConfig


class IpaBenchmarkSet(torch.utils.data.Dataset):
    def __init__(self, frame, source_prefix: str) -> None:
        self.sources = [f"{source_prefix}{word}" for word in frame["word"].astype(str)]
        self.references = frame["phonetic"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "source": self.sources[index],
            "phonetic": self.references[index],
        }


def collate_sources(batch: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "source": [item["source"] for item in batch],
        "phonetic": [item["phonetic"] for item in batch],
    }


@torch.no_grad()
def predict_batch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sources: list[str],
    config: TrainConfig,
    device: torch.device,
) -> list[str]:
    encoded = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_source_length,
    ).to(device)
    generated = model.generate(
        **encoded,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.generation_num_beams,
        early_stopping=True,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


@torch.no_grad()
def eval_loss_batch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sources: list[str],
    references: list[str],
    config: TrainConfig,
    device: torch.device,
) -> float:
    encoded = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_source_length,
    ).to(device)
    labels = tokenizer(
        text_target=references,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_target_length,
    ).input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    outputs = model(**encoded, labels=labels)
    return float(outputs.loss.item())


def evaluate_split(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    frame,
    config: TrainConfig,
    device: torch.device,
) -> tuple[float, list[str], list[str]]:
    dataset = IpaBenchmarkSet(frame, config.source_prefix)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sources,
    )
    losses: list[float] = []
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    for batch in tqdm(loader, desc="benchmark", leave=False):
        batch_loss = eval_loss_batch(
            model,
            tokenizer,
            batch["source"],
            batch["phonetic"],
            config,
            device,
        )
        losses.append(batch_loss)
        batch_preds = predict_batch(
            model,
            tokenizer,
            batch["source"],
            config,
            device,
        )
        predictions.extend(batch_preds)
        references.extend(batch["phonetic"])
    return float(np.mean(losses)), predictions, references


def run_benchmark(
    model_dir: str | Path,
    config: TrainConfig,
    output_path: str | Path,
) -> dict[str, BenchmarkResult]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    frames = load_split_frames(config.data_dir)
    split_frames = {"val": frames["validation"], "test": frames["test"]}
    results: dict[str, BenchmarkResult] = {}
    for split_name, frame in split_frames.items():
        loss, predictions, references = evaluate_split(
            model,
            tokenizer,
            frame,
            config,
            device,
        )
        record = BenchmarkResult(
            split=split_name,
            n_samples=len(frame),
            loss=loss,
            perplexity=perplexity_from_loss(loss),
            token_accuracy=token_accuracy(predictions, references),
            exact_match=exact_match(predictions, references),
            char_accuracy=char_accuracy(predictions, references),
            cer=character_error_rate(predictions, references),
            bleu=corpus_bleu(predictions, references),
        )
        results[split_name] = record
    payload = {name: value.model_dump() for name, value in results.items()}
    Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results
