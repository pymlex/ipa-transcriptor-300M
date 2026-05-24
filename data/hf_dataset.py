from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


def load_split_frames(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    return {
        "train": pd.read_csv(root / "train.csv"),
        "validation": pd.read_csv(root / "val.csv"),
        "test": pd.read_csv(root / "test.csv"),
    }


def build_hf_dataset(
    data_dir: str | Path,
    source_prefix: str,
) -> DatasetDict:
    frames = load_split_frames(data_dir)
    payload: dict[str, Dataset] = {}
    for split, frame in frames.items():
        payload[split] = Dataset.from_dict(
            {
                "word": frame["word"].astype(str).tolist(),
                "phonetic": frame["phonetic"].astype(str).tolist(),
                "source": [f"{source_prefix}{word}" for word in frame["word"].astype(str)],
            }
        )
    return DatasetDict(payload)


def preprocess_function(
    examples: dict,
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> dict:
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["phonetic"],
        max_length=max_target_length,
        truncation=True,
    )
    pad_id = tokenizer.pad_token_id
    masked_labels = []
    for sequence in labels["input_ids"]:
        masked_labels.append(
            [token if token != pad_id else -100 for token in sequence]
        )
    model_inputs["labels"] = masked_labels
    return model_inputs
