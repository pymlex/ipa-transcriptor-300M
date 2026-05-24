import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm


IPA_REPLACEMENTS = {
    "ɘ": "ə",
    "ɝ": "ɜː",
    "ɚ": "ə",
    "ɨ": "ɪ",
    "ʉ": "uː",
    "ɜː": "ɜː",
    "əʊ": "əʊ",
    "aɪ": "aɪ",
    "ɪə": "ɪə",
    "eɪ": "eɪ",
    "aʊ": "aʊ",
    "ɔɪ": "ɔɪ",
    "ʊə": "ʊə",
}


def normalise_ipa(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("/") and cleaned.endswith("/"):
        cleaned = cleaned[1:-1]
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    for source, target in IPA_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def normalise_word(text: str) -> str:
    return text.strip().lower()


def load_raw_csv(raw_dir: Path) -> pd.DataFrame:
    csv_files = list(raw_dir.glob("*.csv"))
    if len(csv_files) == 0:
        nested = list(raw_dir.rglob("*.csv"))
        csv_files = nested
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV found under {raw_dir}")
    frame = pd.read_csv(csv_files[0])
    return frame


def build_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"word", "phonetic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    subset = frame[["word", "phonetic"]].copy()
    subset["word"] = subset["word"].map(normalise_word)
    subset["phonetic"] = subset["phonetic"].map(normalise_ipa)
    subset = subset[subset["word"].str.len() > 0]
    subset = subset[subset["phonetic"].str.len() > 0]
    subset = subset[~subset["phonetic"].str.contains(r"\?|/-", regex=True)]
    subset = subset.drop_duplicates(subset=["word"], keep="first")
    subset = subset.reset_index(drop=True)
    return subset


def split_frame(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    n_train = int(len(frame) * train_ratio)
    n_val = int(len(frame) * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    val_df = frame.iloc[val_idx].reset_index(drop=True)
    test_df = frame.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def extract_kaggle_zip(raw_dir: Path) -> None:
    zip_files = list(raw_dir.glob("*.zip"))
    if len(zip_files) == 0:
        return
    for archive in tqdm(zip_files, desc="extract"):
        with zipfile.ZipFile(archive, "r") as handle:
            handle.extractall(raw_dir)


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, int]:
    load_dotenv()
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    extract_kaggle_zip(raw_path)
    frame = load_raw_csv(raw_path)
    pairs = build_pairs(frame)
    train_df, val_df, test_df = split_frame(
        pairs,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
    )
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    stats = {
        "total": int(len(pairs)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
    return stats
