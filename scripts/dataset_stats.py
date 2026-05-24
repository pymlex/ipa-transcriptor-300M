import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.prepare import build_pairs, load_raw_csv
def length_stats(lengths: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }


def plot_histogram(
    lengths: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    colour: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(1, int(lengths.max()) + 2) - 0.5
    axis.hist(lengths, bins=bins, color=colour, edgecolor="white", linewidth=0.6)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    axis.grid(alpha=0.5)
    axis.set_xlim(0.5, int(lengths.max()) + 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    frame = load_raw_csv(raw_dir)
    pairs = build_pairs(frame)
    word_lengths = pairs["word"].str.len().to_numpy(dtype=np.int64)
    ipa_lengths = pairs["phonetic"].str.len().to_numpy(dtype=np.int64)
    word_stats = length_stats(word_lengths)
    ipa_stats = length_stats(ipa_lengths)
    plot_histogram(
        word_lengths,
        "Word length distribution",
        "characters",
        docs_dir / "word_length_hist.png",
        "#2a6f97",
    )
    plot_histogram(
        ipa_lengths,
        "IPA length distribution",
        "characters",
        docs_dir / "ipa_length_hist.png",
        "#9c6644",
    )
    payload = {
        "total_pairs": int(len(pairs)),
        "word": word_stats,
        "phonetic": ipa_stats,
    }
    (docs_dir / "dataset_stats.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
