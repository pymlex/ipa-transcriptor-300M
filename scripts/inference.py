import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config


def transcribe(
    model_dir: str | Path,
    word: str,
    source_prefix: str,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    source = f"{source_prefix}{word.strip().lower()}"
    encoded = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    ).to(device)
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--word", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ipa = transcribe(
        args.model_dir,
        args.word,
        config.source_prefix,
        config.max_source_length,
        config.max_new_tokens,
        config.generation_num_beams,
        device,
    )
    print(json.dumps({"word": args.word, "ipa": ipa}, ensure_ascii=False))


if __name__ == "__main__":
    main()
