import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "pymlex/ipa-transcriptor-300M"


def build_model_card() -> str:
    return """---
license: gpl-3.0
library_name: transformers
tags:
- phonetics
- ipa
- byt5
- seq2seq
pipeline_tag: text-generation
base_model: google/byt5-small
---

# IPA Transcriptor 300M

Fine-tuned [`google/byt5-small`](https://huggingface.co/google/byt5-small) for English word to IPA transcription.

Task format:

```text
ipa: analytical  ->  ˌænəˈlɪtɪkəl
```

Training data: [English phonetic and syllable count dictionary](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary), 125,925 word–IPA pairs after cleaning.

GitHub: [pymlex/ipa-transcriptor-300M](https://github.com/pymlex/ipa-transcriptor-300M)

## Benchmark

Fine-tuned on NVIDIA L4, run `colab_l4_bf16`, beam search `num_beams=4`.

| Metric | Validation | Test |
| ------ | ----------: | ---: |
| `n_samples` | 6296 | 6297 |
| `loss` | 0.1515 | 0.1478 |
| `perplexity` | 1.1636 | 1.1592 |
| `token_accuracy` | 0.7849 | 0.7858 |
| `exact_match` | 0.5982 | 0.6111 |
| `char_accuracy` | 0.8948 | 0.8969 |
| `cer` | 0.1066 | 0.1045 |
| `bleu` | 59.82 | 61.11 |

## Dataset length distributions

![Word length](docs/word_length_hist.png)

![IPA length](docs/ipa_length_hist.png)

## Training loss

![Loss linear](docs/colab_l4_bf16_loss.png)

![Loss log scale](docs/colab_l4_bf16_loss_log.png)

## Inference

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_id = "pymlex/ipa-transcriptor-300M"
source_prefix = "ipa: "

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def transcribe(word: str, num_beams: int = 4) -> str:
    source = f"{source_prefix}{word.strip().lower()}"
    encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=36).to(device)
    output_ids = model.generate(**encoded, max_new_tokens=56, num_beams=num_beams, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(transcribe("analytical"))
```
"""


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    docs_dir = ROOT / "docs"
    png_files = sorted(docs_dir.glob("*.png"))
    for png_path in png_files:
        hub_path = f"docs/{png_path.name}"
        api.upload_file(
            path_or_fileobj=str(png_path),
            path_in_repo=hub_path,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload {hub_path}",
        )
        print(f"uploaded {hub_path}")
    api.upload_file(
        path_or_fileobj=build_model_card().encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update model card with metrics tables and training plots",
    )
    print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
