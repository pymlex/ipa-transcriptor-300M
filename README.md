# IPA Transcriptor

## Overview

This project fine-tunes a Hugging Face seq2seq model for English word to IPA transcription. The base checkpoint is `google/byt5-small`, trained with `transformers.Seq2SeqTrainer` on the Kaggle dataset [English phonetic and syllable count dictionary](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary). The task format is:

```text
ipa: analytical  ->  ˌænəˈlɪtɪkəl
```

## Dataset

Source: [schwartstack/english-phonetic-and-syllable-count-dictionary](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary).

Each row contains:

- `word`
- `phonetic`
- `part_of_speech`
- `syllable_len`
- `stress_pos`
- `stress_syllable`

For training we keep only `word` and `phonetic`. Slashes and square brackets are stripped from IPA strings. Incomplete API artefacts such as `/-` and `?` are removed. Words are lowercased and deduplicated by `word`.

The split is `90/5/5` with a fixed seed. Processed CSV files are written to `data/processed/`.

## Model

Base model: [`google/byt5-small`](https://huggingface.co/google/byt5-small).

IPA strings are mostly single Unicode symbols. Word-piece tokenizers such as T5 SentencePiece can merge several IPA characters into one subword, which hurts edit-level control. ByT5 tokenises at the UTF-8 byte level, so generation steps are much finer than whole words and usually shorter than typical BPE spans over IPA.

Default configuration in `configs/default.yaml`:

- `model_name_or_path`: `google/byt5-small`
- `source_prefix`: `ipa: `
- `max_source_length`: 64
- `max_target_length`: 128

Training settings:

- GPU: NVIDIA L4, Colab Pro
- `per_device_train_batch_size`: 64
- `gradient_accumulation_steps`: 2
- epochs: 10
- learning rate: `5e-5`
- scheduler: warmup ratio `0.06`
- optimiser: AdamW via `Seq2SeqTrainer`
- mixed precision: `fp16` when CUDA is available
- best checkpoint: lowest `eval_loss`

The fine-tuned weights are saved to `runs/<run_name>/best/` in Hugging Face format.

## Metrics

Teacher-forcing loss from `transformers`:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta\bigl(y_i \mid y_{1:i-1}, x\bigr)
$$

Perplexity:

$$
\mathrm{PPL} = \exp(\mathcal{L})
$$

Exact match:

$$
\mathrm{EM} = \frac{1}{M}\sum_{j=1}^{M} \mathbb{I}(\hat{y}_j = y_j)
$$

Character error rate:

$$
\mathrm{CER} = \frac{\sum_j \mathrm{Levenshtein}(\hat{y}_j, y_j)}{\sum_j |y_j|}
$$

Decoding uses beam search with `num_beams=4`. Corpus BLEU is computed with SacreBLEU.

## Installation

```bash
git clone https://github.com/pymlex/ipa-transcriptor.git
cd ipa-transcriptor
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

Required keys in `.env`:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `HF_TOKEN`
- `HF_REPO_ID` optional, final Hub id becomes `pymlex/ipa-transcriptor-{N}M`

## Data download and preparation

```bash
export PYTHONPATH=.
python main.py download --raw-dir data/raw
python main.py prepare --raw-dir data/raw
```

Shell wrappers:

```bash
bash scripts/download.sh
bash scripts/prepare.sh
```

## Training

```bash
export PYTHONPATH=.
python main.py train --run-name l4_run1
```

Or:

```bash
RUN_NAME=l4_run1 bash scripts/train.sh
```

Per-run artefacts under `runs/<run_name>/`:

- `metrics.jsonl`
- `metrics.csv`
- `run_meta.json`
- `benchmark.json`
- `best/` Hugging Face checkpoint

Intermediate `checkpoint-*` folders are also stored under the same run directory by `Seq2SeqTrainer`.

## Evaluation

```bash
export PYTHONPATH=.
python main.py evaluate \
  --checkpoint runs/l4_run1/best \
  --output runs/l4_run1/benchmark_manual.json
```

Or:

```bash
CHECKPOINT=runs/l4_run1/best \
OUTPUT=runs/l4_run1/benchmark_manual.json \
bash scripts/evaluate.sh
```

## Inference

```bash
export PYTHONPATH=.
python scripts/inference.py \
  --model-dir runs/l4_run1/best \
  --word analytical
```

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
    encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=32).to(device)
    output_ids = model.generate(**encoded, max_new_tokens=64, num_beams=num_beams, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(transcribe("analytical"))
```

## Push to Hugging Face

```bash
export PYTHONPATH=.
export CHECKPOINT=runs/l4_run1/best
export HF_REPO_ID=pymlex/ipa-transcriptor
python main.py push --checkpoint "$CHECKPOINT"
```

The script uploads tokenizer and model with `push_to_hub` and names the repository `pymlex/ipa-transcriptor-{N}M` from the parameter count.

```bash
CHECKPOINT=runs/l4_run1/best bash scripts/push_hf.sh
```

## Colab workflow

Open `notebooks/colab.ipynb`. The notebook calls shell scripts for clone, install, download, prepare, train, evaluate, Hub upload, and git push.

## Project layout

```text
ipa-transcriptor/
├── configs/default.yaml
├── data/prepare.py
├── data/hf_dataset.py
├── training/train.py
├── evaluation/metrics.py
├── evaluation/benchmark.py
├── scripts/
├── notebooks/colab.ipynb
├── main.py
├── schemas.py
└── utils/
```

## Licence

GPL-3.0. See `LICENSE`.
