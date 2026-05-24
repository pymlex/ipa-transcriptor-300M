import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import create_repo
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def model_size_tag(num_parameters: int) -> str:
    millions = max(1, int(round(num_parameters / 1_000_000)))
    return f"{millions}M"


def push_model_dir(model_dir: str | Path, repo_id: str) -> None:
    load_dotenv()
    token = os.environ["HF_TOKEN"]
    directory = Path(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    num_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    size_tag = model_size_tag(num_parameters)
    if "ipa-transcriptor" in repo_id and not repo_id.split("/")[-1].endswith("M"):
        owner = repo_id.split("/")[0]
        repo_id = f"{owner}/ipa-transcriptor-{size_tag}"
    create_repo(repo_id, exist_ok=True, token=token)
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)
    print(json.dumps({"repo_id": repo_id, "num_parameters": num_parameters}, indent=2))
