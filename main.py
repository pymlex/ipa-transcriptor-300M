import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.prepare import prepare_dataset
from evaluation.benchmark import run_benchmark
from schemas import TrainConfig
from training.train import count_parameters, run_training
from utils.config import load_config


def command_download(raw_dir: str) -> None:
    load_dotenv()
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "schwartstack/english-phonetic-and-syllable-count-dictionary",
            "-p",
            str(raw_path),
            "--unzip",
        ],
        check=True,
        env=env,
    )


def command_prepare(config: TrainConfig, raw_dir: str) -> None:
    stats = prepare_dataset(
        raw_dir,
        config.data_dir,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.seed,
    )
    print(json.dumps(stats, indent=2))


def command_train(config: TrainConfig, run_name: str | None) -> None:
    stamp = run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.run_dir) / stamp
    best_dir = run_training(config, run_dir)
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(best_dir)
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_dir),
        "num_parameters": count_parameters(model),
        "model_name_or_path": config.model_name_or_path,
    }
    print(json.dumps(summary, indent=2))


def command_evaluate(config: TrainConfig, checkpoint: str, output: str) -> None:
    results = run_benchmark(checkpoint, config, output)
    print(json.dumps({key: value.model_dump() for key, value in results.items()}, indent=2))


def command_push(checkpoint: str, repo_id: str | None) -> None:
    load_dotenv()
    from scripts.push_hf import push_model_dir

    target_repo = repo_id or os.environ["HF_REPO_ID"]
    push_model_dir(checkpoint, target_repo)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IPA seq2seq transcriptor")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--raw-dir", default="data/raw")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--raw-dir", default="data/raw")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--run-name", default=None)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--output", default="benchmark.json")

    push_parser = subparsers.add_parser("push")
    push_parser.add_argument("--checkpoint", required=True)
    push_parser.add_argument("--repo-id", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    if args.command == "download":
        command_download(args.raw_dir)
    elif args.command == "prepare":
        command_prepare(config, args.raw_dir)
    elif args.command == "train":
        command_train(config, args.run_name)
    elif args.command == "evaluate":
        command_evaluate(config, args.checkpoint, args.output)
    elif args.command == "push":
        command_push(args.checkpoint, args.repo_id)


if __name__ == "__main__":
    main()
