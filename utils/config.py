from pathlib import Path

import yaml

from schemas import TrainConfig


def load_config(path: str | Path) -> TrainConfig:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return TrainConfig.model_validate(payload)
