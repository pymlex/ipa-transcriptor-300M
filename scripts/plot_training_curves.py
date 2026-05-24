import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_metrics_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_steps: list[int] = []
    train_loss: list[float] = []
    eval_steps: list[int] = []
    eval_loss: list[float] = []
    with path.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = [float(value) for value in line.strip().split(",")]
            step = int(parts[-1])
            if len(parts) >= 6 or parts[2] > 1.0:
                eval_steps.append(step)
                eval_loss.append(parts[0])
            elif len(parts) == 5 and parts[2] <= 1.0:
                train_steps.append(step)
                train_loss.append(parts[0])
    return (
        np.asarray(train_steps, dtype=np.int64),
        np.asarray(train_loss, dtype=np.float64),
        np.asarray(eval_steps, dtype=np.int64),
        np.asarray(eval_loss, dtype=np.float64),
    )


def plot_curves(
    train_steps: np.ndarray,
    train_loss: np.ndarray,
    eval_steps: np.ndarray,
    eval_loss: np.ndarray,
    output_path: Path,
    log_scale: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(train_steps, train_loss, label="train", color="#2a6f97", linewidth=1.2)
    axis.plot(eval_steps, eval_loss, label="eval", color="#9c6644", linewidth=1.5, marker="o", markersize=4)
    if log_scale:
        axis.set_yscale("log")
        axis.set_ylabel("loss (log scale)")
        title = "Training and validation loss (log scale)"
    else:
        axis.set_ylabel("loss")
        title = "Training and validation loss"
    axis.set_xlabel("step")
    axis.set_title(title)
    axis.grid(alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    metrics_path = ROOT / "runs" / "colab_l4_bf16" / "metrics.csv"
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    train_steps, train_loss, eval_steps, eval_loss = parse_metrics_csv(metrics_path)
    plot_curves(
        train_steps,
        train_loss,
        eval_steps,
        eval_loss,
        docs_dir / "colab_l4_bf16_loss.png",
        log_scale=False,
    )
    plot_curves(
        train_steps,
        train_loss,
        eval_steps,
        eval_loss,
        docs_dir / "colab_l4_bf16_loss_log.png",
        log_scale=True,
    )
    summary = {
        "train_points": int(train_steps.size),
        "eval_points": int(eval_steps.size),
        "final_train_loss": float(train_loss[-1]),
        "final_eval_loss": float(eval_loss[-1]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
