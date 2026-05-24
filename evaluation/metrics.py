import numpy as np
import editdistance
from sacrebleu.metrics import BLEU


def perplexity_from_loss(loss_value: float) -> float:
    return float(np.exp(loss_value))


def token_accuracy(
    predictions: list[str],
    references: list[str],
) -> float:
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        length = max(len(pred), len(ref))
        if length == 0:
            continue
        for index in range(length):
            pred_char = pred[index] if index < len(pred) else ""
            ref_char = ref[index] if index < len(ref) else ""
            if pred_char == ref_char:
                correct += 1
            total += 1
    if total == 0:
        return 0.0
    return correct / total


def exact_match(predictions: list[str], references: list[str]) -> float:
    matches = sum(pred == ref for pred, ref in zip(predictions, references))
    return matches / len(references)


def char_accuracy(predictions: list[str], references: list[str]) -> float:
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        distance = editdistance.eval(pred, ref)
        length = max(len(pred), len(ref), 1)
        correct += length - distance
        total += length
    return correct / total


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    distance = 0
    total = 0
    for pred, ref in zip(predictions, references):
        distance += editdistance.eval(pred, ref)
        total += max(len(ref), 1)
    return distance / total


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    metric = BLEU(effective_order=True)
    score = metric.corpus_score(predictions, [references])
    return float(score.score)
