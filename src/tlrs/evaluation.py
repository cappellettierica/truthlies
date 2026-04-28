import re
from dataclasses import dataclass
from typing import Dict
from rapidfuzz.fuzz import token_set_ratio

from tlrs.utils import safe_str


@dataclass
class EvaluationResult:
    """
    Evaluation scores for one model answer.
    """
    exact_match: float
    contains_reference: float
    fuzzy_match: float
    contradiction_marker: float
    reasoning_length: int


def normalize_text(text: str) -> str:
    """
    Normalize text for simple answer comparison.
    """
    text = safe_str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def exact_match(prediction: str, reference: str) -> float:
    """
    Compute exact-match score.
    """
    return float(normalize_text(prediction) == normalize_text(reference))

def fuzzy_match_score(prediction: str, reference: str) -> float:
    """
    Compute fuzzy similarity between prediction and reference.
    Returns value in [0, 1].
    """
    pred = normalize_text(prediction)
    ref = normalize_text(reference)

    if not ref:
        return 0.0

    return token_set_ratio(pred, ref) / 100.0

def contains_reference_answer(prediction: str, reference: str) -> float:
    """
    Check whether the reference answer appears in the prediction.
    """
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)

    if not normalized_reference:
        return 0.0

    return float(normalized_reference in normalized_prediction)


def contradiction_marker_score(prediction: str) -> float:
    """
    Detect whether the model explicitly notices possible contradiction.

    This is a lightweight proxy metric, not a complete logical validator.
    """
    markers = [
        "contradiction",
        "inconsistent",
        "false assumption",
        "misleading",
        "not enough information",
        "unsupported",
        "cannot determine",
    ]

    prediction = normalize_text(prediction)

    return float(any(marker in prediction for marker in markers))


def reasoning_length(prediction: str) -> int:
    """
    Count words as a simple proxy for reasoning verbosity.
    """
    return len(normalize_text(prediction).split())


def evaluate_answer(prediction: str, reference: str) -> EvaluationResult:
    """
    Evaluate a generated answer.

    # Inspired by: L10.3-bias-completion-task.ipynb — output aggregation and comparison.
    """
    return EvaluationResult(
        exact_match=exact_match(prediction, reference),
        contains_reference=contains_reference_answer(prediction, reference),
        fuzzy_match=fuzzy_match_score(prediction, reference),
        contradiction_marker=contradiction_marker_score(prediction),
        reasoning_length=reasoning_length(prediction),
    )


def evaluation_to_dict(result: EvaluationResult) -> Dict[str, float]:
    """
    Convert evaluation result to dictionary.
    """
    return {
        "exact_match": result.exact_match,
        "contains_reference": result.contains_reference,
        "fuzzy_match": result.fuzzy_match,
        "contradiction_marker": result.contradiction_marker,
        "reasoning_length": result.reasoning_length,
    }