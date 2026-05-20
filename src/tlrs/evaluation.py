import re
from dataclasses import dataclass
from typing import Dict
from rapidfuzz.fuzz import token_set_ratio

from tlrs.utils import safe_str


@dataclass
class EvaluationResult:
    # exact_match: float
    fuzzy_match: float
    contradiction_marker: float
    reasoning_length: int


def normalize_text(text: str) -> str: # normalise text for simple answer comparison
    text = safe_str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) # every character that is not a letter/number -> space
    text = re.sub(r"\s+", " ", text) # multiple spaces/newlines/tabs -> one space

    return text.strip()

# def exact_match(prediction: str, reference: str) -> float: # exact match score
    return float(normalize_text(prediction) == normalize_text(reference))

def fuzzy_match_score(prediction: str, reference: str) -> float: # fuzzy similarity between prediction and reference, [0, 1]
    pred = normalize_text(prediction)
    ref = normalize_text(reference) 

    if not ref: # failsafe
        return 0.0

    return token_set_ratio(pred, ref) / 100.0

def contradiction_marker_score(prediction: str) -> float: # model explicitly notices possible contradiction?
    markers = [
    "contradiction",
    "inconsistent",
    "false",
    "misleading",
    "incorrect",
    "wrong",
    "not enough information",
    "cannot determine",
    "uncertain",
    "depends",
]

    prediction = normalize_text(prediction)

    return float(any(marker in prediction for marker in markers)) # binary yes 1 or no 0


def reasoning_length(prediction: str) -> int: # count words as a simple proxy for reasoning verbosity
    return len(normalize_text(prediction).split())


def evaluate_answer(prediction: str, reference: str) -> EvaluationResult: # returns 1 object containing all metric scores
    # 10.3 output aggregation and comparison
    return EvaluationResult(
        # exact_match=exact_match(prediction, reference),
        fuzzy_match=fuzzy_match_score(prediction, reference),
        contradiction_marker=contradiction_marker_score(prediction),
        reasoning_length=reasoning_length(prediction),
    )


def evaluation_to_dict(result: EvaluationResult) -> Dict[str, float]: # evaluation result to dictionary
    return {
        # "exact_match": result.exact_match,
        "fuzzy_match": result.fuzzy_match,
        "contradiction_marker": result.contradiction_marker,
        "reasoning_length": result.reasoning_length,
    }