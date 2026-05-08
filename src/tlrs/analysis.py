from typing import Iterable, List, Tuple

def normalize_token(token: str) -> str:
    return token.strip().lower()


def extract_target_token_probability(
    top_tokens: List[Tuple[str, float]],
    target_words: Iterable[str],
) -> float:
    """
    Estimate how much probability the model assigns to truthful target tokens.
    This is a lightweight interpretability measure based on the top-k next-token distribution.
    """
    normalized_targets = {
        word.strip().lower()
        for word in target_words
        if word and word.strip()
    }

    probability = 0.0

    for token, prob in top_tokens:
        if normalize_token(token) in normalized_targets:
            probability += float(prob)

    return probability


def get_reference_target_words(reference_answer: str) -> List[str]:
    """
    Extract simple candidate target words from the reference answer.
    This works best for short answers such as names, countries, dates, yes/no answers, or single factual entities.
    """
    if not reference_answer:
        return []

    words = reference_answer.replace(".", "").replace(",", "").split()

    return words[:3]