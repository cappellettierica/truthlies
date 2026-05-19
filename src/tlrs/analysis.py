from typing import Iterable, List, Tuple

def normalize_token(token: str) -> str: # normalize model tokens for comparison
        return (
        token
        .replace("▁", "")
        .replace("Ġ", "")
        .strip()
        .lower()
    )

def get_reference_target_tokens(
    reference_answer: str,
    tokenizer,
    max_tokens: int = 5,
) -> List[str]:
   # extract tokenizer-level target tokens from the reference answer.
    if not reference_answer:
        return []

    tokens = tokenizer.tokenize(reference_answer)

    normalized_tokens = [
        normalize_token(token)
        for token in tokens
        if normalize_token(token)
    ]

    return normalized_tokens[:max_tokens]

def extract_target_token_probability(
    # estimate how much probability the model assigns to truthful target tokens.
    top_tokens: List[Tuple[str, float]],
    target_tokens: Iterable[str],
) -> float: 
    normalized_targets = {
        normalize_token(token)
        for token in target_tokens
        if token and normalize_token(token)
    }

    probability = 0.0

    for token, prob in top_tokens:
        if normalize_token(token) in normalized_targets:
            probability += float(prob)

    return probability