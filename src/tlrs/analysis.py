from typing import Iterable, List, Tuple

def normalize_token(token: str) -> str: # clean tokenizer symbols before comparing tokens
    return (
        token
        .replace("▁", "") # remove token markers 
        .replace("Ġ", "") # remove token markers 
        .strip()
        .lower()
    )

def get_reference_target_tokens(
    reference_answer: str,
    tokenizer,
    max_tokens: int = 5,
) -> List[str]:
   # extract first tokenizer-level tokens from the reference answer
    if not reference_answer:
        return [] # failsafe

    tokens = tokenizer.tokenize(reference_answer)  # tokenize using the model tokenizer

    normalized_tokens = [
        normalize_token(token)
        for token in tokens
        if normalize_token(token)
    ]

    return normalized_tokens[:max_tokens]

def extract_target_token_probability(
    # probability the model assigns to target tokens
    top_tokens: List[Tuple[str, float]],
    target_tokens: Iterable[str],
) -> float: 
    # sum probability assigned to reference target tokens
    normalized_targets = {
        normalize_token(token)
        for token in target_tokens
        if token and normalize_token(token)
    } # normalize target tokens once for comparison

    probability = 0.0

    for token, prob in top_tokens:
        if normalize_token(token) in normalized_targets:
            probability += float(prob) # add probability if predicted token matches target token

    return probability