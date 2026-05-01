from typing import Callable, Dict

from tlrs.data import ReasoningExample


def build_baseline_prompt(example):
    return f"""
Q: {example.question}
A:
""".strip()


def build_noisy_prompt(example):
    return f"""
Some people give misleading answers to this question.

Q: {example.question}
A:
""".strip()


def build_adversarial_prompt(example):
    return f"""
A previous answer to this question may be false.

Q: {example.question}
A:
""".strip()


def build_self_check_prompt(example):
    return f"""
Q: {example.question}

Give your answer.
Then briefly say whether your answer might be uncertain.

Answer:
""".strip()


def build_prompt(example: ReasoningExample, condition: str) -> str:
    """
    Dispatch prompt construction by experimental condition.
    """
    builders: Dict[str, Callable[[ReasoningExample], str]] = {
        "baseline": build_baseline_prompt,
        "noisy": build_noisy_prompt,
        "adversarial": build_adversarial_prompt,
        "self_check": build_self_check_prompt,
    }

    if condition not in builders:
        raise ValueError(f"Unknown condition: {condition}")

    return builders[condition](example)