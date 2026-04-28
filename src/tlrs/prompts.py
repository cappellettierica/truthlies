from typing import Callable, Dict

from tlrs.data import ReasoningExample


SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Do not invent facts. "
    "If uncertain, say so."
)


def build_baseline_prompt(example: ReasoningExample) -> str:
    """
    Simple factual prompt.

    # Inspired by: L6.0-prompt-engineering.ipynb — minimal instruction prompting.
    """
    return f"""
{SYSTEM_PROMPT}

Answer the following question in one concise sentence.

Question: {example.question}
Answer:
""".strip()


def build_noisy_prompt(example: ReasoningExample) -> str:
    """
    Prompt with misleading context.

    # Inspired by: L10.3-bias-completion-task.ipynb — controlled prompt variation.
    """
    return f"""
{SYSTEM_PROMPT}

There may be misleading or false information related to this question.
Do not assume such information is true.

Question: {example.question}
Answer:
""".strip()


def build_adversarial_prompt(example: ReasoningExample) -> str:
    """
    Adversarial prompt encouraging reconsideration.

    # Inspired by: L6.0-prompt-engineering.ipynb — adversarial instruction design.
    """
    return f"""
{SYSTEM_PROMPT}

A previous answer to this question may be incorrect.
Reconsider carefully, but do not invent facts.

Question: {example.question}
Answer:
""".strip()


def build_self_check_prompt(example: ReasoningExample) -> str:
    """
    Self-check prompt.

    # Inspired by: L6.0-prompt-engineering.ipynb — self-critique prompting.
    """
    return f"""
{SYSTEM_PROMPT}

Answer the question in one concise sentence.
Then briefly check whether your answer depends on any uncertain or unsupported assumption.

Question: {example.question}
Answer:
Check:
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