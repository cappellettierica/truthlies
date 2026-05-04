from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from tlrs.data import ReasoningExample
from tlrs.evaluation import evaluate_answer, evaluation_to_dict
from tlrs.models import CausalLanguageModel
from tlrs.prompts import build_prompt

from tlrs.analysis import (
    extract_target_token_probability,
    get_reference_target_words,
)


class ReasoningExperiment:
    """
    Runs the full baseline/noisy/adversarial/self-check experiment.

    # Inspired by: L10.3-bias-completion-task.ipynb — repeated controlled experiments.
    # Inspired by: L6.0-prompt-engineering.ipynb — comparing prompt conditions.
    """

    def __init__(
        self,
        model: CausalLanguageModel,
        examples: List[ReasoningExample],
        conditions: List[str],
    ):
        self.model = model
        self.examples = examples
        self.conditions = conditions

    def run(self) -> pd.DataFrame:
        """
        Run all examples under all prompt conditions.
        """
        rows = []

        for example in tqdm(self.examples, desc="Running experiment"):
            for condition in self.conditions:
                prompt = build_prompt(example, condition)

                top_tokens = self.model.inspect_next_token_probabilities(
                    prompt,
                    top_k=20,
                )

                target_words = get_reference_target_words(example.reference_answer)

                truth_token_probability = extract_target_token_probability(
                    top_tokens=top_tokens,
                    target_words=target_words,
                )
                model_output = self.model.generate(prompt)
                    
                print("\n---")
                print("Condition:", condition)
                print("Question:", example.question)
                print("Output:", model_output.text)
                                                # [:200]
                scores = evaluate_answer(
                    prediction=model_output.text,
                    reference=example.reference_answer,
                )

                row: Dict = {
                    "example_id": example.example_id,
                    "source_dataset": example.source_dataset,
                    "condition": condition,
                    "question": example.question,
                    "reference_answer": example.reference_answer,
                    "model_output": model_output.text,
                    "truth_token_probability": truth_token_probability,
                    "top_next_tokens": str(top_tokens),
                }

                row.update(evaluation_to_dict(scores))
                rows.append(row)

        return pd.DataFrame(rows)