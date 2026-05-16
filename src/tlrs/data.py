from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset


# data container that defines standard internal representation of one reasoning example.
@dataclass
class ReasoningExample:
    example_id: str
    question: str
    reference_answer: str
    source_dataset: str
    context: str = ""

class DatasetLoader:
    def __init__(self, config: dict):
        self.max_examples = config["data"]["max_examples"]

    def load_truthfulqa(self, split: str = "validation") -> List[ReasoningExample]:
        dataset = load_dataset(
            "truthful_qa",
            "generation",
            split=split
        )

        examples: List[ReasoningExample] = []
        n = min(self.max_examples, len(dataset))

        for index, row in enumerate(dataset.select(range(n))):
            examples.append(
                ReasoningExample(
                    example_id=f"truthfulqa_{index}",
                    question=row["question"],
                    reference_answer=row["best_answer"],
                    source_dataset="truthfulqa",
                    context=""  # TruthfulQA has no evidence context
                )
            )

        return examples

    def load_hotpotqa(self, split: str = "validation") -> List[ReasoningExample]:
        dataset = load_dataset(
            "hotpot_qa",
            "distractor",
            split=split
        )

        examples: List[ReasoningExample] = []
        n = min(self.max_examples, len(dataset))

        for index, row in enumerate(dataset.select(range(n))):

            context_parts = []

            for title, sentences in zip(
                row["context"]["title"],
                row["context"]["sentences"]
            ):
                paragraph = " ".join(sentences)
                context_parts.append(
                    f"{title}: {paragraph}"
                )

            context_text = "\n\n".join(context_parts)

            examples.append(
                ReasoningExample(
                    example_id=f"hotpotqa_{index}",
                    question=row["question"],
                    reference_answer=row["answer"],
                    source_dataset="hotpotqa",
                    context=context_text
                )
            )

        return examples

    def load_all(self, config: Dict) -> List[ReasoningExample]:
        truthfulqa = self.load_truthfulqa(
            config["data"]["truthfulqa_split"]
        )

        hotpotqa = self.load_hotpotqa(
            config["data"]["hotpotqa_split"]
        )

        return truthfulqa + hotpotqa