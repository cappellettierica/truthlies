from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset

# data container tht defines standard internal representation of one reasoning example.
@dataclass
class ReasoningExample:
    example_id: str
    question: str
    reference_answer: str
    source_dataset: str

class DatasetLoader:
    def __init__(self, config: dict):
        self.max_examples = config["data"]["max_examples"]

    def load_truthfulqa(self, split: str = "validation") -> List[ReasoningExample]:
        dataset = load_dataset("truthful_qa", "generation", split=split)

        examples: List[ReasoningExample] = []

        for index, row in enumerate(dataset.select(range(min(self.max_examples, len(dataset))))):
            examples.append(
                ReasoningExample(
                    example_id=f"truthfulqa_{index}",
                    question=row["question"],
                    reference_answer=row["best_answer"],
                    source_dataset="truthfulqa",
                )
            )

        return examples

    def load_hotpotqa(self, split: str = "validation") -> List[ReasoningExample]:
        dataset = load_dataset("hotpot_qa", "distractor", split=split)

        examples: List[ReasoningExample] = []

        for index, row in enumerate(dataset.select(range(min(self.max_examples, len(dataset))))):
            examples.append(
                ReasoningExample(
                    example_id=f"hotpotqa_{index}",
                    question=row["question"],
                    reference_answer=row["answer"],
                    source_dataset="hotpotqa",
                )
            )

        return examples

    def load_all(self, config: Dict) -> List[ReasoningExample]:
        truthfulqa = self.load_truthfulqa(config["data"]["truthfulqa_split"])
        hotpotqa = self.load_hotpotqa(config["data"]["hotpotqa_split"])

        return truthfulqa + hotpotqa