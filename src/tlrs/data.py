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
    context: str = "" # hotpot
    misleading_answer: str = "" # truthful

def truncate_text(text: str, max_chars: int) -> str: # truncate context
    text = text.strip()

    if len(text) <= max_chars: # yaml
        return text

    return text[:max_chars].rsplit(" ", 1)[0].strip() + " [...]"

class DatasetLoader:
    def __init__(self, config: dict):
        self.max_examples = config["data"]["max_examples"]
        self.max_context_chars = config["data"]["max_context_chars"]

    def load_truthfulqa(self, split: str = "validation") -> List[ReasoningExample]:
        dataset = load_dataset(
            "truthful_qa",
            "generation",
            split=split
        )

        examples: List[ReasoningExample] = []
        n = min(self.max_examples, len(dataset)) # failsafe

        for index, row in enumerate(dataset.select(range(n))):
            incorrect_answers = row.get("incorrect_answers", []) 

            misleading_answer = "" 
            if incorrect_answers:
                misleading_answer = incorrect_answers[0] # the first incorrect answer

            examples.append(
                ReasoningExample(
                    example_id=f"truthfulqa_{index}",
                    question=row["question"],
                    reference_answer=row["best_answer"],
                    source_dataset="truthfulqa",
                    context="", # because no context 
                    misleading_answer=misleading_answer,
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
                row["context"]["sentences"],
            ):
                # keep only the first sentences from each paragraph.
                paragraph = " ".join(sentences[:2])
                context_parts.append(f"{title}: {paragraph}")
                # preserves evidence while avoiding massive prompts.

            context_text = "\n\n".join(context_parts)
            context_text = truncate_text( # otherwise runtime > 5h
                context_text,
                max_chars=self.max_context_chars
            )

            examples.append(
                ReasoningExample(
                    example_id=f"hotpotqa_{index}",
                    question=row["question"],
                    reference_answer=row["answer"],
                    source_dataset="hotpotqa",
                    context=context_text,
                    misleading_answer="", # no misleading answer
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