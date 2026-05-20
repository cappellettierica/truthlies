from typing import Dict, List

import pandas as pd
from tqdm import tqdm # progress bar 

from tlrs.data import ReasoningExample
from tlrs.evaluation import evaluate_answer, evaluation_to_dict
from tlrs.models import CausalLanguageModel
from tlrs.prompts import build_prompt

from tlrs.analysis import (
    extract_target_token_probability,
    get_reference_target_tokens,
)


class ReasoningExperiment: # run the baseline/adversarial/self-check experiment
# 10.3 repeated controlled experiments 6.0 comparing prompt conditions
    def __init__(
        self,
        model: CausalLanguageModel,
        examples: List[ReasoningExample],
        conditions: List[str],
    ):
        self.model = model # model wrapper used for generation and token probabilities
        self.examples = examples # prepared truthfulqa and hotpotqa examples
        self.conditions = conditions # prompt conditions to test

    def run(self) -> pd.DataFrame: # run all the examples under all prompt conditions
        rows = [] # stores one result row per example-condition pair so 3 rows * 1 example 

        for example in tqdm(self.examples, desc="Running experiment"):
            for condition in self.conditions:
                prompt = build_prompt(example, condition) # build prompt for current dataset and condition

                top_tokens = self.model.inspect_next_token_probabilities(
                    prompt,
                    top_k=20,
                ) # inspect most likely next tokens before generation

                target_tokens = get_reference_target_tokens(
                    reference_answer=example.reference_answer,
                    tokenizer=self.model.tokenizer,
                ) # extract reference-answer tokens for probability check

                truth_token_probability = extract_target_token_probability(
                    top_tokens=top_tokens,
                    target_tokens=target_tokens,
                ) # sum probability assigned to reference target tokens

                model_output = self.model.generate(prompt) # generate full model answer
                    
                print("\n---")
                print("Condition:", condition)
                print("Question:", example.question)
                print("Output:", model_output.text) # print output for manual inspection
                                                # [:200]
                scores = evaluate_answer(
                    prediction=model_output.text,
                    reference=example.reference_answer,
                ) # compute fuzzy match, contradiction marker, reasoning length

                row: Dict = {
                    "example_id": example.example_id,
                    "source_dataset": example.source_dataset,
                    "condition": condition,
                    "question": example.question,
                    "context": example.context,
                    "reference_answer": example.reference_answer,
                    "model_output": model_output.text,
                    "target_tokens":str(target_tokens),
                    "truth_token_probability": truth_token_probability,
                    "top_next_tokens": str(top_tokens),
                } # store raw output, metadata, token-probability info

                row.update(evaluation_to_dict(scores)) # add evaluation metrics to row
                rows.append(row)

        return pd.DataFrame(rows) # convert all rows to final results dataframe