from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tlrs.utils import get_device


@dataclass
class ModelOutput:
    """
    Standard model output container.
    """
    text: str


class CausalLanguageModel:
    """
    Wrapper around a decoder-only Hugging Face causal language model.

    This is appropriate for the project because GPT-style models generate
    reasoning step by step.

    # Inspired by: L5.4-gpt.ipynb — decoder-only next-token prediction setup.
    # Inspired by: L7.0-practical-llms.ipynb — practical local model interaction.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 120,
        temperature: float = 0.0,
        do_sample: bool = False,
    ):
        self.model_name = model_name
        self.device = get_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> ModelOutput:
        """
        Generate model output from a prompt.

        # Inspired by: L5.4-gpt.ipynb — tokenization, generation, logits/output inspection.
        # Inspired by: L7.1-llama-cpp-example.ipynb — generation control with temperature.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        answer_only = generated_text[len(prompt):].strip()

        stop_markers = ["\nQuestion:", "\n\nQuestion:", "\nAnswer:", "\nCheck:"]

        for marker in stop_markers:
            if marker in answer_only:
                answer_only = answer_only.split(marker)[0].strip()

        return ModelOutput(text=answer_only)

    def inspect_next_token_probabilities(
        self,
        prompt: str,
        top_k: int = 10,
    ):
        """
        Inspect next-token probabilities.

        # Inspired by: L5.4-gpt.ipynb — probability/logit inspection.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        tokens = [
            self.tokenizer.decode(index.item())
            for index in top_indices[0]
        ]

        return list(zip(tokens, top_probs[0].float().cpu().numpy()))