from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tlrs.utils import get_device


@dataclass
class ModelOutput: # standard model output container
    text: str # model output 


class CausalLanguageModel: # wrapper for gpt-style causal models that generate text step by step
# 5.4 decoder-only next-token prediction setup 7.0 practical local model interaction
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 120,
        temperature: float = 0.0,
        do_sample: bool = False,
    ):
        self.model_name = model_name
        self.device = get_device(device) # choose cpu/gpu automatically if requested
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature # controls randomness when sampling
        self.do_sample = do_sample  # false means deterministic generation

        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # lad matching tokenizer
        # tokenizer converts text into token IDs before model reads it
        # converts token IDs back into text after generation

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # use eos token as padding if missing

        self.model = AutoModelForCausalLM.from_pretrained(model_name) # load causal lm from hugging face
        self.model.to(self.device) # move it to device
        self.model.eval() # model in evaluation mode because i use it for inference, not training 

    def generate(self, prompt: str) -> ModelOutput:  # generate output ftom prompt
    # 5.4 tokenization, generation, logits/output inspection
    # 7.1 generation control with temperature 
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt", # return pytorch tensors 
            padding=True,
            truncation=True,
        ).to(self.device) # move tokenized inputs to same device as model

        with torch.no_grad(): # not training the model
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
        ) #  convert generated token ids back to text

        answer_only = generated_text[len(prompt):].strip() # remove original prompt, keep only answer

        stop_markers = [
                            "\nQuestion:",
                            "\n\nQuestion:",
                            "\nQ:",
                            "\nA:",
                            "\nAnswer:",
                            "\nCheck:",
                            "\nBased on",
                            "\nRemember",
                            "\nHowever",
                            "\nMoreover",
                        ] # markers used to cut off extra generated continuation that usually don't make sense

        for marker in stop_markers:
            if marker in answer_only:
                answer_only = answer_only.split(marker)[0].strip() # keep only text before the first stop marker

        return ModelOutput(text=answer_only)
    
    def inspect_next_token_probabilities( # 5.4 probability/logit inspection
        self,
        prompt: str,
        top_k: int = 10,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1, :]  # raw scores for the next token
        probabilities = torch.softmax(logits, dim=-1) # score to probability 

        top_probs, top_indices = torch.topk(probabilities, k=top_k) # get most likely next tokens

        tokens = [
            self.tokenizer.decode(index.item())
            for index in top_indices[0]
        ] # decode token ids into readable strings

        return [
            (token, float(prob))
            for token, prob in zip(tokens, top_probs[0].cpu())
        ]  # return token-probability pairs