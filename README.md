# Truth, lies, and reasoning machines

The goal is to study whether prompt design can improve outputs when a small LLM model faces truth distortion scenarios under multiple prompting conditions:

- **Baseline**: standard factual prompting  
- **Adversarial**: feeds the model potentially incorrect information  
- **Self-check**: asks the model to verify assumptions before answering  

Experiments are performed on:

- **TruthfulQA**: questions designed to trigger common misconceptions and hallucinations  
- **HotpotQA**: multi-hop reasoning questions requiring information integration  

Model outputs are evaluated using four metrics:

- **Fuzzy Match**: semantic similarity between generated and reference answers  
- **Truth Token Probability**: probability assigned to tokens from the reference answer  
- **Contradiction Marker Detection**: whether the model explicitly signals uncertainty or inconsistencies  
- **Reasoning Length**: output length used as a lightweight reasoning proxy  

## Project Structure

```text
src/tlrs/
├── data.py              # dataset loading and preprocessing
├── prompts.py           # prompt construction
├── models.py            # TinyLlama model wrapper
├── experiment.py        # experiment pipeline
├── evaluation.py        # evaluation metrics
├── analysis.py          # token inspection   
├── visualization.py     # result plotting
├── utils.py             # helper functions
```

## Demo Notebook

The complete experimental workflow is demonstrated in **`demo_pipeline.ipynb`**.

The notebook walks through the full project pipeline step-by-step:

- loading and inspecting TruthfulQA and HotpotQA
- constructing prompt variants
- generating model outputs
- inspecting next-token probabilities
- running the complete experiment
- computing evaluation metrics
- visualizing findings

The notebook serves as a readable end-to-end demonstration of the entire experiment.