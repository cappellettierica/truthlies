# Truth, lies, and reasoning machines

The goal is to study whether prompt design can improve outputs when a small LLM model faces truth distortion scenarios under multiple prompting conditions:

- **Baseline**: standard factual prompting  
- **Adversarial**: feeds the model potentially incorrect information  
- **Self-check**: asks the model to verify assumptions before answering  

Experiments are performed on:

- **TruthfulQA**: questions designed to trigger common misconceptions and hallucinations  
- **HotpotQA**: multi-hop reasoning questions requiring information integration
25 questions × 3 promt cnditions = 150 model generations

**Model**: TinyLlama-1.1B-Chat, with deterministic decoding (temperature = 0) for reproducibility

Model outputs are evaluated using four metrics:

- **Fuzzy Match**: semantic similarity between generated and reference answers  
- **Truth Token Probability**: probability assigned to tokens from the reference answer  
- **Contradiction Marker Detection**: whether the model explicitly signals uncertainty or inconsistencies  
- **Reasoning Length**: output length used as a lightweight reasoning proxy


## Results

Average fuzzy-match score:

| Dataset | Baseline | Adversarial | Self-check |
|---|---|---|---|
| TruthfulQA | 0.611 | 0.673 | **0.742** |
| HotpotQA | **0.513** | 0.512 | 0.480 |

Main findings:

- On TruthfulQA, self-check prompting gives the best fuzzy match (0.611 → 0.742) and adversarial prompting also improves on baseline.
- On HotpotQA, extra reliability instructions do not help and slightly lower the score.
- Answers on TruthfulQA become shorter under the modified prompts (33.6 words at baseline vs 17.4 and 19.8).
- The contradiction marker is 0 in all conditions: the model rarely expresses explicit uncertainty, so this metric turned out not to be informative.
- Stronger final answers do not always come with higher next-token probability for the reference answer, so the metrics complement each other rather than measure the same thing.

## Limitations

Small model, only 25 examples per dataset, approximate lexical metrics, truncated HotpotQA context. The results describe behaviour on this sample and should not be generalised to larger models.

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

## References

- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL.
- Yang, Z., Qi, P., Zhang, S., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP.


The notebook serves as a readable end-to-end demonstration of the entire experiment.
