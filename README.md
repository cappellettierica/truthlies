# Truth, Lies, and Reasoning Machines

This project studies how Large Language Models's reasoning reliability changes under truth distortion.

## Research Question

How does controlled misinformation affect the factual accuracy and logical consistency of LLM-generated reasoning, and can self-verification prompts reduce reasoning failures?

## Experimental Conditions

1. Baseline: factual prompt
2. Noisy: prompt with contradictory or misleading context
3. Adversarial: prompt explicitly pushes the model toward a false answer
4. Self-check: model answers, then verifies assumptions

## Datasets

- TruthfulQA
- HotpotQA

## Repository Structure

```text
truth-lies-reasoning-machines/
├── config.yaml
├── scripts/
├── src/
├── notebooks/
└── outputs/