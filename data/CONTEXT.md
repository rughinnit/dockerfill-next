# Data Contract

> Folder: `data/`  
> Purpose: Pre-training, fine-tuning, evaluation datasets, and tokenizer artifacts for DockerFill.

## Component Map

- `TC/`: Token Completion task datasets:
  - `train_data_token.pkl`
  - `valid_data_token.pkl`
  - `test_data_token.pkl`
- `LC/`: Line Completion task datasets:
  - `train_data_line.pkl`
  - `valid_data_line.pkl`
  - `test_data_line.pkl`
- `tokenizer.json`: Trained BPE vocabulary and sub-word tokenizer configuration.
- `manual_evaluation_scores.csv`: 254 samples rated across 2 evaluators for similarity and naturalness.
- `llm_prompt.md`: Prompt templates for LLM baselines.

## Explicit Contract

### Immutability
- All pickle files (`*.pkl`) and the tokenizer file (`tokenizer.json`) represent verified benchmark ground truth and are read-only.
