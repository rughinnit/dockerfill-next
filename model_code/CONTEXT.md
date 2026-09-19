# Model Code Contract

> Folder: `model_code/`  
> Purpose: Multi-Task Learning (MTL) model pre-training, fine-tuning, and task evaluation for DockerFill.

## Component Map

- `config_codesage.py`: PretrainedConfig definition for CodeSage backbone.
- `modeling_codesage.py`: CodeSage architecture (Attention, MLP, Blocks, MaskedLM head).
- `soft_sharing.py`: Soft parameter sharing multi-task models (`RobertaForMLM`, `RobertaForSTI`, `RobertaForMIP`, `MultitaskModel`).
- `hard_sharing.py`: Hard parameter sharing multi-task model (`RobertaForMaskedLM`).
- `multitask_data_collator.py`: Multi-task sampling dataloader (`MultitaskTrainer`, `MultitaskDataloader`, `NLPDataCollator`).
- `model.py`: Training pipeline entrypoint for RoBERTa/CodeSage models.
- `testing_tc.py`: Evaluation routine for Token Completion (Acc-1, Acc-10, MRR).
- `testing_lc.py`: Evaluation routine for Line Completion (EM, ED, BLEU, ROUGE-L, METEOR).
- `tokenizer.py`: BPE tokenizer trainer and SQL dataset extraction.

## Explicit Contract

### Inputs
- Datasets: `data/TC/train_data_token.pkl`, `data/TC/valid_data_token.pkl`, `data/LC/*.pkl`.
- Tokenizer: `data/tokenizer.json`.
- Optional checkpoints: `pretraining/checkpoint/`.

### Outputs
- Checkpoints: `./model/checkpoint-*`.
- Predictions: `results.pkl`, `line_results.pkl`.

### Human Verification Gate
- Ensure CUDA GPU availability or allow automatic CPU fallback.
- Verify loss decrease across epochs before promoting checkpoints to downstream evaluation.
