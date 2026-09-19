import os
import gc
import pickle
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch import nn

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    balanced_accuracy_score,
)
from transformers import (
    RobertaForMaskedLM,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
)

try:
    from modeling_codesage import CodeSageForMaskedLM
except ImportError:
    from .modeling_codesage import CodeSageForMaskedLM

try:
    from GPUtil import showUtilization as gpu_usage
except ImportError:
    gpu_usage = None

try:
    from numba import cuda
except ImportError:
    cuda = None

os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def free_gpu_cache() -> None:
    """Safely clear GPU cache across PyTorch and numba if available."""
    if not torch.cuda.is_available():
        return
    print("Initial GPU Usage:")
    if gpu_usage is not None:
        try:
            gpu_usage()
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    if cuda is not None:
        try:
            cuda.select_device(0)
            cuda.close()
            cuda.select_device(0)
        except Exception as e:
            logging.warning(f"Could not reset numba cuda device: {e}")
    print("GPU Usage after emptying the cache:")
    if gpu_usage is not None:
        try:
            gpu_usage()
        except Exception:
            pass


def set_seed(seed: int = 123) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_path: Optional[str] = None) -> PreTrainedTokenizerFast:
    """Load and configure the Byte-Pair Encoding tokenizer."""
    candidate_paths = [
        tokenizer_path,
        "data/tokenizer.json",
        "../data/tokenizer.json",
        "tokenizer.json",
    ]
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        default_p = base_dir / "data" / "tokenizer.json"
        if default_p.exists():
            resolved_path = str(default_p)
        else:
            resolved_path = "tokenizer.json"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=resolved_path)
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'unk_token': '<unk>'})
    tokenizer.add_special_tokens({'mask_token': '<mask>'})
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def load_dataset_file(file_name: str) -> Any:
    """Locate and load a pickle dataset file from candidate directories."""
    candidates = [
        file_name,
        os.path.join("data", "TC", file_name),
        os.path.join("..", "data", "TC", file_name),
        os.path.join("data", file_name),
        os.path.join("..", "data", file_name),
    ]
    base_dir = Path(__file__).resolve().parent.parent
    candidates.append(str(base_dir / "data" / "TC" / file_name))
    candidates.append(str(base_dir / "data" / file_name))

    for c in candidates:
        if os.path.exists(c):
            with open(c, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not locate dataset file: {file_name}")


def validation(model_checkpoint: str, valid_data: List[Dict[str, torch.Tensor]], tokenizer: PreTrainedTokenizerFast, output_file: str = "results.pkl") -> None:
    """Run validation evaluation on trained checkpoint."""
    model = RobertaForMaskedLM.from_pretrained(model_checkpoint)
    model.to(device)

    results = []
    no = 0
    for data in valid_data:
        no += 1
        input_ids = data['input_ids'].to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        labels = data['labels'].to(device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        labels = labels[0]
        model_input = {'input_ids': input_ids}
        with torch.no_grad():
            output = model(**model_input).logits
        mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 4
        mask_token_index = (input_ids == mask_id)[0].nonzero(as_tuple=True)[0]

        a, idx1 = torch.sort(output[0, mask_token_index], descending=True)
        softmax_probs = torch.softmax(a, dim=1).cpu().numpy()
        predicted_token_id_5 = idx1[:, :5]
        predicted_token_ratio_5 = softmax_probs[:, :5]
        ratios = []
        for items in predicted_token_ratio_5:
            temp = [round(float(item), 5) for item in items]
            ratios.append(temp)

        results.append({
            'id': no,
            'input': data['input_ids'].cpu().numpy(),
            'labels': labels[mask_token_index].cpu().numpy(),
            'predictions': predicted_token_id_5.cpu().numpy(),
            'ratios': ratios,
        })

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Validation completed. Results saved to {output_file}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train DockerFill model")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    args = parser.parse_args()

    set_seed(123)
    free_gpu_cache()

    use_cuda = torch.cuda.is_available() and not args.cpu
    train_device = torch.device("cuda:0" if use_cuda else "cpu")

    # Auto-adjust batch size and gradient accumulation based on GPU VRAM
    grad_accum = 1
    grad_checkpointing = False
    optimizer_name = "adamw_torch"

    if use_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU with {vram_gb:.2f} GB VRAM.")
        if vram_gb < 4.0:
            batch_size = args.batch_size or 1
            grad_accum = 16 // batch_size
            grad_checkpointing = True
            optimizer_name = "adafactor"
            print(f"Low VRAM mode: setting batch_size={batch_size}, gradient_accumulation_steps={grad_accum}, gradient_checkpointing=True, optim=adafactor")
        elif vram_gb < 8.0:
            batch_size = args.batch_size or 4
            grad_accum = 16 // batch_size
            grad_checkpointing = True
        else:
            batch_size = args.batch_size or 16
    else:
        batch_size = args.batch_size or 4
        print(f"Running on CPU: setting batch_size={batch_size}")

    tokenizer = get_tokenizer()

    # Model checkpoint or initialization
    model_path = "pretraining/checkpoint" if os.path.exists("pretraining/checkpoint") else "roberta-base"
    try:
        model = RobertaForMaskedLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Loading pretrained weights failed ({e}), initializing standard RobertaForMaskedLM...")
        from transformers import RobertaConfig
        config = RobertaConfig(vocab_size=len(tokenizer), max_position_embeddings=514)
        model = RobertaForMaskedLM(config)

    model.to(train_device)
    print(f"Model parameters: {model.num_parameters()}")

    # Try loading dataset
    train_file = "train_data_tc_task.pkl" if os.path.exists("train_data_tc_task.pkl") else "train_data_token.pkl"
    valid_file = "valid_data_tc_task.pkl" if os.path.exists("valid_data_tc_task.pkl") else "valid_data_token.pkl"

    try:
        train_dataset = load_dataset_file(train_file)
        valid_dataset = load_dataset_file(valid_file)
    except FileNotFoundError as e:
        print(f"Dataset notice: {e}. Skipping training execution.")
        return

    def format_sample(sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        ids = sample['input_ids']
        if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        input_tensor = torch.tensor(ids, dtype=torch.long)

        lbls = sample['labels']
        if isinstance(lbls, (list, tuple)) and len(lbls) > 0 and isinstance(lbls[0], (list, tuple)):
            lbls = lbls[0]
        labels_tensor = torch.tensor(lbls, dtype=torch.long)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
        attention_mask = (input_tensor != pad_id).long()

        return {
            'input_ids': input_tensor,
            'labels': labels_tensor,
            'attention_mask': attention_mask,
        }

    final_train_data = [format_sample(s) for s in train_dataset]
    final_valid_data = [format_sample(s) for s in valid_dataset]

    print(f"Loaded {len(final_train_data)} train samples and {len(final_valid_data)} validation samples.")

    # Determine evaluation_strategy argument name based on transformers version
    training_kwargs: Dict[str, Any] = {
        "output_dir": "./model/",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 50,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.1,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": grad_checkpointing,
        "seed": 123,
        "save_total_limit": 2,
        "num_train_epochs": args.epochs,
        "optim": optimizer_name,
        "fp16": use_cuda,
        "no_cuda": not use_cuda,
        "report_to": 'none'
    }

    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        training_kwargs["evaluation_strategy"] = "no"
    else:
        training_kwargs["eval_strategy"] = "no"

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_train_data,
    )

    trainer.train()

    if os.path.exists("pretraining/checkpoint"):
        print("Running validation on pretraining/checkpoint...")
        validation("pretraining/checkpoint", final_valid_data, tokenizer)


if __name__ == '__main__':
    main()
