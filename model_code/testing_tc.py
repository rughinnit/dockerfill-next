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
from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM

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
    if not torch.cuda.is_available():
        return
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
        except Exception:
            pass


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_path: Optional[str] = None) -> PreTrainedTokenizerFast:
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
        resolved_path = str(default_p) if default_p.exists() else "tokenizer.json"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=resolved_path)
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'unk_token': '<unk>'})
    tokenizer.add_special_tokens({'mask_token': '<mask>'})
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer


def testing(path: str, model: Optional[RobertaForMaskedLM] = None) -> Dict[str, float]:
    """
    Evaluate Token Completion (TC) accuracy and MRR.
    Loads results from results_{path}.pkl or evaluates on test_dataset with model.
    """
    results_file = f"results_{path}.pkl"
    if not os.path.exists(results_file) and os.path.exists(path):
        results_file = path

    if not os.path.exists(results_file):
        print(f"Results file '{results_file}' not found.")
        return {}

    with open(results_file, 'rb') as f:
        test_dataset = pickle.load(f)

    print(f"Loaded {len(test_dataset)} samples from {results_file}")
    em = 0
    em_10 = 0
    masks = 0
    mrrs = 0.0

    for data in test_dataset:
        if model is not None and 'input_ids' in data:
            input_ids = data['input_ids'].to(device) if isinstance(data['input_ids'], torch.Tensor) else torch.tensor(data['input_ids']).to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            labels = data['labels'][0] if isinstance(data['labels'], (list, tuple)) else data['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            else:
                labels = torch.tensor(labels).to(device)

            with torch.no_grad():
                output = model(input_ids=input_ids).logits

            mask_indices = (input_ids == 4)[0].nonzero(as_tuple=True)[0]
            masks += len(mask_indices)

            predicted_token_id = output[0, mask_indices].argmax(axis=-1)
            for j in range(len(mask_indices)):
                if predicted_token_id[j] == labels[mask_indices[j]]:
                    em += 1

            _, idx1 = torch.sort(output[0, mask_indices], descending=True)
            predicted_token_id_10 = idx1[:, :10]
            for x in range(len(mask_indices)):
                for y in range(min(10, predicted_token_id_10.shape[1])):
                    if labels[mask_indices[x]] == predicted_token_id_10[x, y]:
                        em_10 += 1
                        mrrs += float(1 / (y + 1))
                        break
        elif 'predictions' in data and 'labels' in data:
            # Pre-computed evaluation record
            preds = np.array(data['predictions'])
            labels = np.array(data['labels'])
            num_masks = len(labels)
            masks += num_masks
            if preds.ndim == 2:
                for x in range(num_masks):
                    if labels[x] == preds[x, 0]:
                        em += 1
                    for y in range(min(10, preds.shape[1])):
                        if labels[x] == preds[x, y]:
                            em_10 += 1
                            mrrs += float(1 / (y + 1))
                            break
            elif preds.ndim == 1:
                for x in range(num_masks):
                    if labels[x] == preds[x]:
                        em += 1
                        em_10 += 1
                        mrrs += 1.0

    metrics = {
        'em': em,
        'masks': masks,
        'Acc-1': float(em / masks) if masks > 0 else 0.0,
        'Acc-10': float(em_10 / masks) if masks > 0 else 0.0,
        'MRR': float(mrrs / masks) if masks > 0 else 0.0,
    }
    print("Evaluation metrics:", metrics)
    return metrics


if __name__ == '__main__':
    set_seed(123)
    target_path = "no-pretraining"
    if os.path.exists(f"results_{target_path}.pkl"):
        print(f"Testing TC on results_{target_path}.pkl:")
        testing(target_path)
    else:
        print(f"Target results_{target_path}.pkl not found; ready for evaluation when results are generated.")
