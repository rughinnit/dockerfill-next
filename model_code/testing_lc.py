import os
import gc
import pickle
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from torchmetrics.text import EditDistance, BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score

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


def testing(path: str = "line_results.pkl", tokenizer: Optional[PreTrainedTokenizerFast] = None) -> Dict[str, float]:
    """
    Evaluate Line Completion (LC) metrics: EM, ED, BLEU, ROUGE-L, and METEOR.
    """
    candidates = [
        path,
        f"{path}.pkl",
        f"line_results_{path}.pkl",
        "line_results.pkl",
    ]
    resolved_file = None
    for c in candidates:
        if os.path.exists(c):
            resolved_file = c
            break

    if resolved_file is None:
        print(f"Results file for path '{path}' not found.")
        return {}

    if tokenizer is None:
        tokenizer = get_tokenizer()

    with open(resolved_file, 'rb') as f:
        test_dataset = pickle.load(f)

    number = len(test_dataset)
    if number == 0:
        print("Dataset is empty.")
        return {}

    masks = 0
    perfect = 0
    edits = 0.0
    bleu = 0.0
    rouge = 0.0
    meteor_value = 0.0

    ed_metric = EditDistance()
    bleu_metric = BLEUScore()
    rouge_metric = ROUGEScore()

    for no, data in enumerate(test_dataset, start=1):
        predictions = np.array(data['predictions'])
        labels = data['labels']

        masks += len(labels)

        if np.array_equal(predictions, labels):
            perfect += 1

        preds = [tokenizer.decode(int(item)) for item in (predictions[:, 0] if predictions.ndim == 2 else predictions)]
        trues = [tokenizer.decode(int(item)) for item in labels]

        pred_str = ' '.join(preds)
        true_str = ' '.join(trues)
        length = max(len(''.join(preds)), len(''.join(trues)), 1)

        edits += float(ed_metric([pred_str], [true_str]) / length)
        bleu += float(bleu_metric([pred_str], [[true_str]]))
        rouge += float(rouge_metric(pred_str, true_str)['rougeL_fmeasure'])

        # Compute meteor score with tokenized words
        meteor_value += float(meteor_score([trues], preds))

    metrics = {
        'perfect': perfect,
        'masks': masks,
        'BLEU': float(bleu / number),
        'ROUGE-L': float(rouge / number),
        'METEOR': float(meteor_value / number),
        'EM': float(perfect / number),
        'ED': float(edits / number),
    }
    print("LC Evaluation metrics:", metrics)
    return metrics


if __name__ == '__main__':
    set_seed(123)
    target_path = "pretraining"
    if os.path.exists("line_results.pkl") or os.path.exists(f"line_results_{target_path}.pkl"):
        print(f"Testing LC on {target_path}:")
        testing(target_path)
    else:
        print("line_results.pkl not found; ready for evaluation when results are generated.")
