import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM
)

try:
    from parser_code.Dockerfile_syntax_parser import parse_dockerfile
except ImportError:
    parse_dockerfile = None

logging.basicConfig(level=logging.WARNING)


def load_model_and_tokenizer(model_choice: str = "auto") -> Tuple[torch.nn.Module, any, torch.device, str]:
    """
    Load the chosen model and tokenizer.
    model_choice can be 'local' / 'auto', 'codebert', or 'starcoder'.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_mode = "masked"

    # Option A: Pre-trained StarCoder (Causal Generative)
    if model_choice in ("starcoder", "star"):
        star_path = "pretrained_models/starcoder" if os.path.exists("pretrained_models/starcoder") else "bigcode/tiny_starcoder_py"
        print(f"Loading Pre-trained StarCoder ('{star_path}')...")
        tokenizer = AutoTokenizer.from_pretrained(star_path)
        model = AutoModelForCausalLM.from_pretrained(star_path)
        model_mode = "causal"

    # Option B: Pre-trained CodeBERT (Masked)
    elif model_choice in ("codebert", "bert"):
        bert_path = "pretrained_models/codebert" if os.path.exists("pretrained_models/codebert") else "microsoft/codebert-base-mlm"
        print(f"Loading Pre-trained CodeBERT ('{bert_path}')...")
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        model = AutoModelForMaskedLM.from_pretrained(bert_path)
        model_mode = "masked"

    # Option C: Local Trained Checkpoint / Fallback
    else:
        tok_path = "data/tokenizer.json" if os.path.exists("data/tokenizer.json") else "../data/tokenizer.json"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        
        checkpoint_path = "roberta-base"
        if os.path.exists("model"):
            if os.path.exists("model/config.json"):
                checkpoint_path = "model"
            else:
                checkpoints = sorted(
                    [d for d in Path("model").glob("checkpoint-*") if (d / "config.json").exists()],
                    key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
                    reverse=True
                )
                if checkpoints:
                    checkpoint_path = str(checkpoints[0])
                    
        print(f"Loading local checkpoint: '{checkpoint_path}'...")
        try:
            model = RobertaForMaskedLM.from_pretrained(checkpoint_path)
        except Exception:
            config = RobertaConfig(vocab_size=len(tokenizer), max_position_embeddings=514)
            model = RobertaForMaskedLM(config)
        model_mode = "masked"

    try:
        model.to(device)
    except Exception:
        device = torch.device("cpu")
        print("GPU VRAM occupied or insufficient. Running inference on CPU...")
        model.to(device)

    model.eval()
    return model, tokenizer, device, model_mode


def get_token_predictions(
    prompt: str,
    model: torch.nn.Module,
    tokenizer: any,
    device: torch.device,
    model_mode: str = "masked",
    top_k: int = 4
) -> List[Tuple[str, float]]:
    """Predict top-k candidates for the cursor position at end of prompt."""
    if not prompt.strip():
        return [("FROM ", 0.50), ("# ", 0.20), ("ARG ", 0.15), ("ENV ", 0.10)]

    if model_mode == "causal":
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
        top_indices = torch.topk(next_token_logits, k=top_k * 2).indices.tolist()
        probs = torch.softmax(next_token_logits, dim=-1)
    else:
        # Masked LM
        mask_token = getattr(tokenizer, "mask_token", None) or "<mask >"
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            mask_id = tokenizer.mask_token_id
            inputs = tokenizer(prompt.rstrip() + " " + mask_token, return_tensors="pt").to(device)
            mask_pos = (inputs.input_ids == mask_id)[0].nonzero(as_tuple=True)[0][0]
        else:
            code_ids = tokenizer.encode(prompt)
            input_ids = [0] + code_ids + [4, 2]
            if len(input_ids) > 512:
                input_ids = [input_ids[0]] + input_ids[-(511):]
            inputs = {"input_ids": torch.tensor([input_ids], device=device)}
            mask_pos = (inputs["input_ids"] == 4)[0].nonzero(as_tuple=True)[0][0]

        with torch.no_grad():
            logits = model(**inputs).logits

        mask_logits = logits[0, mask_pos]
        top_indices = torch.topk(mask_logits, k=top_k * 2).indices.tolist()
        probs = torch.softmax(mask_logits, dim=-1)

    results = []
    seen = set()
    for tok_id in top_indices:
        token_str = tokenizer.decode([tok_id])
        clean = token_str.strip()
        if clean and clean not in seen and clean not in ("<s>", "</s>", "<pad>", "<unk>", "<mask >", "<mask"):
            seen.add(clean)
            prob = probs[tok_id].item()
            results.append((clean, prob))
            if len(results) >= top_k:
                break
    return results


def run_live_keystroke_repl(model_choice: str = "auto"):
    """Live interactive REPL that autocompletes on every single keystroke without pressing Enter."""
    import termios
    import tty

    print("=" * 68)
    print("        🐳 DOCKERFILL LIVE KEYSTROKE AUTOCOMPLETE 🐳")
    print("=" * 68)
    model, tokenizer, device, model_mode = load_model_and_tokenizer(model_choice)
    print(f"Model ({model_choice}) loaded! Type Dockerfile instructions in real-time.")
    print("\nControls:")
    print("  • Type normally     : Real-time predictions update on each keystroke")
    print("  • [Tab]             : Accept #1 suggestion instantly")
    print("  • [Enter]           : Add a newline for next instruction")
    print("  • [Backspace]       : Delete previous character")
    print("  • [Ctrl + C / Esc]  : Exit cleanly")
    print("=" * 68)

    buffer = ""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def render(curr_buf: str, suggestions: List[Tuple[str, float]]):
        lines = curr_buf.split("\n")
        current_line = lines[-1] if lines else ""
        header = "\033[1;36m🐳 Dockerfile >\033[0m "
        
        if suggestions:
            sugg_items = []
            for i, (tok, prob) in enumerate(suggestions):
                if i == 0:
                    sugg_items.append(f"\033[1;32m[Tab: {tok} ({prob*100:.0f}%)]\033[0m")
                else:
                    sugg_items.append(f"\033[90m{tok} ({prob*100:.0f}%)\033[0m")
            sugg_line = "   💡 " + " | ".join(sugg_items)
        else:
            sugg_line = "   💡 \033[90m(typing...)\033[0m"

        sys.stdout.write(f"\r\033[K{header}{current_line}\n\033[K{sugg_line}\033[1A\r\033[{len(current_line) + 16}C")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        suggs = get_token_predictions(buffer, model, tokenizer, device, model_mode)
        render(buffer, suggs)

        while True:
            ch = sys.stdin.read(1)

            if ch in ('\x03', '\x1b'):
                break
            elif ch in ('\x7f', '\x08'):
                if len(buffer) > 0:
                    buffer = buffer[:-1]
            elif ch == '\t':
                if suggs:
                    top_token = suggs[0][0]
                    if buffer and not buffer.endswith(("\n", " ", "/", ":", "=", "-")) and not top_token.startswith((".", "-", "_", ":")):
                        buffer += " " + top_token
                    else:
                        buffer += top_token
            elif ch in ('\r', '\n'):
                sys.stdout.write("\n\n")
                buffer += "\n"
            elif ord(ch) >= 32:
                buffer += ch

            suggs = get_token_predictions(buffer, model, tokenizer, device, model_mode)
            render(buffer, suggs)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\n\nExited Live Autocomplete. Current Dockerfile buffer:")
        print("-" * 50)
        print(buffer if buffer else "(empty)")
        print("-" * 50)


if __name__ == '__main__':
    model_arg = "auto"
    args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model_arg = arg.split("=", 1)[1]
        elif arg in ("--starcoder", "--star"):
            model_arg = "starcoder"
        elif arg in ("--codebert", "--bert"):
            model_arg = "codebert"
        elif arg in ("--local", "--checkpoint"):
            model_arg = "local"
        else:
            args.append(arg)

    if args:
        prompt = " ".join(args)
        model, tokenizer, device, mode = load_model_and_tokenizer(model_arg)
        preds = get_token_predictions(prompt, model, tokenizer, device, mode, top_k=5)
        print(f"\nPrompt: {prompt!r} (Model: {model_arg})")
        for r, (tok, p) in enumerate(preds, 1):
            print(f"  [{r}] {tok:<20} ({p*100:.1f}%)")
    else:
        if sys.stdin.isatty():
            run_live_keystroke_repl(model_arg)
        else:
            print("Interactive live prediction requires a TTY terminal.")
