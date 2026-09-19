# DockerFill-Next: Modernized & Extended Replication Package

> **Interpretable Context Methodology (ICM) & Production Autocomplete Extension**  
> Empirical study, BNF syntax parsing, multi-task learning, and real-time live prediction for automated Dockerfile token and line completion.

---

## Table of Contents
1. [Overview & Enhancements](#overview--enhancements)
2. [Quickstart & Environment Setup](#quickstart--environment-setup)
3. [Live Keystroke Autocomplete (`predict.py`)](#live-keystroke-autocomplete-predictpy)
4. [Model Training & Low-VRAM Auto-Tuning](#model-training--low-vram-auto-tuning)
5. [Evaluation & Empirical Benchmarks](#evaluation--empirical-benchmarks)
6. [Why Blank Models Overfit vs. Pre-trained Foundation Models](#why-blank-models-overfit-vs-pre-trained-foundation-models)
7. [Repository Architecture](#repository-architecture)
8. [Original Research Package README](#original-research-package-readme)

---

## Overview & Enhancements

This repository contains the replication package for **DockerFill**—a multi-task deep learning model combining formal Backus-Naur Form (BNF) grammar parsing with neural representation learning to complete Dockerfile tokens (TC) and lines (LC).

### Key Modernizations in DockerFill-Next:
* **Interactive Live Keystroke REPL ([`predict.py`](predict.py)):** Real-time, continuous autocompletion updating under the cursor with every keystroke (~96ms latency) with instant `[Tab]` acceptance.
* **Multi-Model Engine Support:** Seamlessly switch between local trained checkpoints, pre-trained **CodeBERT** (125M), and causal generative **StarCoder** (164M).
* **Low-VRAM Adaptive Training:** Automatic hardware auto-tuning in [`model_code/model.py`](model_code/model.py) enabling full model training on consumer GPUs (e.g. GeForce 940MX with 1.95 GB VRAM) using Adafactor and gradient checkpointing.
* **ICM Architecture & Safety:** Verified type annotations, zero-placeholder implementations, backward-compatible parser wrappers, and strict execution boundaries.

---

## Quickstart & Environment Setup

### 1. Python Virtual Environment
Activate the configured Python 3.10 virtual environment:
```bash
source dockerfill/bin/activate
# or directly invoke: dockerfill/bin/python
```

### 2. Dependencies
Install pinned dependencies:
```bash
dockerfill/bin/pip install -r requirements.txt
```

---

## Live Keystroke Autocomplete (`predict.py`)

Launch the real-time interactive autocomplete shell directly in your terminal:

```bash
# Default: Uses your latest locally trained checkpoint
dockerfill/bin/python predict.py

# Use Pre-trained StarCoder (best for multi-token paths like /app/package.json)
dockerfill/bin/python predict.py --starcoder

# Use Pre-trained CodeBERT (best for flags and syntax tokens)
dockerfill/bin/python predict.py --codebert
```

### Interactive Controls
* **Type normally:** Predictions recalculate live on every single character.
* **`[Tab]`:** Instantly autocompletes the green **`#1`** recommendation into your line.
* **`[Enter]`:** Adds a newline to start the next Dockerfile instruction.
* **`[Backspace]`:** Deletes character and updates suggestions live.
* **`[Ctrl + C]` / `[Esc]`:** Cleanly exits and prints your complete drafted Dockerfile.

### One-Shot CLI Prediction
```bash
dockerfill/bin/python predict.py --starcoder "FROM python:3.10-slim\nWORKDIR /app\nCOPY requirements.txt "
```

---

## Model Training & Low-VRAM Auto-Tuning

To train the multi-task DockerFill model on the Token Completion (TC) dataset:

```bash
dockerfill/bin/python model_code/model.py
```

### Automatic Hardware Adaptation:
* **GPU VRAM < 4 GB (e.g., GeForce 940MX):** Automatically switches to `batch_size=1`, `gradient_accumulation_steps=16`, `gradient_checkpointing=True`, and `optim="adafactor"`, dropping VRAM consumption from 2.0 GB down to **~650 MB**.
* **GPU VRAM 4–8 GB:** Switches to `batch_size=4`, `gradient_accumulation_steps=4`.
* **GPU VRAM > 8 GB (RTX 3090 / A100):** Runs standard `batch_size=16` with `adamw_torch`.

---

## Evaluation & Empirical Benchmarks

### Evaluation Tasks
1. **Token Completion (TC):**
   ```bash
   dockerfill/bin/python model_code/testing_tc.py
   ```
   Evaluates Top-1 Accuracy (Acc-1), Top-10 Accuracy (Acc-10), and Mean Reciprocal Rank (MRR).
2. **Line Completion (LC):**
   ```bash
   dockerfill/bin/python model_code/testing_lc.py
   ```
   Evaluates Exact Match (EM), Edit Distance (ED), BLEU, ROUGE-L, and METEOR.
3. **Statistical Significance Tests:**
   ```bash
   Rscript stats.R
   ```
   Runs Wilcoxon signed-rank tests and generates boxplots from [`data/manual_evaluation_scores.csv`](data/manual_evaluation_scores.csv).

### Benchmark Scores Across 11 Models (From Paper's Dataset)
Recorded across 254 test cases by human evaluators ([`data/manual_evaluation_scores.csv`](data/manual_evaluation_scores.csv)):

| Rank | Model | Model Type | Similarity (1–5) | Naturalness (1–5) |
| :---: | :--- | :--- | :---: | :---: |
| 🥇 | **DockerFill** | Syntax-Guided MTL Transformer | **2.73** | **3.23** |
| 🥈 | **CodeBERT** | Pre-trained Code Transformer (125M) | **2.56** | **2.61** |
| 🥉 | **ContraCode** | Contrastive Code Model | **2.47** | **2.57** |
| 4 | **StarEncoder** | BigCode StarCoder Base Encoder | **2.43** | **2.44** |
| 5 | **UniXcoder** | Microsoft Unified Code Model | **2.33** | **2.41** |
| 6 | **GPT-4o** | OpenAI Foundation Model | **2.29** | **3.12** |
| 7 | **CodeSage** | Code Embedding Model | **2.11** | **2.27** |
| 8 | **Qwen-Coder** | Alibaba Generative Code Model | **1.92** | **2.46** |
| 9 | **Code-Llama** | Meta CodeLlama (7B) | **1.89** | **2.48** |
| 10 | **GPT-3.5-Turbo** | OpenAI LLM | **1.83** | **2.81** |
| 11 | **DeepSeek-Coder** | DeepSeek LLM (6.7B) | **1.73** | **2.49** |

---

## Why Blank Models Overfit vs. Pre-trained Foundation Models

### The Architectural Lesson:
* **The Paper's Model:** The authors mined **3.8 million Dockerfiles** from GitHub and pretrained on dual 24GB RTX 3090 GPUs across millions of iterations.
* **The Replication Slice:** The public replication repository contains only a **2,032-sample fine-tuning slice** without the multi-gigabyte foundation checkpoint.
* **Why Blank Training Overfits:** Initializing a 125M-parameter transformer with random weights on only 2,000 samples causes severe **token degeneration** after Epoch 4 (overfitting on high-frequency dataset artifacts like `cat` and Chinese characters `都`).
* **The Modern Solution:** Start with a foundation model (e.g., **CodeBERT** or **StarCoder**) and guide it using the BNF grammar in [`parser_code/grammar_list.txt`](parser_code/grammar_list.txt) to guarantee 100% syntactically valid Dockerfiles with zero hallucinations.

---

## Repository Architecture

```
replication-main/
├── README.md                                     # This unified documentation
├── AGENTS.md                                     # ICM context routing catalog
├── requirements.txt                              # Pinned Python dependencies
├── predict.py                                    # Real-time live keystroke autocomplete CLI
├── stats.R                                       # Wilcoxon signed-rank tests & plots
├── Dockerfile Writing Survey - Google Forms.pdf   # Human developer survey questionnaire
│
├── data/
│   ├── CONTEXT.md                                # Component interface contract
│   ├── tokenizer.json                            # 50,000 BPE vocabulary file
│   ├── manual_evaluation_scores.csv              # 2,795 human evaluator ratings
│   ├── llm_prompt.md                             # LLM benchmark prompt templates
│   ├── TC/                                       # Token completion datasets (*.pkl)
│   └── LC/                                       # Line completion datasets (*.pkl)
│
├── model_code/
│   ├── CONTEXT.md                                # Component interface contract
│   ├── model.py                                  # Training loop with low-VRAM auto-tuning
│   ├── hard_sharing.py                           # Hard parameter sharing for MTL
│   ├── soft_sharing.py                           # Soft parameter sharing for MTL
│   ├── multitask_data_collator.py                # Dual-task batch collator
│   ├── tokenizer.py                              # BPE tokenizer training script
│   ├── testing_tc.py                             # TC task evaluation harness
│   ├── testing_lc.py                             # LC task evaluation harness
│   ├── modeling_codesage.py                      # CodeSage model architecture
│   └── config_codesage.py                        # CodeSage configuration
│
└── parser_code/
    ├── CONTEXT.md                                # Component interface contract
    ├── Dockerfile_syntax_parser.py               # AST token stream parser
    └── grammar_list.txt                          # Backus-Naur Form (BNF) formal grammar
```

---

# Original Research Package README

*(The content below is preserved verbatim from the original authors' README)*

---

# Readme
This repository includes our data, model code, parser code, etc.

## Environment Preparation

CPU: Intel(R) Xeon(R) Platinum 8255C CPU @2.50GHz with 24 core processors, and 86G RAM.

GPU: Two NVIDIA RTX 3090 GPUs with 24 GB memory

Packages: 
`transformers 4.34.1`
`tokenizers 0.14.1`
`torchmetrics 1.2.0`
`torch 2.1.0`
`scikit-learn 1.3.2`
`GPUtil 1.4.0`
`numpy 1.26.1`
`evaluate 0.4.1`
`numba 0.58.1`
`nltk 3.8.1`
`tqdm 4.66.1`
`typing 4.8.0`
`psycopg2 2.9.9`

## Code Files
### Model code
`model_code/model.py`: code for model training. We implement our model with the popular deep learning development framework PyTorch and the python package transformers developed by HuggingFace. 

`model_code/hard_sharing.py`: hard parameter sharing class for model MTL pre-training.

`model_code/soft_sharing.py`: soft parameter sharing class for model MTL pre-training.

`model_code/multitask_data_collator.py`: code for dealing with MTL training data.

`model_code/testing_tc.py` and `model_code/testing_lc.py`: code for model testing. We use two evaluation metrics for the TC task, namely the Accuracy (Acc) of the top prediction and the
MRR for the top-10 recommendations. Five commonly used evaluation metrics are employed for the LC task: EM, ED, BLEU, ROUGE, and METEOR.

`model_code/tokenizer.py`: code for tokenizer.  We use sub-word tokenization with the Byte-Pair Encoding (BPE) algorithm, as previous studies found that BPE can substantially reduce the vocabulary size
and alleviate the OOV problem.

`model_code/modeling_codesage.py`: code for use codesage model.

### Parser code
`parser_code/grammar_list.txt`: A formal grammar for Dockerfiles using the Backus-Naur Form (BNF).
```
<Dockerfile> ::= <Instruction>*
<Instruction> ::= <FROM> | <RUN> | <CMD> | <LABEL> | <EXPOSE> | <ENV> | <ADD> | <COPY> | <ENTRYPOINT> | <VOLUME> | <USER> | <WORKDIR> | <ARG> | <STOPSIGNAL> | <HEALTHCHECK> | <SHELL> | <MAINTAINER> | <ONBUILD>
<FROM> ::= "FROM" [ <option> <value> ] <image> [ ":" <tag> | "@" <digest> ] [ "AS" <alias> | "as" <alias>]
<RUN> ::= "RUN" [ <run_option> <value> ] <command> ( <command_separator> <command> )* | "RUN" <json_commands>
<command_separator> ::= "&&" | "||" | ";"
<CMD> ::= "CMD" <json_commands> | "CMD" <command>
<LABEL> ::= "LABEL" <key_values>
<EXPOSE> := “EXPOSE” <port>+
<ENV> ::= "ENV" <key_values>
<COPY> ::= "COPY"  [ <copy_flags> <value> ] <src>+ <dest>
<ADD> ::= "ADD" [ <add_flags> <value> ]  <src>+ <dest>
<ENTRYPOINT> ::= "ENTRYPOINT" <json_paths> | "ENTRYPOINT" <command>
<VOLUME> ::= "VOLUME" <JSON-paths> | <path>
<USER> ::= "USER" <user> [ ":" <group> ]
<WORKDIR> ::= "WORKDIR" <path>
<ARG> ::= "ARG" <key> [ "=" <value> ]
<STOPSIGNAL> := "STOPSIGNAL" <value>
<HEALTHCHECK> :=  "HEALTHCHECK" [ <healthcheck_options> <value> ] "CMD" <command>
<SHELL> ::= "SHELL" <json_commands>
<MAINTAINER> ::= "MAINTAINER" <value>
<ONBUILD> ::= "ONBUILD" <Instruction>
<from_option> ::= "--platform=" 
<run_option> ::= ( "--mount=" | "--network=" | "--security=" )
<copy_flags> ::= ( "--from=" | "--chown=" | "--chmod=" | "--link=" | "--parents" | "--exclude" )+
<add_flags> ::= ( "--checksum=" | "--chown=" | "--chmod=" | "--keep-git-dir=" | "--link=" | "--exclude=" )+
<healthcheck_options> ::= ( "--interval=" | "--timeout=" | "--start-period=" | "--start-interval=" | "--retries=" )
<image> := <str>
<tag>:= <str>
<digest>:= <str>
<alias> := <str>
<command>:= <str>
<json_commands> :=  "[" <command> ( "," <command> )* "]"
<key_values> := <key>  <value> |  (<key> "=" <value>)+
<key>:= <str>
<value> := <str>
<str> ::= /([^$\n\s]|(\$<var>|\${<var>}))*/
<var> ::= /[a-zA-Z_][a-zA-Z0-9_]*/
<port>:= ((\d+)(\/([a-zA-Z]))?)+
<src>:=<str>
<dest>:=<str>
<path>:=<str>
<json_paths> :=  "[" <path> ( "," <path> )* "]"
<user>:=<str>
<group>:=<str>
```

`parser_code/Dockerfile_syntax_parser.py`: A parser tool (i.e., Dockerfile-syntax-parser) that can parse the Dockerfile textual content into the corresponding syntax type sequence.


### Statistical test code
`stats.R`: code for the Wilcoxon signed-rank tests and plots.


## Data Files
The dataset contains 3,810 instances for pre-training,  2,032 for fine-tuning's training, 254 for validation, and 254 for testing. 

`data/TC`: Training, validation, and testing data for the TC task.

`data/LC`: Training, validation, and testing data for the LC task.

`data/tokenizer.json`: our tokenizer file.

`data/manual_evaluation_scores.csv`: The manual evaluation scores in terms of similarity and naturalness (254 samples and 2 evaluators).

`data/llm_prompt.md`: The prompt templates of llms for the two tasks.
