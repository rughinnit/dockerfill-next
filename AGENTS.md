# AGENTS.md

> **DockerFill-Next Routing & Context Catalog**
> Method: Interpretable Context Methodology (ICM)

## Workspace Architecture

This repository contains data, multi-task deep learning model code, and formal grammar parsers for automated Dockerfile token and line completion.

```
replication-main/
├── AGENTS.md                  # Root routing catalog
├── README.md                  # Project overview and paper details
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excluded environments and caches
├── data/                      # TC & LC datasets, tokenizer vocab
├── model_code/                # Multi-task learning & CodeSage model code
├── parser_code/               # Dockerfile BNF grammar & syntax parser
├── paser_code/                # Backward-compatibility alias for parser_code
├── dockerfill/                # Python 3.10 virtual environment
└── stats.R                    # Wilcoxon signed-rank tests & boxplots
```

## Task Routing Table

| Task | Working Directory | Primary Entrypoint | Expected Outputs |
| :--- | :--- | :--- | :--- |
| **Dockerfile Parsing** | `parser_code/` | `Dockerfile_syntax_parser.py` | AST / syntax token stream |
| **Tokenizer Training** | `model_code/` | `tokenizer.py` | `tokenizer/` or `data/tokenizer.json` |
| **Model Pretraining/FT** | `model_code/` | `model.py` | `./model/` checkpoints |
| **TC Task Evaluation** | `model_code/` | `testing_tc.py` | Acc-1, Acc-10, MRR scores |
| **LC Task Evaluation** | `model_code/` | `testing_lc.py` | EM, ED, BLEU, ROUGE-L, METEOR |
| **Statistical Analysis** | `./` | `stats.R` | Wilcoxon p-values, boxplots |

## Execution Guidelines

1. **Virtual Environment**: Always use Python from `dockerfill/bin/python`.
2. **Component Contracts**: Read the local `CONTEXT.md` in each component folder before executing changes.
3. **Data Protection**: Treat `data/*.pkl` files as read-only fixtures.
