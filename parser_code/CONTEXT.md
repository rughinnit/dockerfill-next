# Parser Code Contract

> Folder: `parser_code/` (and alias `paser_code/`)  
> Purpose: Dockerfile syntax parsing into AST / token-type sequences according to formal BNF grammar.

## Component Map

- `grammar_list.txt`: Formal BNF grammar for Dockerfile instructions (`FROM`, `RUN`, `CMD`, `LABEL`, `EXPOSE`, `ENV`, `ADD`, `COPY`, etc.).
- `Dockerfile_syntax_parser.py`: Regex-based parser transforming Dockerfile textual instructions into syntactic `(token, token_type)` tuples.
- `__init__.py`: Package export interface for external modules.

## Explicit Contract

### Inputs
- Dockerfile plain text (multiline string or file).
- Optional: `data/tokenizer.json` for sub-word token ID alignment.

### Outputs
- List of `(token_str, syntax_type_str)` tuples.
- Aligned token IDs when run against MySQL database.

### Human Verification Gate
- Validate that all Dockerfile instructions (including multi-argument `HEALTHCHECK` and multiline `RUN && ...`) produce non-UNKNOWN syntax types.
