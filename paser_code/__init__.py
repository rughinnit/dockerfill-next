# coding=utf-8
"""DockerFill paser_code package (alias for parser_code)."""

from .Dockerfile_syntax_paser import parse_dockerfile, INSTRUCTIONS, INSTRUCTION_PATTERN

__all__ = [
    "parse_dockerfile",
    "INSTRUCTIONS",
    "INSTRUCTION_PATTERN",
]
