# coding=utf-8
"""DockerFill parser_code package."""

from .Dockerfile_syntax_parser import parse_dockerfile, INSTRUCTIONS, INSTRUCTION_PATTERN

__all__ = [
    "parse_dockerfile",
    "INSTRUCTIONS",
    "INSTRUCTION_PATTERN",
]
