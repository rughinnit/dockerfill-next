# coding=utf-8
"""DockerFill model_code package."""

from .config_codesage import CodeSageConfig
from .modeling_codesage import CodeSageForMaskedLM, CodeSageModel
from .soft_sharing import (
    RobertaForMLM,
    RobertaForSTI,
    RobertaForMIP,
    MultitaskModel,
)
from .hard_sharing import RobertaForMaskedLM as HardSharingRobertaForMaskedLM
from .multitask_data_collator import (
    NLPDataCollator,
    MultitaskDataloader,
    MultitaskTrainer,
    DataLoaderWithTaskname,
)

__all__ = [
    "CodeSageConfig",
    "CodeSageForMaskedLM",
    "CodeSageModel",
    "RobertaForMLM",
    "RobertaForSTI",
    "RobertaForMIP",
    "MultitaskModel",
    "HardSharingRobertaForMaskedLM",
    "NLPDataCollator",
    "MultitaskDataloader",
    "MultitaskTrainer",
    "DataLoaderWithTaskname",
]
