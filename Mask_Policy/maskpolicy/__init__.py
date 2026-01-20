#!/usr/bin/env python
"""
MaskPolicy: Mask Action Chunking with Transformers

This project is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""

from maskpolicy.__version__ import __version__  # noqa: F401

# Core APIs
from maskpolicy.policies.factory import make_policy, get_policy_class
from maskpolicy.datasets.factory import make_dataset
from maskpolicy.policies.pretrained import PreTrainedPolicy
from maskpolicy.configs.policies import PreTrainedConfig

__all__ = [
    "__version__",
    "make_policy",
    "get_policy_class",
    "make_dataset",
    "PreTrainedPolicy",
    "PreTrainedConfig",
]

