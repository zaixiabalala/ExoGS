"""
Image transforms configuration (lerobot-style).

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class TransformConfig:
    """Configuration for a single transform."""
    weight: float = 1.0
    type: str = "ColorJitter"  # "ColorJitter", "SharpnessJitter", etc.
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """Configuration for image transforms (lerobot-style)."""
    enable: bool = True
    max_num_transforms: int = 3
    random_order: bool = True
    tfs: Dict[str, TransformConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert dict to TransformConfig if needed."""
        if self.tfs and isinstance(list(self.tfs.values())[0], dict):
            self.tfs = {
                name: TransformConfig(**config) if isinstance(config, dict) else config
                for name, config in self.tfs.items()
            }
