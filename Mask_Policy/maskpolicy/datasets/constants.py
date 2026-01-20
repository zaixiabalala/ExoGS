"""
Dataset constants that are policy-agnostic and can be customized per policy type.
"""
import numpy as np
from typing import Set


# Camera to base transformation (robot-specific, but can be overridden)
CAM_2_BASE = np.array([
    [1, 0, 0, 0],
    [0, -np.sin(70 / 180 * np.pi), np.cos(70 / 180 * np.pi), 0],
    [0, -np.cos(70 / 180 * np.pi), -np.sin(70 / 180 * np.pi), 0.59],
    [0, 0, 0, 1]
])


def get_tensor_keys_from_data(data: dict) -> Set[str]:
    """
    Automatically detect which keys should be tensors based on data types.
    This is a fallback when policy type is not known.
    
    Args:
        data: Sample data dict
    
    Returns:
        Set of keys that appear to be tensor-like (torch.Tensor or np.ndarray)
    """
    import torch
    import numpy as np
    
    tensor_keys = set()
    for key, value in data.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            tensor_keys.add(key)
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            # Check if list contains tensors/arrays
            if isinstance(value[0], (torch.Tensor, np.ndarray)):
                tensor_keys.add(key)
    
    return tensor_keys

