"""
Utility functions for dataset operations.

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""
import json
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
from maskpolicy.configs.types import FeatureType, PolicyFeature


def load_json(fpath: Path) -> Any:
    """Load JSON file."""
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    """Write JSON file."""
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Path) -> list[Any]:
    """Load JSONL file."""
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def write_jsonlines(data: dict, fpath: Path) -> None:
    """Write JSONL file."""
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    ```
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary by expanding keys with separators into nested dictionaries."""
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset features dict to PolicyFeature dict."""
    policy_features = {}
    for key, feature_info in features.items():
        key_lower = key.lower()
        if "image" in key_lower or "img" in key_lower:
            feature_type = FeatureType.VISUAL
        elif "action" in key_lower:
            feature_type = FeatureType.ACTION
        elif "reward" in key_lower:
            feature_type = FeatureType.REWARD
        else:
            feature_type = FeatureType.STATE

        shape = tuple(feature_info["shape"]) if "shape" in feature_info and isinstance(feature_info["shape"], (list, tuple)) else feature_info.get("shape", (1,))
        policy_features[key] = PolicyFeature(type=feature_type, shape=shape)
    return policy_features
