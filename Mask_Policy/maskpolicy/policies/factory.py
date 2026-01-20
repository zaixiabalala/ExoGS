#!/usr/bin/env python
"""
Policy factory for creating policy instances.

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""

import logging

from torch import nn

from maskpolicy.configs.policies import PreTrainedConfig
from maskpolicy.configs.types import FeatureType, NormalizationMode
from maskpolicy.configs.default import HDF5DatasetConfig
from maskpolicy.datasets.utils import dataset_to_policy_features
from maskpolicy.policies.mask_act.configuration_mask_act import MaskACTConfig
from maskpolicy.policies.pretrained import PreTrainedPolicy


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class given a name."""
    if name == "mask_act":
        from maskpolicy.policies.mask_act.modeling_mask_act import MaskACTPolicy
        return MaskACTPolicy
    else:
        raise NotImplementedError(
            f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    # Remove 'type' from kwargs as it's not a config parameter
    kwargs.pop('type', None)
    if policy_type == "mask_act":
        return MaskACTConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def infer_normalization_mapping(
    cfg: PreTrainedConfig,
    hdf5_config: HDF5DatasetConfig = None,
    features: dict = None,
) -> tuple[dict[str, NormalizationMode], dict[str, NormalizationMode]]:
    """Infer normalization_mapping from dataset configuration.
    
    Priority: dataset config > policy config > defaults
    """
    # Start with empty mapping - will be filled from dataset config first
    feature_type_mapping = {}
    field_mapping = {}

    if hdf5_config is not None and features is not None:
        all_fields = {**hdf5_config.input_fields, **hdf5_config.output_fields}
        for key, field_config in all_fields.items():
            if key not in features:
                continue

            norm_mode_str = field_config.normalization
            if norm_mode_str is None or norm_mode_str == "null":
                norm_mode = NormalizationMode.IDENTITY
            elif norm_mode_str in ["imagenet", "meanstd"]:
                norm_mode = NormalizationMode.MEAN_STD
            elif norm_mode_str == "minmax":
                norm_mode = NormalizationMode.MIN_MAX
            else:
                norm_mode = NormalizationMode.IDENTITY

            field_mapping[key] = norm_mode
            if norm_mode != NormalizationMode.IDENTITY:
                feature_type_str = features[key].type.value if isinstance(features[key].type, FeatureType) else str(features[key].type)
                # Dataset config takes priority - always use it if available
                if feature_type_str not in feature_type_mapping:
                    feature_type_mapping[feature_type_str] = norm_mode
                elif feature_type_mapping[feature_type_str] != norm_mode:
                    logging.warning(f"Conflicting normalization modes for {feature_type_str} (field: {key}): "
                                  f"existing={feature_type_mapping[feature_type_str]}, new={norm_mode}")

    policy_mapping = (cfg.normalization_mapping if hasattr(cfg, 'normalization_mapping') and cfg.normalization_mapping
                      else {})
    for feature_type, norm_mode in policy_mapping.items():
        if feature_type not in feature_type_mapping:
            feature_type_mapping[feature_type] = norm_mode

    defaults = [("VISUAL", NormalizationMode.MEAN_STD), ("STATE", NormalizationMode.MIN_MAX), ("ACTION", NormalizationMode.MIN_MAX)]
    for key, default in defaults:
        if key not in feature_type_mapping:
            feature_type_mapping[key] = default

    return feature_type_mapping, field_mapping


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta=None,
    hdf5_config: HDF5DatasetConfig = None,
    **kwargs,
) -> PreTrainedPolicy:
    """Make an instance of a policy class."""
    if isinstance(cfg, dict):
        policy_type = cfg.get('type')
        if not policy_type:
            raise ValueError("Policy config dict must have 'type' key")
        cfg_dict = {k: v for k, v in cfg.items() if k not in ('type', 'name')}
        cfg = make_policy_config(policy_type, **cfg_dict)
        policy_cls = get_policy_class(policy_type)
    else:
        policy_cls = get_policy_class(cfg.type)

    policy_kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        policy_kwargs["dataset_stats"] = ds_meta.stats
        policy_kwargs["ds_meta"] = ds_meta
    else:
        if not cfg.pretrained_path:
            logging.warning("Instantiating policy without dataset metadata. Normalization will have infinite values.")
        features = {}

    if not cfg.output_features and not cfg.input_features:
        if hdf5_config and hdf5_config.output_fields:
            output_field_keys = set(hdf5_config.output_fields.keys())
            cfg.output_features = {key: ft for key, ft in features.items()
                                  if key in output_field_keys or ft.type is FeatureType.ACTION}
        else:
            cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    feature_type_mapping, field_mapping = infer_normalization_mapping(cfg, hdf5_config, features)
    if hasattr(cfg, 'normalization_mapping') and (not cfg.normalization_mapping or len(cfg.normalization_mapping) == 0):
        cfg.normalization_mapping = feature_type_mapping
        logging.info(f"Inferred normalization_mapping: {feature_type_mapping}")

    policy_kwargs["field_norm_map"] = field_mapping if field_mapping else None
    policy_kwargs["config"] = cfg
    policy_kwargs.update(kwargs)

    if cfg.pretrained_path:
        policy_kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**policy_kwargs)
    else:
        policy = policy_cls(**policy_kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)
    return policy
