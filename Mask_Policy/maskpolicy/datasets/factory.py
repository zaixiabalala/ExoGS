"""
Dataset factory for creating dataset instances.

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""
import logging
import torch
import numpy as np
from tqdm import tqdm

from maskpolicy.datasets.robot_dataset import ConfigurableRobotDataset
from maskpolicy.configs.default import HDF5DatasetConfig
from maskpolicy.configs.train import TrainPipelineConfig


def compute_dataset_stats(dataset: ConfigurableRobotDataset, hdf5_config: HDF5DatasetConfig) -> dict:
    """
    Compute dataset statistics for normalization from raw HDF5 data.

    Statistics are computed based on normalization config in input_fields/output_fields.
    - "imagenet": Use fixed ImageNet statistics
    - "meanstd": Compute mean and std from raw dataset
    - "minmax": Compute min and max from raw dataset
    - None: No statistics needed

    Args:
        dataset: The dataset to compute statistics from
        hdf5_config: Dataset configuration

    Returns:
        Dictionary mapping feature keys to their statistics
    """
    stats = {}

    IMAGENET_STATS = {
        "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32),
        "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32),
    }

    fields_to_compute = {}
    for fields_dict in [hdf5_config.input_fields, hdf5_config.output_fields]:
        for key, field_config in fields_dict.items():
            if not field_config.normalization or not field_config.hdf5_fields:
                continue
            hdf5_field_key = field_config.hdf5_fields[0]
            if hdf5_field_key not in hdf5_config.hdf5_fields:
                continue
            fields_to_compute[key] = {
                "normalization": field_config.normalization,
                "normalization_params": field_config.normalization_params,
                "shape": field_config.shape,
                "hdf5_key": hdf5_field_key,
                "hdf5_field_config": hdf5_config.hdf5_fields[hdf5_field_key],
            }

    # Compute statistics for each field
    for key, field_info in fields_to_compute.items():
        norm_mode = field_info["normalization"]
        norm_params = field_info["normalization_params"]

        if norm_mode == "imagenet":
            # Use fixed ImageNet statistics for images
            if "mask" not in key.lower():
                # Reshape ImageNet stats to match feature shape
                shape = field_info["shape"]
                if len(shape) == 3:  # (C, H, W) for images
                    c, h, w = shape
                    # ImageNet stats are already in (C, 1, 1) format
                    stats[key] = {
                        "mean": IMAGENET_STATS["mean"],
                        "std": IMAGENET_STATS["std"],
                    }
                else:
                    # For non-image fields, use scalar values
                    stats[key] = {
                        "mean": torch.tensor(0.485, dtype=torch.float32),
                        "std": torch.tensor(0.229, dtype=torch.float32),
                    }
        elif norm_mode in ["meanstd", "minmax"]:
            hdf5_key = field_info["hdf5_key"]
            all_values = []
            for demo_name in dataset.data_group.keys():
                if hdf5_key in dataset.data_group[demo_name]:
                    all_values.append(np.array(dataset.data_group[demo_name][hdf5_key]))
            
            if not all_values:
                logging.warning(f"No data found for key {hdf5_key} to compute statistics.")
                continue
            all_values = np.concatenate(all_values, axis=0)

            shape = field_info["shape"]
            if len(shape) == 3:
                # assume (C, H, W)
                assert len(all_values.shape) == 4 and all_values.shape[1] == shape[0]
            elif len(shape) == 2:
                # assume (N, D)
                assert len(all_values.shape) == 2 and all_values.shape[1] == shape[1]
            elif len(shape) == 1:
                # assume (D,)
                assert len(all_values.shape) == 2 and all_values.shape[1] == shape[0]
            else:
                raise NotImplementedError(f"Normalization not implemented for shape {shape}")
            if norm_mode == "meanstd":
                if norm_params is not None:
                    stats[key] = {
                        "mean": torch.tensor(norm_params["mean"], dtype=torch.float32),
                        "std": torch.tensor(norm_params["std"], dtype=torch.float32),
                    }
                else:
                    if len(shape) == 3:
                        # assume (C, H, W)
                        mean_val, std_val = all_values.mean(axis=(0, 2, 3)), all_values.std(axis=(0, 2, 3)) + 1e-8
                        stats[key] = {
                            "mean": torch.tensor(mean_val, dtype=torch.float32).view(-1, 1, 1),
                            "std": torch.tensor(std_val, dtype=torch.float32).view(-1, 1, 1),
                        }
                    elif len(shape) == 2:
                        # assume (N, D)
                        mean_val, std_val = all_values.mean(axis=0), all_values.std(axis=0) + 1e-8
                        stats[key] = {
                            "mean": torch.tensor(mean_val, dtype=torch.float32),
                            "std": torch.tensor(std_val, dtype=torch.float32),
                        }
                    elif len(shape) == 1:
                        # assume (D,)
                        mean_val, std_val = all_values.mean(axis=0), all_values.std(axis=0) + 1e-8
                        stats[key] = {
                            "mean": torch.tensor(mean_val, dtype=torch.float32),
                            "std": torch.tensor(std_val, dtype=torch.float32),
                        }
                    else:
                        raise NotImplementedError(f"MeanStd normalization not implemented for shape {shape}")
            elif norm_mode == "minmax":
                if norm_params is not None:
                    stats[key] = {
                        "min": torch.tensor(norm_params["min"], dtype=torch.float32),
                        "max": torch.tensor(norm_params["max"], dtype=torch.float32),
                    }
                else:
                    if len(shape) == 3:
                        # assume (C, H, W)
                        min_val, max_val = all_values.min(axis=(0, 2, 3)), all_values.max(axis=(0, 2, 3))
                        stats[key] = {
                            "min": torch.tensor(min_val, dtype=torch.float32).view(-1, 1, 1),
                            "max": torch.tensor(max_val, dtype=torch.float32).view(-1, 1, 1),
                        }
                    elif len(shape) == 2:
                        min_val, max_val = all_values.min(axis=0), all_values.max(axis=0)
                        stats[key] = {
                            "min": torch.tensor(min_val, dtype=torch.float32),
                            "max": torch.tensor(max_val, dtype=torch.float32),
                        }
                    elif len(shape) == 1:
                        min_val, max_val = all_values.min(axis=0), all_values.max(axis=0)
                        stats[key] = {
                            "min": torch.tensor(min_val, dtype=torch.float32),
                            "max": torch.tensor(max_val, dtype=torch.float32),
                        }
                    else:
                        raise NotImplementedError(f"MinMax normalization not implemented for shape {shape}")
            else:
                pass

    return stats


class RobotDatasetAdapter(ConfigurableRobotDataset):
    """Adapter for meta properties compatible with lerobot-style training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta = None

    @property
    def meta(self):
        if self._meta is None:
            class Meta:
                def __init__(self, dataset):
                    self.dataset = dataset
                    self.fps = 30
                    self.features = {}
                    self.stats = {}
                    self.camera_keys = []
                    self.already_normalized = False

                    for fields_dict in [dataset.config.input_fields, dataset.config.output_fields]:
                        for key, field_config in fields_dict.items():
                            self.features[key] = {
                                "shape": list(field_config.shape),
                                "dtype": field_config.dtype,
                            }
                            if "image" in key.lower() or "img" in key.lower():
                                self.camera_keys.append(key)

            self._meta = Meta(self)
        return self._meta

    @property
    def num_frames(self):
        return len(self)

    @property
    def num_episodes(self):
        return len(self.demo_attrs.keys())

    @property
    def episode_data_index(self):
        return {'from': torch.tensor([0]), 'to': torch.tensor([len(self)])}


def make_dataset(cfg):
    """Create dataset from configuration."""
    if not isinstance(cfg, TrainPipelineConfig):
        raise TypeError(f"Expected TrainPipelineConfig, got {type(cfg)}")
    
    if cfg.dataset.custom is None:
        raise NotImplementedError("Only custom dataset is supported. Set dataset.custom.path")
    
    custom_cfg = cfg.dataset.custom
    if custom_cfg.hdf5_config is None:
        raise ValueError("dataset.custom.hdf5_config must be provided.")
    
    hdf5_config = custom_cfg.hdf5_config

    if cfg.policy is not None:
        n_obs_steps = getattr(cfg.policy, 'n_obs_steps', 1)
        chunk_size = getattr(cfg.policy, 'chunk_size', 100)
        delta_indices = {}

        if hasattr(cfg.policy, 'action_delta_indices') and cfg.policy.action_delta_indices is not None:
            delta_indices["action"] = cfg.policy.action_delta_indices
        else:
            delta_indices["action"] = list(range(1, chunk_size + 1))

        obs_deltas = (cfg.policy.observation_delta_indices 
                     if hasattr(cfg.policy, 'observation_delta_indices') and cfg.policy.observation_delta_indices is not None
                     else list(range(-(n_obs_steps - 1), 1)))

        for key in hdf5_config.input_fields.keys():
            if key.startswith("observation."):
                delta_indices[key] = obs_deltas

        for key, field_config in hdf5_config.output_fields.items():
            if key != "action":
                delta_indices[key] = (list(range(1, chunk_size + 1)) 
                                    if len(field_config.shape) > 1 and field_config.shape[0] == chunk_size 
                                    else [0])

        hdf5_config.n_obs_steps = n_obs_steps
        hdf5_config.chunk_size = chunk_size
        hdf5_config.delta_indices = delta_indices
    else:
        hdf5_config.n_obs_steps = getattr(hdf5_config, 'n_obs_steps', 1)
        hdf5_config.chunk_size = getattr(hdf5_config, 'chunk_size', 100)

    dataset = RobotDatasetAdapter(
        path=custom_cfg.path,
        config=hdf5_config,
        with_raw=False,
        image_transforms=cfg.dataset.image_transforms
    )
    dataset.meta.stats = compute_dataset_stats(dataset, hdf5_config)
    return dataset
