"""Normalization utilities."""
import logging
import torch
from torch import Tensor, nn
from typing import Dict, Optional, TYPE_CHECKING

from maskpolicy.configs.types import FeatureType, NormalizationMode, PolicyFeature

if TYPE_CHECKING:
    from maskpolicy.configs.types import PolicyFeature as PolicyFeatureType


def create_stats_buffers(
    features: Dict[str, PolicyFeature],
    norm_map: Dict[str, NormalizationMode],
    stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    field_norm_map: Optional[Dict[str, NormalizationMode]] = None,
) -> Dict[str, Dict[str, nn.ParameterDict]]:
    """Create buffers per field containing normalization statistics."""
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = (field_norm_map.get(key) if field_norm_map else None) or norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            assert len(shape) == 3, f"{key} must be 3D, got {shape}"
            c, h, w = shape
            assert c < h and c < w, f"{key} must be channel-first, got {shape}"
            shape = (c, 1, 1)
        elif len(shape) > 1 and ft.type in (FeatureType.ACTION, FeatureType.STATE):
            shape = (shape[-1],)

        if norm_mode is NormalizationMode.MEAN_STD:
            buffer = nn.ParameterDict({
                "mean": nn.Parameter(torch.ones(shape, dtype=torch.float32) * torch.inf, requires_grad=False),
                "std": nn.Parameter(torch.ones(shape, dtype=torch.float32) * torch.inf, requires_grad=False),
            })
        else:
            buffer = nn.ParameterDict({
                "min": nn.Parameter(torch.ones(shape, dtype=torch.float32) * torch.inf, requires_grad=False),
                "max": nn.Parameter(torch.ones(shape, dtype=torch.float32) * torch.inf, requires_grad=False),
            })

        if stats and key in stats:
            stat_data = stats[key]
            if norm_mode is NormalizationMode.MEAN_STD:
                mean_stat = torch.tensor(stat_data["mean"], dtype=torch.float32) if isinstance(stat_data.get("mean"), (list, tuple)) else stat_data["mean"].clone().to(dtype=torch.float32)
                std_stat = torch.tensor(stat_data["std"], dtype=torch.float32) if isinstance(stat_data.get("std"), (list, tuple)) else stat_data["std"].clone().to(dtype=torch.float32)
                if mean_stat.shape != buffer["mean"].shape:
                    mean_stat, std_stat = mean_stat.expand_as(buffer["mean"]).clone(), std_stat.expand_as(buffer["std"]).clone()
                buffer["mean"].data, buffer["std"].data = mean_stat, std_stat
            else:
                min_stat = torch.tensor(stat_data["min"], dtype=torch.float32) if isinstance(stat_data.get("min"), (list, tuple)) else stat_data["min"].clone().to(dtype=torch.float32)
                max_stat = torch.tensor(stat_data["max"], dtype=torch.float32) if isinstance(stat_data.get("max"), (list, tuple)) else stat_data["max"].clone().to(dtype=torch.float32)
                if min_stat.shape != buffer["min"].shape:
                    min_stat, max_stat = min_stat.expand_as(buffer["min"]).clone(), max_stat.expand_as(buffer["max"]).clone()
                buffer["min"].data, buffer["max"].data = min_stat, max_stat

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return f"`{name}` is infinity. Initialize with `stats` or use a pretrained model."


def _should_skip_inf_stats(field_norm_map: Optional[Dict[str, NormalizationMode]], key: str) -> bool:
    """Return True if field is configured as IDENTITY (normalization=null)."""
    field_norm = field_norm_map.get(key) if field_norm_map else None
    return field_norm is NormalizationMode.IDENTITY


class Normalize(nn.Module):
    """Normalizes data using saved statistics."""


    def __init__(
        self,
        features: Dict[str, PolicyFeature],
        norm_map: Dict[str, NormalizationMode],
        stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
        dataset_already_normalized: bool = False,
        field_norm_map: Optional[Dict[str, NormalizationMode]] = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.field_norm_map = field_norm_map
        self.stats = stats

        stats_buffers = create_stats_buffers(features, norm_map, stats, field_norm_map)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = (self.field_norm_map.get(key) if self.field_norm_map else None) or self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            if norm_mode is NormalizationMode.MEAN_STD:
                mean, std = buffer["mean"], buffer["std"]
                if torch.isinf(mean).any() or torch.isinf(std).any():
                    if not _should_skip_inf_stats(self.field_norm_map, key):
                        logging.warning(f"Field '{key}' has {norm_mode} but stats are uninitialized. Skipping normalization.")
                    continue
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            else:
                min_val, max_val = buffer["min"], buffer["max"]
                if torch.isinf(min_val).any() or torch.isinf(max_val).any():
                    if not _should_skip_inf_stats(self.field_norm_map, key):
                        logging.warning(f"Field '{key}' has {norm_mode} but stats are uninitialized. Skipping normalization.")
                    continue
                batch[key] = (batch[key] - min_val) / (max_val - min_val + 1e-8) * 2 - 1
        return batch


class Unnormalize(nn.Module):
    """Unnormalizes output data back to original range."""


    def __init__(
        self,
        features: Dict[str, PolicyFeature],
        norm_map: Dict[str, NormalizationMode],
        stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
        dataset_already_normalized: bool = False,
        field_norm_map: Optional[Dict[str, NormalizationMode]] = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.field_norm_map = field_norm_map
        self.stats = stats

        stats_buffers = create_stats_buffers(features, norm_map, stats, field_norm_map)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = (self.field_norm_map.get(key) if self.field_norm_map else None) or self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            if norm_mode is NormalizationMode.MEAN_STD:
                mean, std = buffer["mean"], buffer["std"]
                if torch.isinf(mean).any() or torch.isinf(std).any():
                    if not _should_skip_inf_stats(self.field_norm_map, key):
                        logging.warning(f"Field '{key}' has {norm_mode} but stats are uninitialized. Skipping unnormalization.")
                    continue
                batch[key] = batch[key] * std + mean
            else:
                min_val, max_val = buffer["min"], buffer["max"]
                if torch.isinf(min_val).any() or torch.isinf(max_val).any():
                    if not _should_skip_inf_stats(self.field_norm_map, key):
                        logging.warning(f"Field '{key}' has {norm_mode} but stats are uninitialized. Skipping unnormalization.")
                    continue
                batch[key] = (batch[key] + 1) / 2 * (max_val - min_val) + min_val
        return batch


def create_normalization_modules(
    input_features: Dict[str, "PolicyFeatureType"],
    output_features: Dict[str, "PolicyFeatureType"],
    normalization_mapping: Dict[str, NormalizationMode],
    dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ds_meta=None,
    field_norm_map: Optional[Dict[str, NormalizationMode]] = None,
):
    """Create normalization modules for a policy."""
    return (
        Normalize(input_features, normalization_mapping, dataset_stats, False, field_norm_map),
        Normalize(output_features, normalization_mapping, dataset_stats, False, field_norm_map),
        Unnormalize(output_features, normalization_mapping, dataset_stats, False, field_norm_map)
    )
