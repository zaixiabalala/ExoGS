from maskpolicy.utils.utils import log_model_loading_keys
from maskpolicy.configs.policies import PreTrainedConfig
import abc
import builtins
import logging
import os
from pathlib import Path
from typing import TypeVar

import packaging
import safetensors
import torch
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn
from maskpolicy.configs.types import NormalizationMode

# Constants for model file names
SAFETENSORS_SINGLE_FILE = "model.safetensors"


T = TypeVar("T", bound="PreTrainedPolicy")


class PreTrainedPolicy(nn.Module, abc.ABC):
    """Base class for policy models."""

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save the policy to a local directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self.config.save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(
            save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        load_normalization_stats: bool = True,
        **kwargs,
    ) -> T:
        """Load the policy from a local directory."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path, **kwargs)
        model_id = str(pretrained_name_or_path)
        if not os.path.isdir(model_id):
            raise FileNotFoundError(f"Model directory not found: {model_id}")

        print("Loading weights from local directory")

        if load_normalization_stats and "field_norm_map" not in kwargs:
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            if os.path.exists(model_file):
                inferred_field_norm_map = cls._infer_field_norm_map_from_model(
                    model_file, config)
                if inferred_field_norm_map:
                    kwargs["field_norm_map"] = inferred_field_norm_map

        instance = cls(config, **kwargs)
        model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        policy = cls._load_as_safetensor(
            instance, model_file, config.device, strict, load_normalization_stats)
        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(
        cls, model: T, model_file: str, map_location: str, strict: bool, load_normalization_stats: bool = True
    ) -> T:
        state_dict = safetensors.torch.load_file(
            model_file, device=map_location)
        norm_keys = ("normalize_inputs", "normalize_targets",
                     "unnormalize_outputs")

        if load_normalization_stats:
            filtered_state_dict = {}
            for k, v in state_dict.items():
                is_norm_key = any(k.startswith(nk) for nk in norm_keys)
                if is_norm_key and torch.is_tensor(v) and (torch.isinf(v).any() or torch.isnan(v).any()):
                    logging.warning(f"Skipping invalid normalization stat {k} (contains inf/nan)")
                    continue
                filtered_state_dict[k] = v
        else:
            filtered_state_dict = {k: v for k, v in state_dict.items() if not any(k.startswith(nk) for nk in norm_keys)}

        model_state_dict = model.state_dict()
        compatible_state_dict = {}
        for k, v in filtered_state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                compatible_state_dict[k] = v
            elif k in model_state_dict:
                logging.warning(
                    f"Skipping key {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")

        missing_keys, unexpected_keys = model.load_state_dict(
            compatible_state_dict, strict=False)

        if missing_keys:
            non_norm_missing = [k for k in missing_keys if not any(k.startswith(nk) for nk in norm_keys)]
            if non_norm_missing:
                logging.warning(f"Missing keys (excluding normalization): {non_norm_missing}")
            if load_normalization_stats:
                norm_missing = [k for k in missing_keys if any(k.startswith(nk) for nk in norm_keys)]
                if norm_missing:
                    logging.warning(f"Missing normalization stats: {norm_missing}")

        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")

        if map_location != "cpu":
            model.to(map_location)
        return model

    @classmethod
    def _infer_field_norm_map_from_model(
        cls, model_file: str, config: "PreTrainedConfig"
    ) -> dict[str, NormalizationMode] | None:
        """Infer field_norm_map from saved normalization statistics."""
        try:
            state_dict = safetensors.torch.load_file(model_file, device="cpu")
            norm_prefixes = ("normalize_inputs.buffer_",
                             "normalize_targets.buffer_", "unnormalize_outputs.buffer_")
            field_norm_map = {}

            for field_name in list(config.input_features.keys()) + list(config.output_features.keys()):
                buffer_key_base = field_name.replace(".", "_")
                for prefix in norm_prefixes:
                    mean_key = f"{prefix}{buffer_key_base}.mean"
                    if mean_key in state_dict:
                        v = state_dict[mean_key]
                        if torch.is_tensor(v) and not (torch.isinf(v).any() or torch.isnan(v).any()):
                            field_norm_map[field_name] = NormalizationMode.MEAN_STD
                            break
                    min_key = f"{prefix}{buffer_key_base}.min"
                    if min_key in state_dict:
                        v = state_dict[min_key]
                        if torch.is_tensor(v) and not (torch.isinf(v).any() or torch.isnan(v).any()):
                            field_norm_map[field_name] = NormalizationMode.MIN_MAX
                            break

            return field_norm_map if field_norm_map else None
        except Exception as e:
            logging.debug(
                f"Could not infer field_norm_map from model file: {e}")
            return None

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """Returns the policy-specific parameters dict for the optimizer."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset policy state (clear caches, etc.)."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Forward pass during training. Returns (loss, output_dict)."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk for a given observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment."""
        raise NotImplementedError
