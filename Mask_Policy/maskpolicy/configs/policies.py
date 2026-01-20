# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import builtins
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import draccus

from maskpolicy.configs.types import FeatureType, NormalizationMode, PolicyFeature

# Constants for config file names
CONFIG_NAME = "config.json"
from maskpolicy.optim.optimizers import OptimizerConfig
from maskpolicy.optim.schedulers import LRSchedulerConfig
from maskpolicy.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available

T = TypeVar("T", bound="PreTrainedConfig")


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mp
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    def __post_init__(self):
        self.pretrained_path = None
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logging.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logging.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    def save_pretrained(self, save_directory: str | Path) -> None:
        """
        Save the configuration to a local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the configuration will be saved.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        **policy_kwargs,
    ) -> T:
        """
        Load the configuration from a local directory.

        Args:
            pretrained_name_or_path (`str`, `Path`):
                Path to a `directory` containing the config file saved using `.save_pretrained`,
                e.g., `../path/to/my_model_directory/`.
            policy_kwargs:
                Additional kwargs to pass to the config parser.
        """
        model_id = str(pretrained_name_or_path)
        if not Path(model_id).is_dir():
            raise FileNotFoundError(
                f"Config directory not found: {model_id}. Please provide a valid local path."
            )
        
        config_file = os.path.join(model_id, CONFIG_NAME)
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"{CONFIG_NAME} not found in {Path(model_id).resolve()}"
            )

        with open(config_file) as f:
            config = json.load(f)

        # Save top-level 'type' for first parse
        top_level_type = config.get("type")
        
        # Remove deprecated fields that are no longer in config class
        config.pop("optimizer_config", None)
        config.pop("type", None)

        # Create config for second parse (without 'type')
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            temp_file_2 = f.name
            f.flush()

        try:
            # Create config for first parse (with 'type' to determine subclass)
            if top_level_type:
                config_with_type = config.copy()
                config_with_type["type"] = top_level_type
            else:
                config_with_type = config.copy()

            with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
                json.dump(config_with_type, f)
                temp_file_1 = f.name
                f.flush()

            try:
                # HACK: Parse the original config to get the config subclass, so that we can
                # apply cli overrides.
                # This is very ugly, ideally we'd like to be able to do that natively with draccus
                # something like --policy.path (in addition to --policy.type)
                with draccus.config_type("json"):
                    orig_config = draccus.parse(cls, temp_file_1, args=[])

                cli_overrides = policy_kwargs.pop("cli_overrides", [])
                with draccus.config_type("json"):
                    return draccus.parse(orig_config.__class__, temp_file_2, args=cli_overrides)
            finally:
                if os.path.exists(temp_file_1):
                    os.unlink(temp_file_1)
        finally:
            if os.path.exists(temp_file_2):
                os.unlink(temp_file_2)
