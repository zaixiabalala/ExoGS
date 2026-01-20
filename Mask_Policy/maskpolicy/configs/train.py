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
import builtins
import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus

# from maskpolicy import envs  # Not needed for custom HDF5 datasets
from maskpolicy.configs import parser
from maskpolicy.configs.default import DatasetConfig, EvalConfig
from maskpolicy.configs.policies import PreTrainedConfig
# Import optimizer/scheduler configs lazily to avoid circular imports
# from maskpolicy.optim import OptimizerConfig
# from maskpolicy.optim.schedulers import LRSchedulerConfig

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig:
    dataset: DatasetConfig
    env: Any | None = None  # envs.EnvConfig | None, but envs not available for custom datasets
    policy: Any | None = None  # PreTrainedConfig | None, imported lazily to avoid circular import
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Train/val split ratio. If set (0 < ratio < 1), dataset will be split into train/val sets.
    # Training will use the train set, and split info will be saved to output_dir.
    train_val_ratio: float | None = None
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: Any | None = None  # OptimizerConfig | None, imported lazily
    scheduler: Any | None = None  # LRSchedulerConfig | None, imported lazily
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Save user's policy params from config file before checkpoint config overwrites them
            original_policy_dict = {}
            if self.policy is not None:
                if isinstance(self.policy, dict):
                    original_policy_dict = self.policy.copy()
                else:
                    # Convert dataclass to dict
                    original_policy_dict = draccus.encode(self.policy)
            
            # Load policy config from checkpoint
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
            
            # Training-related parameters that should be overridden from user's config file
            # These are typically changed between training stages, not model architecture params
            training_override_keys = [
                'training_stage',           # training stage (stage1/stage2)
                'mask_loss_weight',         # mask loss weight
                'action_loss_weight',       # action loss weight
                'stage2_freeze_mask_head',  # whether to freeze mask head in stage2
                'stage2_dino_lr_ratio',     # DINO learning rate ratio in stage2
                'use_mask',                 # whether to use mask
                'label_attention_rules',    # label attention rules
                'use_lowdim',
                'optimizer_lr',             # learning rate
                'optimizer_lr_backbone',    # backbone learning rate
                'optimizer_weight_decay',   # weight decay
                'kl_weight',                # KL divergence weight
            ]
            
            # Override checkpoint config with user's training params from config file
            overridden_params = []
            for key in training_override_keys:
                if key in original_policy_dict and original_policy_dict[key] is not None:
                    if hasattr(self.policy, key):
                        checkpoint_value = getattr(self.policy, key)
                        config_value = original_policy_dict[key]
                        if checkpoint_value != config_value:
                            setattr(self.policy, key, config_value)
                            overridden_params.append(
                                f"  - {key}: {checkpoint_value} (checkpoint) -> {config_value} (config)"
                            )
            
            # Log overridden parameters
            if overridden_params:
                logging.info(
                    f"Config overrides checkpoint for the following parameters:\n" + 
                    "\n".join(overridden_params)
                )
            
            # Warn about config params that differ but will be ignored (use checkpoint value)
            ignored_params = []
            for key, config_value in original_policy_dict.items():
                if key in training_override_keys or key in ('type', 'name', 'pretrained_path'):
                    continue
                if hasattr(self.policy, key):
                    checkpoint_value = getattr(self.policy, key)
                    if checkpoint_value != config_value and config_value is not None:
                        ignored_params.append(
                            f"  - {key}: config={config_value}, using checkpoint={checkpoint_value}"
                        )
            
            if ignored_params:
                logging.warning(
                    f"Config params ignored (using checkpoint values):\n" + 
                    "\n".join(ignored_params)
                )
                
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent
        elif isinstance(self.policy, dict):
            # Convert dict to policy config object
            from maskpolicy.policies.factory import make_policy_config
            policy_type = self.policy.get('type')
            if not policy_type:
                raise ValueError("Policy config dict must have 'type' key")
            self.policy = make_policy_config(
                policy_type,
                **{k: v for k, v in self.policy.items() if k not in ('type', 'name')}
            )

        if not self.job_name:
            policy_type = getattr(self.policy, 'type', 'unknown')
            if self.env is None:
                self.job_name = f"{policy_type}"
            else:
                env_type = getattr(self.env, 'type', 'unknown') if self.env else 'unknown'
                self.job_name = f"{env_type}_{policy_type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        # For custom HDF5 datasets, validate hdf5_config if provided
        if self.dataset.custom is not None and self.dataset.custom.hdf5_config is not None:
            self.dataset.custom.hdf5_config.validate()
        
        # Custom dataset is required
        if self.dataset.custom is None:
            raise ValueError("dataset.custom must be provided. Only custom HDF5 datasets are supported.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            # Policy is now always an object after validate()
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """
        Save the training configuration to a local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the configuration will be saved.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        **kwargs,
    ) -> "TrainPipelineConfig":
        """
        Load the training configuration from a local directory or file.

        Args:
            pretrained_name_or_path (`str`, `Path`):
                Path to a `directory` containing the config file, or a direct path to the config file.
            kwargs:
                Additional kwargs to pass to the config parser.
        """
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        
        if Path(model_id).is_dir():
            config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}"
                )
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            raise FileNotFoundError(
                f"Config path not found: {model_id}. Please provide a valid local path to a directory or file."
            )

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)
