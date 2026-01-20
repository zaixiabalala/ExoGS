#!/usr/bin/env python

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


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from maskpolicy.configs.train import TrainPipelineConfig
from maskpolicy.policies.pretrained import PreTrainedPolicy


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    
    # If using policy preset and optimizer is None, get it from policy config
    if cfg.use_policy_training_preset and cfg.optimizer is None:
        optimizer_config = policy.config.get_optimizer_preset()
        scheduler_config = policy.config.get_scheduler_preset()
    else:
        if cfg.optimizer is None:
            raise ValueError("Optimizer config is None. Set use_policy_training_preset=True or provide optimizer config.")
        optimizer_config = cfg.optimizer
        scheduler_config = cfg.scheduler
    
    optimizer = optimizer_config.build(params)
    lr_scheduler = scheduler_config.build(optimizer, cfg.steps) if scheduler_config is not None else None
    return optimizer, lr_scheduler
