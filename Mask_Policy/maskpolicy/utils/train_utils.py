#!/usr/bin/env python
import logging
import os
import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from maskpolicy.configs.train import TrainPipelineConfig
from maskpolicy.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from maskpolicy.datasets.utils import load_json, write_json
from maskpolicy.optim.optimizers import load_optimizer_state, save_optimizer_state
from maskpolicy.optim.schedulers import load_scheduler_state, save_scheduler_state
from maskpolicy.policies.pretrained import PreTrainedPolicy
from maskpolicy.utils.utils import format_big_number


# Training utilities
def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_history(train_history, num_epochs, ckpt_dir, seed):
    """Save training curves."""
    plt.figure()
    plt.plot(np.linspace(0, num_epochs, len(train_history)),
             train_history, label='train')
    plt.tight_layout()
    plt.legend()
    plt.title("loss")
    plt.savefig(os.path.join(ckpt_dir, f'train_seed_{seed}.png'))


def sync_loss(loss, device):
    """Synchronize loss across distributed processes."""
    t = [loss]
    t = torch.tensor(t, dtype=torch.float64, device=device)
    dist.barrier()
    dist.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
    return t[0]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """A helper class to track and log metrics over time."""

    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        self.metrics = metrics

        self.steps = initial_step
        # A sample is an (observation,action) pair, where observation and action
        # can be on multiple timestamps. In a batch, we have `batch_size` number of samples.
        self.samples = self.steps * self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """Updates metrics that depend on 'step' for one step."""
        self.steps += 1
        self.samples += self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """Returns the current metric values (or averages if `use_avg=True`) as a dict."""
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow",
                 attrs=["bold"]) + f" {out_dir}")


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> None:
    """Save checkpoint with policy, config, optimizer, and scheduler states."""
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """Save the training step, optimizer state, scheduler state, and rng state."""
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """Load training state to resume a training run."""
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler


def save_rng_state(save_dir: Path):
    """Save random number generator states (torch, numpy, python random)."""
    import random
    import json
    save_dir.mkdir(parents=True, exist_ok=True)

    numpy_state = np.random.get_state()
    numpy_state_serializable = (
        numpy_state[0],
        numpy_state[1].tolist() if isinstance(
            numpy_state[1], np.ndarray) else numpy_state[1],
        int(numpy_state[2]) if isinstance(
            numpy_state[2], np.ndarray) else numpy_state[2],
        int(numpy_state[3]) if isinstance(
            numpy_state[3], np.ndarray) else numpy_state[3],
        numpy_state[4].tolist() if isinstance(
            numpy_state[4], np.ndarray) else numpy_state[4],
    )

    rng_state = {
        "torch": torch.get_rng_state().cpu().numpy().tolist(),
        "numpy": numpy_state_serializable,
        "python": list(random.getstate()),
    }

    from maskpolicy.constants import RNG_STATE
    with open(save_dir / RNG_STATE, "w") as f:
        json.dump(rng_state, f)


def load_rng_state(save_dir: Path):
    """Load random number generator states (torch, numpy, python random)."""
    import random
    import json
    from maskpolicy.constants import RNG_STATE

    rng_file = save_dir / RNG_STATE
    if not rng_file.exists():
        return

    with open(rng_file, "r") as f:
        rng_state = json.load(f)

    if "torch" in rng_state:
        torch.set_rng_state(torch.tensor(rng_state["torch"]))
    if "numpy" in rng_state:
        numpy_state = rng_state["numpy"]
        numpy_state_tuple = (
            numpy_state[0],
            np.array(numpy_state[1]),
            numpy_state[2],
            numpy_state[3],
            np.array(numpy_state[4]),
        )
        np.random.set_state(tuple(numpy_state_tuple))
    if "python" in rng_state:
        random.setstate(tuple(rng_state["python"]))
