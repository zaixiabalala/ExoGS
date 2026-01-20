#!/usr/bin/env python
"""Training script for MaskPolicy."""

from maskpolicy.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from maskpolicy.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from maskpolicy.utils.train_utils import AverageMeter, MetricsTracker, set_seed
from maskpolicy.utils.utils import get_device_from_parameters
from maskpolicy.policies.pretrained import PreTrainedPolicy
from maskpolicy.policies.factory import make_policy
from maskpolicy.optim.factory import make_optimizer_and_scheduler
from maskpolicy.datasets.factory import make_dataset
from maskpolicy.configs.train import TrainPipelineConfig
from maskpolicy.configs import parser
from maskpolicy.datasets.robot_dataset import collate_fn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.amp import GradScaler
import torch
import matplotlib.pyplot as plt
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg') 


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Dict[str, Any],
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler: Optional[LRScheduler] = None,
    use_amp: bool = False,
) -> Tuple[MetricsTracker, Dict[str, Any]]:
    """Update policy with one batch of training data."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)

        # Check for NaN/inf loss before backward pass
        # If loss is NaN/Inf, skip backward pass entirely to avoid corrupting gradients
        if torch.isnan(loss) or torch.isinf(loss):
            loss_value = loss.item()
            logging.warning(
                f"NaN/Inf loss detected at step (loss={loss_value}). "
                f"Skipping this step to prevent parameter corruption. "
                f"GradScaler scale: {grad_scaler.get_scale()}"
            )
            optimizer.zero_grad()
            
            train_metrics.loss = float(
                'nan') if torch.isnan(loss) else float('inf')
            train_metrics.action_loss = float(output_dict['action_loss'])
            train_metrics.mask_loss = float(output_dict['mask_loss'])
            train_metrics.grad_norm = float('nan')
            train_metrics.lr = optimizer.param_groups[0]["lr"]
            train_metrics.update_s = time.perf_counter() - start_time
            # Clear loss and output_dict before returning
            del loss
            if output_dict is not None and isinstance(output_dict, dict):
                for key, value in list(output_dict.items()):
                    if isinstance(value, torch.Tensor):
                        del output_dict[key]
                output_dict.clear()
            return train_metrics, output_dict

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Check for NaN/inf gradients before optimizer step
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        grad_norm_value = grad_norm.item()
        loss_value = loss.item()
        logging.warning(
            f"NaN/Inf gradient norm detected: {grad_norm_value}. "
            f"Skipping optimizer step. Loss: {loss_value:.4f}, "
            f"GradScaler scale: {grad_scaler.get_scale()}"
        )
        optimizer.zero_grad()

        grad_scaler.step(optimizer)
        grad_scaler.update()  # Update scaler state (may reduce scale if inf/nan detected)
        train_metrics.loss = loss_value
        train_metrics.action_loss = float(output_dict['action_loss'])
        train_metrics.mask_loss = float(output_dict['mask_loss'])
        train_metrics.grad_norm = float('nan')
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - start_time
        # Clear tensors before returning
        del loss, grad_norm
        if output_dict is not None and isinstance(output_dict, dict):
            for key, value in list(output_dict.items()):
                if isinstance(value, torch.Tensor):
                    del output_dict[key]
            output_dict.clear()
        return train_metrics, output_dict

    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    # Save loss and grad_norm values before clearing references
    loss_value = loss.item()
    grad_norm_value = grad_norm.item()

    del loss
    del grad_norm

    if output_dict is not None and isinstance(output_dict, dict):
        for key, value in list(output_dict.items()):
            if isinstance(value, torch.Tensor):
                del output_dict[key]

    train_metrics.loss = loss_value
    train_metrics.action_loss = float(output_dict['action_loss'])
    train_metrics.mask_loss = float(output_dict['mask_loss'])
    train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def save_loss_curve(
    loss_history: List[float],
    output_dir: Path,
    step: int,
    log_freq: int = 200,
    output_name: str = "loss_curve.png",
) -> None:
    """Save training loss curve to file."""
    if len(loss_history) == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = [i * log_freq for i in range(len(loss_history))]

    ax.plot(steps, loss_history, label='Training Loss', linewidth=1.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot_path = output_dir / output_name
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


@parser.wrap()
def train(cfg: TrainPipelineConfig) -> None:
    """Main training function."""
    cfg.validate()
    logging.info("Starting training")

    if cfg.seed is not None:
        set_seed(cfg.seed)

    policy_device = getattr(cfg.policy, 'device', None)
    device = get_safe_torch_device(
        policy_device if policy_device else "cuda",
        log=True
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Save dataset meta and properties before splitting (Subset doesn't have these attributes)
    dataset_meta = dataset.meta
    dataset_num_frames = dataset.num_frames
    dataset_num_episodes = dataset.num_episodes

    # Train/val split if train_val_ratio is specified
    train_val_ratio = getattr(cfg, 'train_val_ratio', None)
    if train_val_ratio is not None and 0 < train_val_ratio < 1:
        from maskpolicy.utils.dataset_utils import split_dataset, save_split_info

        logging.info(
            f"Splitting dataset with train/val ratio: {train_val_ratio}")
        train_dataset, val_dataset = split_dataset(
            dataset,
            train_ratio=train_val_ratio,
            seed=cfg.seed
        )
        dataset = train_dataset

        # Update num_frames to reflect the training subset size
        # num_episodes remains the same as original dataset
        dataset_num_frames = len(train_dataset)

        # Save split info
        if cfg.output_dir:
            from pathlib import Path
            save_split_info(
                len(train_dataset),
                len(val_dataset),
                Path(cfg.output_dir),
                train_val_ratio,
                cfg.seed
            )

        logging.info(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    hdf5_config = None
    if cfg.dataset.custom is not None and cfg.dataset.custom.hdf5_config is not None:
        hdf5_config = cfg.dataset.custom.hdf5_config

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset_meta,
        hdf5_config=hdf5_config,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    use_amp = policy.config.use_amp
    grad_scaler = GradScaler(device.type, enabled=use_amp)

    if cfg.use_policy_training_preset and cfg.optimizer is None:
        optimizer_config = policy.config.get_optimizer_preset()
        grad_clip_norm = optimizer_config.grad_clip_norm
    else:
        if cfg.optimizer is None:
            grad_clip_norm = 10.0
        else:
            grad_clip_norm = cfg.optimizer.grad_clip_norm

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(p.numel()
                               for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(f"Output dir: {cfg.output_dir}")
    logging.info(f"steps={cfg.steps} ({format_big_number(cfg.steps)})")
    logging.info(
        f"dataset.num_frames={dataset_num_frames} ({format_big_number(dataset_num_frames)})")
    logging.info(f"dataset.num_episodes={dataset_num_episodes}")
    logging.info(
        f"num_learnable_params={num_learnable_params} ({format_big_number(num_learnable_params)})")
    logging.info(
        f"num_total_params={num_total_params} ({format_big_number(num_total_params)})")

    policy_drop_n_last_frames = getattr(policy, "drop_n_last_frames", None) if hasattr(
        policy, "drop_n_last_frames") else None
    if policy_drop_n_last_frames is not None:
        shuffle = False
        sampler = None
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )
    # Use an explicit iterator instead of itertools.cycle to avoid
    # retaining all batches in memory (which would cause a memory leak).
    dl_iter = iter(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "mask_loss": AverageMeter("mask_l", ":.3f"),
        "action_loss": AverageMeter("act_l", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset_num_frames, dataset_num_episodes, train_metrics, initial_step=step
    )

    # Track loss history for plotting
    loss_history: List[float] = []
    mask_loss_history: List[float] = []
    action_loss_history: List[float] = []

    logging.info("Start training")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        try:
            batch = next(dl_iter)
        except StopIteration:
            # Restart the dataloader when the iterator is exhausted.
            # This avoids storing all batches in memory while still
            # providing an effectively infinite stream of data.
            dl_iter = iter(dataloader)
            batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(
                    device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=use_amp,
        )

        # Clear batch and output_dict to free memory immediately
        del batch
        if output_dict is not None:
            if isinstance(output_dict, dict):
                for key, value in list(output_dict.items()):
                    if isinstance(value, torch.Tensor):
                        del output_dict[key]
                output_dict.clear()
            del output_dict

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        # Periodically clear CUDA cache to prevent memory fragmentation
        # This is especially important for long training runs with large datasets
        # Clear cache more frequently to prevent OOM
        if device.type == "cuda" and step % 50 == 0 and step > 0:
            torch.cuda.empty_cache()

        # Check for NaN in model parameters periodically (every 100 steps)
        # This helps detect if parameters are getting corrupted
        if step % 100 == 0:
            has_nan_params = False
            nan_params_list = []
            for name, param in policy.named_parameters():
                if param.requires_grad:
                    if torch.isnan(param).any():
                        nan_params_list.append(name)
                        has_nan_params = True
                    if torch.isinf(param).any():
                        nan_params_list.append(f"{name} (Inf)")
                        has_nan_params = True
            if has_nan_params:
                logging.error(
                    f"NaN/Inf detected in parameters at step {step}: {nan_params_list[:5]}. "
                    f"Training may be unstable. Loss: {train_tracker.metrics['loss'].avg:.4f}, "
                    f"GradScaler scale: {grad_scaler.get_scale()}. "
                    f"Consider reducing learning rate or gradient clipping."
                )
                # Don't stop training automatically, but log the warning for monitoring

        if is_log_step:
            logging.info(train_tracker)
            loss_history.append(train_tracker.metrics["loss"].avg)
            mask_loss_history.append(train_tracker.metrics["mask_loss"].avg)
            action_loss_history.append(train_tracker.metrics["action_loss"].avg)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(
                cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg,
                            policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            # Save loss curve at checkpoint
            if len(loss_history) > 0:
                save_loss_curve(loss_history, cfg.output_dir,
                                step, cfg.log_freq)
                save_loss_curve(mask_loss_history, cfg.output_dir, step, cfg.log_freq, output_name="mask_loss_curve.png")
                save_loss_curve(action_loss_history, cfg.output_dir, step, cfg.log_freq, output_name="action_loss_curve.png")

    # Save final loss curve
    if len(loss_history) > 0:
        save_loss_curve(loss_history, cfg.output_dir, step, cfg.log_freq)
        save_loss_curve(mask_loss_history, cfg.output_dir, step, cfg.log_freq, output_name="mask_loss_curve.png")
        save_loss_curve(action_loss_history, cfg.output_dir, step, cfg.log_freq, output_name="action_loss_curve.png")
    logging.info("End of training")


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
