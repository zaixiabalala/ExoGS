#!/usr/bin/env python
"""Visualization script for MaskPolicy datasets."""

from maskpolicy.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from maskpolicy.utils.train_utils import set_seed
from maskpolicy.datasets.factory import make_dataset
from maskpolicy.configs.train import TrainPipelineConfig
from maskpolicy.configs import parser
import torch
import logging
import numpy as np

from r3kit.utils.vis import SequenceKeyboardListener
from r3kit.utils.vis import Sequence1DVisualizer, Sequence2DVisualizer, Sequence3DVisualizer
from r3kit.utils.camera import lookat
from r3kit.utils.transformation import xyzrot6d2mat

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

    # Save dataset meta and properties before splitting
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

    listener = SequenceKeyboardListener()
    vis3d = Sequence3DVisualizer()
    vis2d = Sequence2DVisualizer(top=720)
    vis1d = Sequence1DVisualizer(top=1600)
    vis3d.update_frame('world', pose=xyzrot6d2mat(), size=0.5)
    dataset_len = len(dataset)
    idx = 0
    while True:
        data = dataset[idx]

        image = data['observation.images.cam_0'][0]
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype('uint8')
        vis2d.update_image('image', image, type='rgb')

        current_state = data['observation.states'][0].cpu().numpy()
        width = current_state[-1:]
        vis1d.update_item('width_o', width, index=0)
        xyz = current_state[0:3]
        rot6d = current_state[3:9]
        pose = xyzrot6d2mat(xyz, rot6d)
        vis3d.update_frame('pose_o', pose, size=0.1)
        vis3d.update_view(lookat(eye=np.array([1.2, -0.2, 1.])), enforce=True)

        action_state = data['action'].cpu().numpy()
        chunk_size = action_state.shape[0]
        for i in range(chunk_size):
            width = action_state[i, -1:]
            vis1d.update_item('width_a', width, index=i)
            xyz = action_state[i, 0:3]
            rot6d = action_state[i, 3:9]
            pose = xyzrot6d2mat(xyz, rot6d)
            vis3d.update_frame('pose_a', pose, size=0.05)
            vis3d.update_view(lookat(eye=np.array([1.2, -0.2, 1.])), enforce=True)
        
        if not listener.pause:
            delta_idx = listener.speed
            idx += delta_idx
        if listener.forward:
            delta_idx = max(0, min(idx + listener.speed, dataset_len - 1)) - idx
            idx += delta_idx
            listener.forward = False
        if listener.backward:
            delta_idx = max(0, min(idx - listener.speed, dataset_len - 1)) - idx
            idx += delta_idx
            listener.backward = False
        if listener.zero:
            idx = 0
            listener.zero = False
        if listener.quit:
            listener.quit = False
            break

    logging.info("End of training")


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
