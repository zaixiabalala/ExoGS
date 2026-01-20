"""Dataset utilities for train/val split and sample extraction."""

import json
import random
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, random_split


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.9,
    seed: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and validation sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training samples (default: 0.9)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    
    return train_dataset, val_dataset


def extract_hdf5_samples(
    hdf5_path: str,
    output_dir: Path,
    num_samples: int = 10,
    data_group: str = "data",
    seed: int = 1000
) -> None:
    """Extract sample images and masks from HDF5 for visualization.
    
    Args:
        hdf5_path: Path to HDF5 dataset
        output_dir: Output directory for samples
        num_samples: Number of samples to extract
        data_group: HDF5 data group name
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f[data_group]
        
        # Collect all (demo_name, frame_idx) pairs
        all_samples = []
        for demo_name in sorted(data_group.keys()):
            if 'images' in data_group[demo_name]:
                num_frames = len(data_group[demo_name]['images'])
                for frame_idx in range(num_frames):
                    all_samples.append((demo_name, frame_idx))
        
        # Randomly select samples
        selected = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        for i, (demo_name, frame_idx) in enumerate(selected):
            # Read image
            img = data_group[demo_name]['images'][frame_idx]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            
            # Read mask
            mask = data_group[demo_name]['masks'][frame_idx]
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0
            else:
                mask = mask.astype(np.float32)
            
            # Convert to BGR for OpenCV (assuming RGB input)
            img_bgr = (img * 255).astype(np.uint8)
            if img_bgr.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            mask_bgr = (mask * 255).astype(np.uint8)
            if mask_bgr.shape[2] == 3:
                mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_RGB2BGR)
            
            # Save
            img_path = output_dir / f"sample_{i:03d}_rgb.png"
            mask_path = output_dir / f"sample_{i:03d}_mask.png"
            
            cv2.imwrite(str(img_path), img_bgr)
            cv2.imwrite(str(mask_path), mask_bgr)
            
            print(f"Extracted sample {i+1}/{num_samples}: {demo_name}[{frame_idx}]")
            print(f"  -> {img_path.name}, {mask_path.name}")


def save_split_info(
    train_size: int,
    val_size: int,
    output_dir: Path,
    train_ratio: float = 0.9,
    seed: Optional[int] = None
) -> None:
    """Save train/val split information to JSON file.
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        output_dir: Output directory
        train_ratio: Training ratio
        seed: Random seed used
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = output_dir / "train_val_split.json"
    
    split_info = {
        "total_samples": train_size + val_size,
        "train_samples": train_size,
        "val_samples": val_size,
        "train_ratio": train_ratio,
        "seed": seed
    }
    
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split info saved to: {split_file}")

