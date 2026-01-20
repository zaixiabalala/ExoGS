#!/usr/bin/env python
"""Visualization utilities for MaskPolicy models."""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2


# Default color palette for mask classes
# 0: background (black), 1: robot arm (green), 2: object (red)
DEFAULT_MASK_COLORS = np.array([
    [0, 0, 0],        # 0: background - black
    [0, 255, 0],      # 1: robot arm - green
    [255, 0, 0],      # 2: object (all blocks) - red
], dtype=np.uint8)

DEFAULT_MASK_NAMES = ['Background', 'Robot Arm', 'Object']


def visualize_stage1_masks(
    images: torch.Tensor,
    pixel_preds: torch.Tensor,
    gt_masks: Optional[torch.Tensor] = None,
    pixel_logits: Optional[torch.Tensor] = None,
    num_classes: int = 4,
    mask_colors: Optional[np.ndarray] = None,
    mask_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (15, 5),
) -> np.ndarray:
    """Visualize stage1 mask predictions.
    
    Args:
        images: (B, T, C, H, W) or (B, C, H, W) input images
        pixel_preds: (B, T, H, W) or (B, H, W) predicted mask labels
        gt_masks: (B, T, H, W) or (B, H, W) ground truth masks (optional)
        pixel_logits: (B, T, num_classes, H, W) pixel logits for confidence (optional)
        num_classes: Number of mask classes
        mask_colors: (num_classes, 3) RGB color array for each class
        mask_names: List of class names
        save_path: Path to save the visualization
        show: Whether to display the figure
        figsize: Figure size (width, height)
    
    Returns:
        Visualization image as numpy array
    """
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(pixel_preds, torch.Tensor):
        pixel_preds = pixel_preds.detach().cpu().numpy()
    if gt_masks is not None and isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu().numpy()
    if pixel_logits is not None and isinstance(pixel_logits, torch.Tensor):
        pixel_logits = pixel_logits.detach().cpu().numpy()
    
    
    # Handle dimensions
    if images.ndim == 4:  # (B, C, H, W)
        images = images[:, np.newaxis, ...]  # (B, 1, C, H, W)
    if pixel_preds.ndim == 3:  # (B, H, W)
        pixel_preds = pixel_preds[:, np.newaxis, ...]  # (B, 1, H, W)
    if gt_masks is not None and gt_masks.ndim == 3:
        gt_masks = gt_masks[:, np.newaxis, ...]


    B, T, C, H, W = images.shape
    mask_colors = mask_colors if mask_colors is not None else DEFAULT_MASK_COLORS[:num_classes]
    mask_names = mask_names if mask_names is not None else DEFAULT_MASK_NAMES[:num_classes]
    
    # Process first batch item, first timestep
    img = images[0, 0].transpose(1, 2, 0)  # (H, W, C) - 从 (C, H, W) 转置
    pred_mask = pixel_preds[0, 0]  # (H, W)
    gt_mask = gt_masks[0, 0] if gt_masks is not None else None
    
    if img.max() > 1.0:
        img = img / 255.0

    img = np.clip(img, 0, 1)

    
    # Convert mask predictions to colored image
    pred_colored = mask_colors[pred_mask]  # (H, W, 3)
    pred_colored = pred_colored.astype(np.float32) / 255.0
    
    # Create figure
    n_cols = 3 if gt_masks is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    axes[0].imshow(img, vmin=0.0, vmax=1.0)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(pred_colored)
    axes[1].set_title('Predicted Mask', fontsize=12)
    axes[1].axis('off')
    
    # Ground truth mask (if available)
    if gt_masks is not None:
        gt_colored = mask_colors[gt_mask]  # (H, W, 3)
        gt_colored = gt_colored.astype(np.float32) / 255.0
        axes[2].imshow(gt_colored)
        axes[2].set_title('Ground Truth Mask', fontsize=12)
        axes[2].axis('off')
    
    # Add legend
    patches = [mpatches.Patch(color=color/255.0, label=name) 
               for color, name in zip(mask_colors, mask_names)]
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # Get visualization array before closing
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape((height, width, 4))
    vis_array = rgba[..., :3].copy()
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return vis_array


def visualize_patch_labels(
    images: torch.Tensor,
    patch_labels: torch.Tensor,
    pixel_preds: Optional[torch.Tensor] = None,
    patch_size: int = 16,
    num_classes: int = 4,
    mask_colors: Optional[np.ndarray] = None,
    mask_names: Optional[List[str]] = None,
    overlay_alpha: float = 0.5,
    draw_grid: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (15, 5),
) -> np.ndarray:
    """Visualize patch labels from stage2.
    
    Args:
        images: (B, T, C, H, W) or (B, C, H, W) input images
        patch_labels: (B, T*N) or (B, T, H_feat, W_feat) patch labels
        pixel_preds: (B, T, H, W) pixel-level predictions (optional, for comparison)
        patch_size: Patch size (16 for DINOv3)
        num_classes: Number of mask classes
        mask_colors: (num_classes, 3) RGB color array for each class
        mask_names: List of class names
        overlay_alpha: Transparency for overlay (0-1)
        draw_grid: Whether to draw patch grid lines
        save_path: Path to save the visualization
        show: Whether to display the figure
        figsize: Figure size (width, height)
    
    Returns:
        Visualization image as numpy array
    """
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(patch_labels, torch.Tensor):
        patch_labels = patch_labels.detach().cpu().numpy()
    if pixel_preds is not None and isinstance(pixel_preds, torch.Tensor):
        pixel_preds = pixel_preds.detach().cpu().numpy()
    
    # Handle dimensions
    if images.ndim == 4:  # (B, C, H, W)
        images = images[:, np.newaxis, ...]  # (B, 1, C, H, W)

    if patch_labels.ndim == 3:  # (B, T, N)
        # Reshape to spatial layout
        B, T, N = patch_labels.shape
        H_feat = W_feat = int(N ** 0.5)
        patch_labels = patch_labels.reshape(B, T, H_feat, W_feat)
    if pixel_preds is not None and pixel_preds.ndim == 3:
        pixel_preds = pixel_preds[:, np.newaxis, ...]
    
    B, T, C, H, W = images.shape
    _, _, H_feat, W_feat = patch_labels.shape
    
    mask_colors = mask_colors if mask_colors is not None else DEFAULT_MASK_COLORS[:num_classes]
    mask_names = mask_names if mask_names is not None else DEFAULT_MASK_NAMES[:num_classes]
    
    # Process first batch item, first timestep
    img = images[0, 0].transpose(1, 2, 0)  # (H, W, C)
    patch_labels_t = patch_labels[0, 0]  # (H_feat, W_feat)
    
    # Normalize image
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0, 1)
    
    # Upsample patch labels to image size
    patch_labels_upsampled = cv2.resize(
        patch_labels_t.astype(np.float32),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)
    
    # Create colored overlay
    patch_colored = mask_colors[patch_labels_upsampled]  # (H, W, 3)
    patch_colored = patch_colored.astype(np.float32) / 255.0
    
    # Create overlay
    overlay = (1 - overlay_alpha) * img + overlay_alpha * patch_colored
    
    # Create figure
    n_cols = 3 if pixel_preds is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Patch labels overlay
    axes[1].imshow(overlay)
    if draw_grid:
        # Draw patch grid
        patch_h, patch_w = H // H_feat, W // W_feat
        for i in range(H_feat + 1):
            y = i * patch_h
            axes[1].axhline(y, color='white', linewidth=0.5, alpha=0.3)
        for j in range(W_feat + 1):
            x = j * patch_w
            axes[1].axvline(x, color='white', linewidth=0.5, alpha=0.3)
    axes[1].set_title('Patch Labels Overlay', fontsize=12)
    axes[1].axis('off')
    
    # Pixel predictions (if available)
    if pixel_preds is not None:
        pred_mask = pixel_preds[0, 0]  # (H, W)
        pred_colored = mask_colors[pred_mask]  # (H, W, 3)
        pred_colored = pred_colored.astype(np.float32) / 255.0
        axes[2].imshow(pred_colored)
        axes[2].set_title('Pixel Predictions', fontsize=12)
        axes[2].axis('off')
    
    # Add legend
    patches = [mpatches.Patch(color=color/255.0, label=name) 
               for color, name in zip(mask_colors, mask_names)]
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # Get visualization array before closing
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape((height, width, 4))
    vis_array = rgba[..., :3].copy()
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return vis_array


def get_stage1_predictions(
    policy,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Stage1 inference API: similar to `select_action`, given a batch, return the prediction results directly."""
    policy.eval()
    device = next(policy.parameters()).device

    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.no_grad():
        results = policy.predict_masks_stage1(batch)

    image_key = list(policy.config.image_features.keys())[0]
    images = batch[image_key]

    out: Dict[str, torch.Tensor] = {
        "images": images,
        "pixel_preds": results["pixel_preds"],
    }
    if "gt_masks" in results:
        out["gt_masks"] = results["gt_masks"]
    if "pixel_logits" in results:
        out["pixel_logits"] = results["pixel_logits"]
    return out


def batch_visualize_stage1(
    policy,
    batch: Dict[str, torch.Tensor],
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    idx: int = 0,
) -> np.ndarray:
    """Stage1一站式可视化：先调用 `get_stage1_predictions`，再调用 `visualize_stage1_masks`。"""
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    preds = get_stage1_predictions(policy, batch)
    save_path = save_dir / f"stage1_sample_{idx:04d}.png" if save_dir else None

    vis = visualize_stage1_masks(
        images=preds["images"],
        pixel_preds=preds["pixel_preds"],
        gt_masks=preds.get("gt_masks"),
        pixel_logits=preds.get("pixel_logits"),
        num_classes=policy.config.num_mask,
        save_path=save_path,
        show=show,
    )
    return vis


def get_stage2_predictions(
    policy,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Stage2 inference API: given a batch, return the prediction results directly."""
    policy.eval()
    device = next(policy.parameters()).device

    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.no_grad():
        results = policy.get_patch_labels_stage2(batch)

    image_key = [k for k in policy.config.image_features.keys() if "mask" not in k.lower()][0]
    images = batch[image_key]

    out: Dict[str, torch.Tensor] = {
        "images": images,
        "patch_labels": results["patch_labels"],
    }
    if "pixel_preds" in results:
        out["pixel_preds"] = results["pixel_preds"]
    return out


def visualize_single_stage2(
    policy,
    batch: Dict[str, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> np.ndarray:
    preds = get_stage2_predictions(policy, batch)

    vis = visualize_patch_labels(
        images=preds["images"],
        patch_labels=preds["patch_labels"],
        pixel_preds=preds.get("pixel_preds"),
        patch_size=16,  # DINOv3 patch size
        num_classes=policy.config.num_mask,
        save_path=save_path,
        show=show,
    )
    
    return vis

