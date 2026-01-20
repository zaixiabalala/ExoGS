"""
Data augmentation utilities.

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""
import random
import numpy as np
import torch
from torchvision import transforms
from typing import Dict, List, Optional, Any, Union, Tuple

from maskpolicy.datasets.transforms import ImageTransformsConfig, TransformConfig


def apply_color_jitter(image: np.ndarray, brightness: float = 0.0, contrast: float = 0.0,
                       saturation: float = 0.0, hue: float = 0.0) -> np.ndarray:
    """Apply color jitter augmentation to an image.
    
    Note: ColorJitter expects [0, 1] range input, so we normalize temporarily
    if input is in [0, 255] range, then restore original range after augmentation.
    Final normalization to [0, 1] is handled in _process_field_data.
    """
    is_numpy = isinstance(image, np.ndarray)
    original_dtype = image.dtype if is_numpy else None
    
    # Check if we need to normalize for ColorJitter (which expects [0, 1])
    needs_normalization = False
    if is_numpy:
        if image.dtype == np.uint8 or (image.dtype != np.uint8 and image.max() > 1.0):
            needs_normalization = True
            if image.dtype == np.uint8:
                image_tensor = torch.from_numpy(image).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image).float() / 255.0
        else:
            image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image

    original_shape = image_tensor.shape
    if len(original_shape) == 3:
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        squeeze_output = True
    elif len(original_shape) == 4:
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported image shape: {original_shape}")

    def is_nonzero(val):
        return any(v != 0 for v in val) if isinstance(val, (list, tuple)) and len(val) > 0 else (val > 0 if isinstance(val, (int, float)) else False)

    if is_nonzero(brightness) or is_nonzero(contrast) or is_nonzero(saturation) or is_nonzero(hue):
        brightness_range = brightness if isinstance(
            brightness, (list, tuple)) else (max(0, 1 - brightness), 1 + brightness)
        contrast_range = contrast if isinstance(
            contrast, (list, tuple)) else (max(0, 1 - contrast), 1 + contrast)
        saturation_range = saturation if isinstance(
            saturation, (list, tuple)) else (max(0, 1 - saturation), 1 + saturation)
        hue_range = hue if isinstance(hue, (list, tuple)) else (-hue, hue)

        jitter_kwargs = {}
        if is_nonzero(brightness):
            jitter_kwargs['brightness'] = brightness_range
        if is_nonzero(contrast):
            jitter_kwargs['contrast'] = contrast_range
        if is_nonzero(saturation):
            jitter_kwargs['saturation'] = saturation_range
        if is_nonzero(hue):
            jitter_kwargs['hue'] = hue_range

        if jitter_kwargs:
            image_tensor = transforms.ColorJitter(
                **jitter_kwargs)(image_tensor)

    if squeeze_output:
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
    else:
        image_tensor = image_tensor.permute(0, 2, 3, 1)

    # Restore original range if we normalized
    if needs_normalization:
        image_tensor = image_tensor * 255.0

    if is_numpy:
        image_tensor = image_tensor.numpy()
        # Restore original dtype and range
        if original_dtype == np.uint8:
            image_tensor = np.clip(image_tensor, 0, 255).astype(original_dtype)
        else:
            image_tensor = image_tensor.astype(original_dtype)

    return image_tensor


def apply_sharpness_jitter(image: np.ndarray, sharpness: float = 0.0) -> np.ndarray:
    """Apply sharpness jitter augmentation to an image.
    
    Note: RandomAdjustSharpness expects [0, 1] range input, so we normalize temporarily
    if input is in [0, 255] range, then restore original range after augmentation.
    Final normalization to [0, 1] is handled in _process_field_data.
    """
    is_numpy = isinstance(image, np.ndarray)
    original_dtype = image.dtype if is_numpy else None
    
    # Check if we need to normalize for RandomAdjustSharpness (which expects [0, 1])
    needs_normalization = False
    if is_numpy:
        if image.dtype == np.uint8 or (image.dtype != np.uint8 and image.max() > 1.0):
            needs_normalization = True
            if image.dtype == np.uint8:
                image_tensor = torch.from_numpy(image).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image).float() / 255.0
        else:
            image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image

    original_shape = image_tensor.shape
    if len(original_shape) == 3:
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        squeeze_output = True
    elif len(original_shape) == 4:
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported image shape: {original_shape}")

    if isinstance(sharpness, (list, tuple)) and len(sharpness) == 2:
        sharpness_factor = random.uniform(sharpness[0], sharpness[1])
        image_tensor = transforms.RandomAdjustSharpness(
            sharpness_factor=sharpness_factor, p=1.0)(image_tensor)
    elif isinstance(sharpness, (int, float)) and sharpness > 0:
        sharpness_factor = random.uniform(max(0, 1 - sharpness), 1 + sharpness)
        image_tensor = transforms.RandomAdjustSharpness(
            sharpness_factor=sharpness_factor, p=1.0)(image_tensor)

    if squeeze_output:
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
    else:
        image_tensor = image_tensor.permute(0, 2, 3, 1)

    # Restore original range if we normalized
    if needs_normalization:
        image_tensor = image_tensor * 255.0

    if is_numpy:
        image_tensor = image_tensor.numpy()
        # Restore original dtype and range
        if original_dtype == np.uint8:
            image_tensor = np.clip(image_tensor, 0, 255).astype(original_dtype)
        else:
            image_tensor = image_tensor.astype(original_dtype)

    return image_tensor


def apply_noise(data: np.ndarray, std: float = 0.002) -> np.ndarray:
    """Apply Gaussian noise augmentation to 1D data."""
    return data + np.random.normal(0, std, data.shape).astype(data.dtype)


def apply_image_transforms(image: np.ndarray, config: ImageTransformsConfig) -> np.ndarray:
    """Apply image transforms based on configuration."""
    if not config.enable or not config.tfs:
        return image

    available_transforms = [(name, tf_config) for name, tf_config in config.tfs.items() 
                           if random.random() < tf_config.weight]
    if not available_transforms:
        return image

    if config.random_order:
        random.shuffle(available_transforms)
    selected_transforms = available_transforms[:min(
        len(available_transforms), config.max_num_transforms)]

    result = image
    for name, tf_config in selected_transforms:
        kwargs = tf_config.kwargs
        if tf_config.type == "ColorJitter":
            result = apply_color_jitter(result, brightness=kwargs.get("brightness", 0.0),
                                      contrast=kwargs.get("contrast", 0.0),
                                        saturation=kwargs.get(
                                            "saturation", 0.0),
                                      hue=kwargs.get("hue", 0.0))
        elif tf_config.type == "SharpnessJitter":
            result = apply_sharpness_jitter(
                result, sharpness=kwargs.get("sharpness", 0.0))
        else:
            raise ValueError(f"Unknown transform type: {tf_config.type}")

    return result


def apply_spatial_augmentation(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    flip_horizontal: bool = True,
    flip_vertical: bool = False,
    rotation: bool = True,
    rotation_range: Tuple[float, float] = (-15, 15),
    crop: bool = True,
    crop_scale: Tuple[float, float] = (0.8, 1.0),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply spatial augmentation to image and mask together.

    Args:
        image: Input image (H, W, C) or (H, W)
        mask: Optional mask (H, W) or (H, W, C) - will be transformed the same way
        flip_horizontal: Whether to apply random horizontal flip
        flip_vertical: Whether to apply random vertical flip
        rotation: Whether to apply random rotation
        rotation_range: Rotation angle range in degrees
        crop: Whether to apply random crop
        crop_scale: Crop scale range (0.8 means crop to 80% of original size)

    Returns:
        Tuple of (augmented_image, augmented_mask)
    """
    if mask is not None:
        assert image.shape[:2] == mask.shape[:2], \
            f"Image and mask must have same spatial dimensions. Got {image.shape[:2]} vs {mask.shape[:2]}"

    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    original_dtype_img = image.dtype
    original_dtype_mask = mask.dtype if mask is not None else None

    # Ensure float for processing
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if mask is not None and mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    H, W = image.shape[:2]

    # Random horizontal flip
    if flip_horizontal and random.random() < 0.5:
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)

    # Random vertical flip
    if flip_vertical and random.random() < 0.5:
        image = np.flipud(image)
        if mask is not None:
            mask = np.flipud(mask)

    # Random rotation
    if rotation and random.random() < 0.5:
        angle = random.uniform(rotation_range[0], rotation_range[1])
        # Use scipy for rotation
        from scipy.ndimage import rotate
        image = rotate(image, angle, axes=(0, 1), reshape=False,
                       order=1, mode='constant', cval=0.0)
        if mask is not None:
            # For masks, use nearest neighbor interpolation to preserve class labels
            mask = rotate(mask, angle, axes=(0, 1), reshape=False,
                          order=0, mode='constant', cval=0.0)

    # Random crop and resize
    if crop and random.random() < 0.5:
        scale = random.uniform(crop_scale[0], crop_scale[1])
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        # Random crop position
        top = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))

        # Crop
        if len(image.shape) == 3:
            image_cropped = image[top:top+crop_h, left:left+crop_w, :]
        else:
            image_cropped = image[top:top+crop_h, left:left+crop_w]

        if mask is not None:
            if len(mask.shape) == 3:
                mask_cropped = mask[top:top+crop_h, left:left+crop_w, :]
            else:
                mask_cropped = mask[top:top+crop_h, left:left+crop_w]
        else:
            mask_cropped = None

        # Resize back to original size
        import cv2
        if len(image_cropped.shape) == 3:
            image = cv2.resize(image_cropped, (W, H),
                               interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image_cropped, (W, H),
                               interpolation=cv2.INTER_LINEAR)

        if mask_cropped is not None:
            if len(mask_cropped.shape) == 3:
                mask = cv2.resize(mask_cropped, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            else:
                mask = cv2.resize(mask_cropped, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

    # Restore original dtype
    if original_dtype_img != np.float32:
        if original_dtype_img == np.uint8:
            image = np.clip(image, 0, 255).astype(original_dtype_img)
        else:
            image = image.astype(original_dtype_img)

    if mask is not None and original_dtype_mask != np.float32:
        if original_dtype_mask == np.uint8:
            mask = np.clip(mask, 0, 255).astype(original_dtype_mask)
        else:
            mask = mask.astype(original_dtype_mask)

    # Ensure arrays are contiguous (no negative strides) before returning
    # This is necessary because np.fliplr/flipud can create views with negative strides
    # which PyTorch doesn't support when converting to tensors
    image = np.ascontiguousarray(image)
    if mask is not None:
        mask = np.ascontiguousarray(mask)

    return image, mask
