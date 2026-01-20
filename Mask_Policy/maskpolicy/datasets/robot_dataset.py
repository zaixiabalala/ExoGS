"""
Config-driven HDF5 Robot Dataset.

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""
import collections.abc as container_abcs
from copy import deepcopy
import random
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

# Import rotation_transform only when needed (for quaternion_to_6d transform)
try:
    from maskpolicy.utils.transformation import rotation_transform, apply_mat_to_pcd
    ROTATION_TRANSFORM_AVAILABLE = True
except ImportError:
    rotation_transform = None
    apply_mat_to_pcd = None
    ROTATION_TRANSFORM_AVAILABLE = False

from maskpolicy.datasets.augmentation import (
    apply_color_jitter, apply_noise, apply_image_transforms, apply_spatial_augmentation
)
from maskpolicy.datasets.transforms import ImageTransformsConfig
from maskpolicy.datasets.constants import CAM_2_BASE, get_tensor_keys_from_data
from maskpolicy.configs.default import HDF5DatasetConfig, HDF5FieldConfig, DatasetFieldConfig
from typing import Optional


def random_pose(x_min=-0.05, x_max=0.05, y_min=-0.04, y_max=0.04, z_min=-0.04, z_max=0.05, z_theta_min=-np.pi/3, z_theta_max=np.pi/3):
    """Generate random pose augmentation."""
    x_trans = np.random.uniform(x_min, x_max)
    y_trans = np.random.uniform(y_min, y_max)
    z_trans = np.random.uniform(z_min, z_max)
    z_theta = np.random.uniform(z_theta_min, z_theta_max)
    rot = Rot.from_euler('XYZ', np.array([0, 0, z_theta])).as_matrix()
    random_pose = np.identity(4)
    random_pose[:3, :3] = rot
    random_pose[:3, 3] = np.array([x_trans, y_trans, z_trans])
    cam2base = deepcopy(CAM_2_BASE)
    transform_pose = np.linalg.inv(cam2base) @ random_pose @ cam2base
    return transform_pose


def random_rotation(theta: float = 2/180*np.pi):
    """Generate random rotation augmentation."""
    random_angle = np.random.uniform(0, theta)
    random_axis = np.random.randn(3)
    random_axis /= np.linalg.norm(random_axis)
    transform_rotation = Rot.from_rotvec(
        random_axis * random_angle).as_matrix()
    return transform_rotation


class ConfigurableRobotDataset(Dataset):
    """Config-driven HDF5 robot dataset."""

    def __init__(self, path: str, config: HDF5DatasetConfig, with_raw: bool = False,
                 image_transforms: Optional[ImageTransformsConfig] = None):
        super().__init__()
        self.path = path
        self.config = config
        self.with_raw = with_raw
        # Convert dict to ImageTransformsConfig if needed
        if image_transforms is None:
            self.image_transforms = ImageTransformsConfig(enable=False)
        elif isinstance(image_transforms, dict):
            self.image_transforms = ImageTransformsConfig(**image_transforms)
        else:
            self.image_transforms = image_transforms

        config.validate()
        self._hdf5_file = None
        self._data_group = None
        with h5py.File(self.path, 'r', swmr=True, libver='latest') as temp_file:
            temp_data_group = temp_file[config.hdf5_data_group]
            self.data_attrs = dict(temp_data_group.attrs)
            self.demo_attrs = {
                demo_name: dict(temp_data_group[demo_name].attrs)
                for demo_name in temp_data_group.keys()
            }

        self.episode_frames = []
        self.episode_lengths = []
        self.episode_names = []
        with h5py.File(self.path, 'r', swmr=True, libver='latest') as temp_file:
            temp_data_group = temp_file[config.hdf5_data_group]
            for demo_name in sorted(self.demo_attrs.keys()):
                if len(self.config.hdf5_fields) > 0:
                    episode_length = len(
                        temp_data_group[demo_name][list(self.config.hdf5_fields.keys())[0]])
                else:
                    episode_length = self.demo_attrs[demo_name].get(
                        'num_samples', 0)

                if self.config.allow_padding:
                    valid_start, valid_end = 0, max(0, episode_length - 1)
                else:
                    valid_start, valid_end = 0, max(0, episode_length - 1)
                    if self.config.delta_indices:
                        min_delta = min((min(delta_list) for key, delta_list in self.config.delta_indices.items()
                                         if key.startswith("observation.") and delta_list), default=0)
                        if min_delta < 0:
                            valid_start = max(0, -min_delta)

                        max_delta = max((max(delta_list) for key, delta_list in self.config.delta_indices.items()
                                         if (key in self.config.output_fields or key == "action") and delta_list), default=0)
                        if max_delta > 0:
                            valid_end = max(valid_start - 1,
                                            episode_length - max_delta - 1)
                    else:
                        valid_start = max(0, self.config.n_obs_steps - 1)
                        valid_end = max(
                            valid_start - 1, episode_length - self.config.chunk_size)

                if episode_length > 0:
                    if self.config.allow_padding:
                        self.episode_frames.append(
                            (demo_name, valid_start, valid_end, episode_length))
                        self.episode_lengths.append(episode_length)
                        self.episode_names.append(demo_name)
                    elif valid_end >= valid_start:
                        self.episode_frames.append(
                            (demo_name, valid_start, valid_end, episode_length))
                        self.episode_lengths.append(
                            valid_end - valid_start + 1)

        prefix_sum = [0]
        for length in self.episode_lengths:
            prefix_sum.append(prefix_sum[-1] + length)
        self.prefix_sums = (np.array(prefix_sum[1:], dtype=int), np.array(
            prefix_sum[:-1], dtype=int))

    @property
    def hdf5_file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(
                self.path, 'r', swmr=True, libver='latest')
        return self._hdf5_file

    @property
    def data_group(self):
        if self._data_group is None:
            self._data_group = self.hdf5_file[self.config.hdf5_data_group]
        return self._data_group

    def __del__(self):
        if hasattr(self, "_hdf5_file") and self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None
        self._data_group = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hdf5_file'] = None
        state['_data_group'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return sum(self.episode_lengths) if self.episode_lengths else 0

    def _apply_augmentation(self, data: np.ndarray, field_config: HDF5FieldConfig,
                            aug_pose: np.ndarray = None) -> np.ndarray:
        """Apply augmentation based on field configuration."""
        if "mask" in field_config.hdf5_key.lower() or field_config.augmentation is None:
            return data

        if field_config.augmentation == "pose" and aug_pose is not None:
            # if apply_mat_to_pcd is None:
            #     raise ImportError(
            #         "apply_mat_to_pcd is required for pose augmentation. Install dependencies.")
            # if field_config.dims == 3:  # Point cloud
            #     # Apply pose transform to point cloud
            #     data = np.stack([apply_mat_to_pcd(data[i].copy(), aug_pose)
            #                     for i in range(len(data))])
            # elif field_config.dims == 1:  # Pose data
            #     if "xyz" in field_config.hdf5_key.lower():
            #         data = apply_mat_to_pcd(data.copy(), aug_pose)
            #     # Note: quaternion transform would need additional implementation
            raise NotImplementedError

        elif field_config.augmentation == "color":
            if (hasattr(self.image_transforms, 'enable') and self.image_transforms.enable and
                    hasattr(self.image_transforms, 'tfs') and self.image_transforms.tfs):
                if field_config.dims == 2:
                    data = apply_image_transforms(data, self.image_transforms)
                elif field_config.dims == 3:
                    raise NotImplementedError
            else:
                if field_config.dims == 2:
                    data = apply_color_jitter(
                        data, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                elif field_config.dims == 3:
                    raise NotImplementedError

        elif field_config.augmentation == "spatial":
            # For spatial augmentation, we still apply color augmentation first if image_transforms are enabled
            # Spatial augmentation will be applied later together with masks
            if (hasattr(self.image_transforms, 'enable') and self.image_transforms.enable and
                    hasattr(self.image_transforms, 'tfs') and self.image_transforms.tfs):
                if field_config.dims == 2:
                    data = apply_image_transforms(data, self.image_transforms)
                elif field_config.dims == 3:
                    raise NotImplementedError
            else:
                if field_config.dims == 2:
                    data = apply_color_jitter(
                        data, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                elif field_config.dims == 3:
                    raise NotImplementedError

        elif field_config.augmentation == "noise":
            # if field_config.dims == 1:
            #     data = apply_noise(data, std=0.002)
            raise NotImplementedError

        return data

    def _apply_transform(self, data: np.ndarray, field_config: HDF5FieldConfig) -> np.ndarray:
        """Apply transformation based on field configuration."""
        if field_config.transform == "quaternion_to_6d":
            # if rotation_transform is None:
            #     raise ImportError(
            #         "rotation_transform is required. Install pytorch3d: pip install pytorch3d")
            # data = rotation_transform(
            #     data[:, [3, 0, 1, 2]], from_rep="quaternion", to_rep="rotation_6d")
            raise NotImplementedError
        return data

    def __getitem__(self, idx):
        episode_idx = np.logical_and(
            idx < self.prefix_sums[0], idx >= self.prefix_sums[1]).nonzero()[0][0]
        frame_idx_in_episode = idx - self.prefix_sums[1][episode_idx]
        demo_name, valid_start, valid_end, episode_length = self.episode_frames[episode_idx]
        demo_group = self.data_group[demo_name]
        frame_idx = valid_start + frame_idx_in_episode

        aug_pose = None
        if any(field.augmentation == "pose" for field in self.config.hdf5_fields.values()):
            aug_pose = random_pose()

        if self.config.output_format == "policy":
            output = self._extract_data_with_delta_indices(
                demo_group, frame_idx, episode_length, aug_pose)
            output['_normalized'] = False
            return output
        else:
            return {'pad_o': torch.zeros(1, dtype=torch.bool), 'pad_a': torch.zeros(1, dtype=torch.bool)}

    def _extract_data_with_delta_indices(
        self, demo_group, frame_idx: int, episode_length: int, aug_pose: np.ndarray | None
    ) -> dict:
        """Extract data for all fields using delta_indices."""
        if self.config.delta_indices is None:
            raise ValueError("delta_indices must be set in config.")

        output = {}
        ep_start, ep_end = 0, episode_length
        query_indices = {}
        padding_masks = {}

        for key, delta_list in self.config.delta_indices.items():
            query_indices[key] = [
                max(ep_start, min(ep_end - 1, frame_idx + delta)) for delta in delta_list]
            if self.config.allow_padding:
                padding_masks[key] = torch.BoolTensor([
                    (frame_idx + delta < ep_start) or (frame_idx + delta >= ep_end) for delta in delta_list
                ])
            else:
                padding_masks[key] = torch.zeros(
                    len(delta_list), dtype=torch.bool)

        for key, indices in query_indices.items():
            field_config = self.config.input_fields.get(
                key) or self.config.output_fields.get(key)
            if not field_config:
                continue

            hdf5_field_key = field_config.hdf5_fields[0]
            hdf5_field_config = self.config.hdf5_fields.get(hdf5_field_key)

            data_chunk = demo_group[hdf5_field_key][indices]
            data_tensor = torch.from_numpy(data_chunk).float() if isinstance(
                    data_chunk, np.ndarray) else torch.tensor(data_chunk, dtype=torch.float32)
            if hdf5_field_config is not None:
                data_tensor = self._process_field_data(
                    data_tensor, hdf5_field_config, aug_pose)

            output[key] = data_tensor
            output[f"{key}_is_pad"] = padding_masks[key]

        # Apply spatial augmentation to images and masks together if enabled
        # Check if any field has "spatial" augmentation type
        image_keys = [k for k in query_indices.keys(
        ) if 'images' in k and 'mask' not in k]
        mask_keys = [k for k in query_indices.keys() if 'mask' in k]

        # Check if spatial augmentation is enabled for any field
        has_spatial_aug = False
        for key in list(query_indices.keys()):
            field_config = self.config.input_fields.get(
                key) or self.config.output_fields.get(key)
            if field_config:
                hdf5_field_key = field_config.hdf5_fields[0]
                hdf5_field_config = self.config.hdf5_fields.get(hdf5_field_key)
                if hdf5_field_config and hdf5_field_config.augmentation == "spatial":
                    has_spatial_aug = True
                    break

        # Apply spatial augmentation if enabled and we have both images and masks
        if has_spatial_aug and len(image_keys) > 0 and len(mask_keys) > 0 and random.random() < 0.5:
            # Get first image and mask (assuming n_obs_steps=1)
            img_key = image_keys[0]
            mask_key = mask_keys[0]

            if img_key in output and mask_key in output:
                # Convert to numpy for augmentation
                # output[key] shape: (T, C, H, W) or (T, H, W)
                img_data = output[img_key]  # (T, C, H, W)
                mask_data = output[mask_key]  # (T, C, H, W) or (T, H, W)

                # Process each timestep
                T = img_data.shape[0]
                augmented_imgs = []
                augmented_masks = []

                for t in range(T):
                    # Convert to (H, W, C) format for augmentation
                    if img_data.dim() == 4:  # (C, H, W)
                        img_np = img_data[t].permute(1, 2, 0).numpy()
                    else:  # (H, W)
                        img_np = img_data[t].numpy()

                    if mask_data.dim() == 4:  # (C, H, W)
                        mask_np = mask_data[t].permute(1, 2, 0).numpy()
                    elif mask_data.dim() == 3:  # (H, W)
                        mask_np = mask_data[t].numpy()
                    else:
                        mask_np = None

                    # Apply spatial augmentation
                    aug_img, aug_mask = apply_spatial_augmentation(
                        img_np, mask_np,
                        flip_horizontal=True,
                        flip_vertical=False,
                        rotation=True,
                        rotation_range=(-10, 10),
                        crop=True,
                        crop_scale=(0.85, 1.0)
                    )

                    # Convert back to tensor
                    if len(aug_img.shape) == 3:  # (H, W, C)
                        aug_img = torch.from_numpy(aug_img).permute(2, 0, 1)
                    else:  # (H, W)
                        aug_img = torch.from_numpy(aug_img)
                    augmented_imgs.append(aug_img)

                    if aug_mask is not None:
                        if len(aug_mask.shape) == 3:  # (H, W, C)
                            aug_mask = torch.from_numpy(
                                aug_mask).permute(2, 0, 1)
                        else:  # (H, W)
                            aug_mask = torch.from_numpy(aug_mask)
                        augmented_masks.append(aug_mask)

                # Stack back with time dimension
                if len(augmented_imgs) > 0:
                    output[img_key] = torch.stack(augmented_imgs, dim=0)
                    if output[img_key].shape[0] == 1 and self.config.n_obs_steps > 1:
                        output[img_key] = output[img_key].expand(
                            self.config.n_obs_steps, *output[img_key].shape[1:])
                if augmented_masks:
                    output[mask_key] = torch.stack(augmented_masks, dim=0)
                    if output[mask_key].shape[0] == 1 and self.config.n_obs_steps > 1:
                        output[mask_key] = output[mask_key].expand(
                            self.config.n_obs_steps, *output[mask_key].shape[1:])

        for output_key, output_config in self.config.output_fields.items():
            if output_key not in output and output_config.default_value is not None:
                # Add time dimension: (n_obs_steps, ...)
                shape_with_time = (self.config.n_obs_steps,
                                   ) + output_config.shape
                output[output_key] = torch.full(
                    shape_with_time, output_config.default_value, dtype=torch.float32)

        if self.config.delta_indices:
            obs_keys = [k for k in self.config.delta_indices.keys()
                        if k.startswith("observation.")]
            output['pad_o'] = (padding_masks.get(obs_keys[0], torch.zeros(len(self.config.delta_indices[obs_keys[0]]), dtype=torch.bool))
                               if obs_keys else torch.zeros(self.config.n_obs_steps, dtype=torch.bool))
            if "action_is_pad" in output:
                output['pad_a'] = output["action_is_pad"]
            elif "action" in self.config.delta_indices:
                output['pad_a'] = padding_masks.get("action", torch.zeros(
                    len(self.config.delta_indices.get("action", [])), dtype=torch.bool))
            else:
                output['pad_a'] = torch.zeros(
                    self.config.chunk_size, dtype=torch.bool)
        else:
            output['pad_o'] = torch.zeros(
                self.config.n_obs_steps, dtype=torch.bool)
            output['pad_a'] = torch.zeros(
                self.config.chunk_size, dtype=torch.bool)

        return output

    def _process_field_data(
        self, data_tensor: torch.Tensor, hdf5_field_config: "HDF5FieldConfig",
        aug_pose: np.ndarray | None
    ) -> torch.Tensor:
        """Process a single field's data: augmentation, basic conversion, and transforms."""
        data_np = data_tensor.numpy() if isinstance(
            data_tensor, torch.Tensor) else data_tensor

        if hdf5_field_config.augmentation:
            data_np = self._apply_augmentation(
                data_np, hdf5_field_config, aug_pose)

        # Normalize image data to [0, 1] range if needed
        # Check actual value range, not just dtype (augmentation may change dtype)
        if hdf5_field_config.dims == 2:
            if data_np.max() > 1.0:
                data_np = data_np / 255.0

        if hdf5_field_config.transform:
            data_np = self._apply_transform(data_np, hdf5_field_config)

        data_tensor = torch.from_numpy(data_np).float()

        if hdf5_field_config.dims == 2 and len(data_tensor.shape) == 4:
            data_tensor = data_tensor.permute(0, 3, 1, 2)

        return data_tensor


def collate_fn(batch):
    """Collate function that automatically detects tensor-like keys."""
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        sample = batch[0]
        tensor_keys = get_tensor_keys_from_data(sample)
        ret_dict = {}
        for key in sample:
            if key in tensor_keys:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]

        if 'input_coords_list' in ret_dict:
            if ME is None:
                raise ImportError(
                    "MinkowskiEngine is required for point cloud data. Install: pip install MinkowskiEngine")
            coords_batch, feats_batch = ME.utils.sparse_collate(
                ret_dict['input_coords_list'], ret_dict['input_feats_list'])
            ret_dict['input_coords_list'] = coords_batch
            ret_dict['input_feats_list'] = feats_batch

        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]

    raise TypeError(
        f"batch must contain tensors, dicts or lists; found {type(batch[0])}")
