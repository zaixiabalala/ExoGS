import os
import re
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict

def xyzquat2mat(xyz:np.ndarray=np.zeros(3), quat:np.ndarray=np.array([0., 0., 0., 1.])) -> np.ndarray:
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = Rot.from_quat(quat).as_matrix()
    pose_4x4[:3, 3] = xyz
    return pose_4x4


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def load_mask(mask_path: str) -> np.ndarray:
    img = Image.open(mask_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def load_joint(joint_path: str) -> np.ndarray:
    return np.load(joint_path)


def load_tcp(tcp_path: str) -> np.ndarray:
    return np.load(tcp_path)


def convert_tcp_format(tcp_data: np.ndarray) -> np.ndarray:
    """Convert TCP from [xyz, quat, gripper] to [xyz, rot_col1, rot_col2, gripper]."""
    xyz = tcp_data[:3]
    quat = tcp_data[3:7]
    gripper_width = tcp_data[7]
    transformation_matrix = xyzquat2mat(xyz, quat)
    rotation_matrix = transformation_matrix[:3, :3]
    first_column = rotation_matrix[:, 0]
    second_column = rotation_matrix[:, 1]
    new_tcp_data = np.concatenate([xyz, first_column, second_column, [gripper_width]])
    return new_tcp_data


class ImageProcessor:
    def __init__(self, **params):
        self.params = params

    def process(self, image: np.ndarray) -> np.ndarray:
        if 'crop' in self.params:
            crop = self.params['crop']
            image = image[crop['start_y']:crop['end_y'],
                         crop['start_x']:crop['end_x'], :]

        if self.params.get('center_crop_square', False):
            H, W = image.shape[:2]
            min_dim = min(W, H)
            start_x = (W - min_dim) // 2
            start_y = (H - min_dim) // 2
            image = image[start_y:start_y+min_dim, start_x:start_x+min_dim, :]

        if 'border' in self.params:
            border = self.params['border']
            image = self.add_border(image, border['top'], border['bottom'],
                                  border['left'], border['right'])

        if 'resize' in self.params:
            image = self.resize_image(image, self.params['resize'], is_mask=False)

        if 'mask_resize' in self.params:
            image = self.resize_image(image, self.params['mask_resize'], is_mask=True)

        if self.params.get('bgr2rgb', False):
            image = image[:, :, ::-1]

        return image

    def add_border(self, image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        H, W, C = image.shape
        new_H = H + top + bottom
        new_W = W + left + right
        bordered = np.zeros((new_H, new_W, C), dtype=image.dtype)
        bordered[top:top+H, left:left+W, :] = image
        return bordered

    def resize_image(self, image: np.ndarray, target_size: tuple, is_mask: bool = False) -> np.ndarray:
        if is_mask:
            img_pil = Image.fromarray(image)
            img_resized = img_pil.resize(target_size, Image.Resampling.NEAREST)
            return np.array(img_resized)
        else:
            img_pil = Image.fromarray(image)
            img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
            return np.array(img_resized)


def find_rgb_dirs(traj_path: Path, base_dir: str = "cam_0", rgb_selection_mode: str = "all") -> List[str]:
    base_path = traj_path / base_dir
    if not base_path.exists():
        return []

    rgb_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('rgb'):
            if rgb_selection_mode == 'no_color_aug':
                if item.name.startswith('rgbs') and item.name.endswith('bg0'):
                    rgb_dirs.append(item.name)
            elif rgb_selection_mode == 'all':
                rgb_dirs.append(item.name)
            elif rgb_selection_mode == 'bg0_only':
                if item.name.endswith('bg0'):
                    rgb_dirs.append(item.name)
            else:
                rgb_dirs.append(item.name)

    def sort_key(name: str) -> tuple:
        if name == 'rgbs':
            return (0, '')
        parts = name.split('_')
        if len(parts) > 1:
            try:
                num = int(parts[-1])
                return (1, num)
            except ValueError:
                return (2, name)
        return (2, name)

    rgb_dirs = list(dict.fromkeys(rgb_dirs))
    rgb_dirs.sort(key=sort_key)
    return rgb_dirs


def convert_to_hdf5(
    trajectory_dirs: List[str],
    output_path: str,
    image_base_dir: str = "cam_0",
    mask_subdir: str = "cam_0/masks_clean_labels_3",
    joint_subdir: str = "angles",
    tcp_subdir: str = "tcps",
    image_pattern: str = "*.png",
    mask_pattern: str = "*.png",
    joint_pattern: str = "*.npy",
    tcp_pattern: str = "*.npy",
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
    image_processor_params: Optional[Dict[str, Dict]] = None,
    mask_processor_params: Optional[Dict[str, Dict]] = None,
    exclude_trajectories: Optional[List[str]] = None,
    rgb_selection_mode: str = "all"
):
    """Convert rendered 3DGS trajectories to HDF5 format for training."""
    image_processor = ImageProcessor(**(image_processor_params or {}))
    mask_processor = ImageProcessor(**(mask_processor_params or {}))

    with h5py.File(output_path, 'w') as f:
        data_group = f.create_group('data')
        demo_idx = 0
        exclude_list = exclude_trajectories or []

        for traj_idx, traj_dir in enumerate(trajectory_dirs):
            traj_path = Path(traj_dir)
            if not traj_path.exists():
                print(f"Warning: skipping non-existent path: {traj_dir}")
                continue

            if traj_path.name in exclude_list:
                print(f"Skipping excluded trajectory: {traj_path.name}")
                continue

            print(f"\nProcessing trajectory {traj_idx + 1}/{len(trajectory_dirs)}: {traj_path.name}")

            rgb_dirs = find_rgb_dirs(traj_path, image_base_dir, rgb_selection_mode)
            if len(rgb_dirs) == 0:
                print(f"  Warning: no rgb directories found, skipping")
                continue

            print(f"  Found {len(rgb_dirs)} color versions: {rgb_dirs}")

            mask_dir = traj_path / mask_subdir
            joint_dir = traj_path / joint_subdir
            tcp_dir = traj_path / tcp_subdir

            if not mask_dir.exists():
                print(f"  Warning: mask directory not found: {mask_dir}")
                continue
            if not joint_dir.exists():
                print(f"  Warning: joint directory not found: {joint_dir}")
                continue
            if not tcp_dir.exists():
                print(f"  Warning: tcp directory not found: {tcp_dir}")
                continue

            mask_files = sorted(mask_dir.glob(mask_pattern))
            joint_files = sorted(joint_dir.glob(joint_pattern))
            tcp_files = sorted(tcp_dir.glob(tcp_pattern))

            def extract_frame_number(file_path: Path) -> int:
                try:
                    stem = file_path.stem
                    match = re.search(r'(\d+)', stem)
                    if match:
                        return int(match.group(1))
                    else:
                        return -1
                except (ValueError, AttributeError):
                    return -1

            mask_dict = {extract_frame_number(f): f for f in mask_files}
            joint_dict = {extract_frame_number(f): f for f in joint_files}
            tcp_dict = {extract_frame_number(f): f for f in tcp_files}

            for color_idx, rgb_dir_name in enumerate(rgb_dirs):
                image_dir = traj_path / image_base_dir / rgb_dir_name

                if not image_dir.exists():
                    print(f"  Warning: image directory not found: {image_dir}, skipping")
                    continue

                print(f"  Processing color version {color_idx + 1}/{len(rgb_dirs)}: {rgb_dir_name}")

                image_files = sorted(image_dir.glob(image_pattern))
                image_dict = {extract_frame_number(f): f for f in image_files}

                common_frames = sorted(set(image_dict.keys()) & set(mask_dict.keys()) & set(joint_dict.keys()) & set(tcp_dict.keys()))
                common_frames = [f for f in common_frames if f >= 0]

                if len(common_frames) == 0:
                    print(f"    Warning: no matching frames found, skipping")
                    continue

                image_files = [image_dict[f] for f in common_frames]
                mask_files_sorted = [mask_dict[f] for f in common_frames]
                joint_files_sorted = [joint_dict[f] for f in common_frames]
                tcp_files_sorted = [tcp_dict[f] for f in common_frames]

                print(f"    Found {len(image_files)} matching frames")

                demo_group = data_group.create_group(f'demo_{demo_idx}')

                first_image = load_image(str(image_files[0]))
                first_mask = load_mask(str(mask_files_sorted[0]))
                first_joint = load_joint(str(joint_files_sorted[0]))
                first_tcp = load_tcp(str(tcp_files_sorted[0]))
                first_tcp = convert_tcp_format(first_tcp)

                if image_processor_params:
                    first_image = image_processor.process(first_image)
                if mask_processor_params:
                    first_mask = mask_processor.process(first_mask)

                num_frames = len(image_files)

                images_ds = demo_group.create_dataset(
                    'images',
                    shape=(num_frames, *first_image.shape),
                    dtype=np.uint8,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

                masks_ds = demo_group.create_dataset(
                    'masks',
                    shape=(num_frames, *first_mask.shape),
                    dtype=np.uint8,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

                joints_ds = demo_group.create_dataset(
                    'joints',
                    shape=(num_frames, *first_joint.shape),
                    dtype=first_joint.dtype,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

                tcps_ds = demo_group.create_dataset(
                    'tcps',
                    shape=(num_frames, *first_tcp.shape),
                    dtype=first_tcp.dtype,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

                for frame_idx, (img_path, mask_path, joint_path, tcp_path) in enumerate(zip(image_files, mask_files_sorted, joint_files_sorted, tcp_files_sorted)):
                    image_data = load_image(str(img_path))
                    mask_data = load_mask(str(mask_path))
                    joint_data = load_joint(str(joint_path))
                    tcp_data = load_tcp(str(tcp_path))
                    tcp_data = convert_tcp_format(tcp_data)

                    if image_processor_params:
                        image_data = image_processor.process(image_data)
                    if mask_processor_params:
                        mask_data = mask_processor.process(mask_data)

                    images_ds[frame_idx] = image_data
                    masks_ds[frame_idx] = mask_data
                    joints_ds[frame_idx] = joint_data
                    tcps_ds[frame_idx] = tcp_data

                    if (frame_idx + 1) % 100 == 0:
                        print(f"      Processed {frame_idx + 1}/{num_frames} frames")

                demo_group.attrs['num_frames'] = num_frames
                demo_group.attrs['image_shape'] = first_image.shape
                demo_group.attrs['mask_shape'] = first_mask.shape
                demo_group.attrs['joint_shape'] = first_joint.shape
                demo_group.attrs['tcp_shape'] = first_tcp.shape
                demo_group.attrs['trajectory_name'] = traj_path.name
                demo_group.attrs['color_version'] = rgb_dir_name

                print(f"    Done: {num_frames} frames (demo_{demo_idx})")
                demo_idx += 1

    print(f"\nConversion complete! Output: {output_path}")
    print(f"Total demos created: {demo_idx}")


if __name__ == '__main__':
    base_dir = Path("/media/ubuntu/B0A8C06FA8C0361E/Data/GS_Data/records_unscrew_cap_1221")
    trajectory_dirs = sorted([str(p) for p in base_dir.iterdir() if p.is_dir() and p.name.startswith('record_')])

    output_file = "/home/ubuntu/data/Train_Data/records_unscrew_cap_1221.hdf5"

    image_base_dir = "cam_0"
    mask_subdir = "cam_0/masks_clean_labels_3"
    joint_subdir = "angles"
    tcp_subdir = "tcps"

    image_pattern = "*.png"
    mask_pattern = "*.png"
    joint_pattern = "*.npy"
    tcp_pattern = "*.npy"

    image_processor_params = {
        'crop': {'start_x': 100, 'end_x': 580, 'start_y': 0, 'end_y': 480},
        'resize': (224, 224),
    }

    mask_processor_params = {
        'crop': {'start_x': 100, 'end_x': 580, 'start_y': 0, 'end_y': 480},
        'mask_resize': (224, 224),
    }

    convert_to_hdf5(
        trajectory_dirs=trajectory_dirs,
        output_path=output_file,
        image_base_dir=image_base_dir,
        mask_subdir=mask_subdir,
        joint_subdir=joint_subdir,
        tcp_subdir=tcp_subdir,
        image_pattern=image_pattern,
        mask_pattern=mask_pattern,
        joint_pattern=joint_pattern,
        tcp_pattern=tcp_pattern,
        image_processor_params=image_processor_params,
        mask_processor_params=mask_processor_params,
        rgb_selection_mode="all",
    )
