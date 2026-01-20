import os
import cv2
import numpy as np
import re
import shutil
from r3kit.utils.sequence import find_nearest_idx
from utils import mapping, TCPPoseCalculator


def align_data_by_timestamp(
    data_path: str,
    reference: str = 'camera',
    max_time_diff_ms: float = 50.0,
    camera_name: str = 'cam0',
    save_tcp_pose: bool = True
) -> None:
    encoder_path = os.path.join(data_path, 'encoder')
    camera_path = os.path.join(data_path, 'camera')

    encoder_timestamps = np.load(os.path.join(encoder_path, 'timestamps.npy'))
    camera_timestamps = np.load(os.path.join(camera_path, 'timestamps.npy'))

    encoder_angles = np.load(os.path.join(encoder_path, 'angle.npy'))

    encoder_sort_idx = np.argsort(encoder_timestamps)
    encoder_timestamps = encoder_timestamps[encoder_sort_idx]
    encoder_angles = encoder_angles[encoder_sort_idx]

    camera_sort_idx = np.argsort(camera_timestamps)
    camera_timestamps = camera_timestamps[camera_sort_idx]

    if reference == 'camera':
        ref_timestamps = camera_timestamps
        other_timestamps = encoder_timestamps
    else:
        ref_timestamps = encoder_timestamps
        other_timestamps = camera_timestamps

    aligned_indices = []
    time_diffs = []

    for ref_ts in ref_timestamps:
        nearest_idx = find_nearest_idx(other_timestamps, ref_ts)
        time_diff = abs(ref_ts - other_timestamps[nearest_idx])

        if time_diff <= max_time_diff_ms:
            aligned_indices.append(nearest_idx)
            time_diffs.append(time_diff)
        else:
            aligned_indices.append(-1)
            time_diffs.append(time_diff)

    data_dir_name = os.path.basename(data_path)
    timestamp_match = re.search(r'(\d{8}_\d{6})', data_dir_name)
    timestamp_str = timestamp_match.group(1)
    output_dir_name = f"record_{timestamp_str}"

    parent_dir = os.path.dirname(data_path)
    output_path = os.path.join(parent_dir, output_dir_name)

    os.makedirs(output_path, exist_ok=True)

    camera_output = os.path.join(output_path, camera_name)
    angles_output = os.path.join(output_path, 'angles')
    os.makedirs(camera_output, exist_ok=True)
    os.makedirs(os.path.join(camera_output, 'color'), exist_ok=True)
    os.makedirs(angles_output, exist_ok=True)

    tcp_poses_output = None
    tcp_calculator = None
    if save_tcp_pose:
        tcp_poses_output = os.path.join(output_path, 'tcp_poses')
        os.makedirs(tcp_poses_output, exist_ok=True)
        tcp_calculator = TCPPoseCalculator()

    depth_path = os.path.join(camera_path, 'depth')
    has_depth = os.path.exists(depth_path) and len(os.listdir(depth_path)) > 0
    if has_depth:
        os.makedirs(os.path.join(camera_output, 'depth'), exist_ok=True)

    color_dir = os.path.join(camera_path, 'color')
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])

    if has_depth:
        depth_dir = os.path.join(camera_path, 'depth')
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    valid_frame_idx = 0
    for ref_idx, other_idx in enumerate(aligned_indices):
        if other_idx < 0:
            continue

        if reference == 'camera':
            angle_data = encoder_angles[other_idx].copy()
            original_camera_idx = camera_sort_idx[ref_idx]
            color_file = color_files[original_camera_idx]
            color_img = cv2.imread(os.path.join(color_dir, color_file))
            cv2.imwrite(os.path.join(camera_output, 'color', f"{valid_frame_idx:016d}.png"), color_img)

            if has_depth:
                depth_file = depth_files[original_camera_idx]
                depth_img = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(camera_output, 'depth', f"{valid_frame_idx:016d}.png"), depth_img)
        else:
            angle_data = encoder_angles[ref_idx].copy()
            original_camera_idx = camera_sort_idx[other_idx]
            color_file = color_files[original_camera_idx]
            color_img = cv2.imread(os.path.join(color_dir, color_file))
            cv2.imwrite(os.path.join(camera_output, 'color', f"{valid_frame_idx:016d}.png"), color_img)

            if has_depth:
                depth_file = depth_files[original_camera_idx]
                depth_img = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(camera_output, 'depth', f"{valid_frame_idx:016d}.png"), depth_img)

        angle_data = mapping(angle_data)

        angle_path = os.path.join(angles_output, f"angle_{valid_frame_idx:05d}.npy")
        np.save(angle_path, angle_data)

        if save_tcp_pose and tcp_calculator is not None:
            tcp_pose = tcp_calculator.compute_tcp_pose(angle_data[:7])
            tcp_data = np.concatenate([tcp_pose, angle_data[7:8]])
            tcp_path = os.path.join(tcp_poses_output, f"tcp_{valid_frame_idx:05d}.npy")
            np.save(tcp_path, tcp_data)

        valid_frame_idx += 1

    for config_file in ['intrinsics.txt', 'depth_scale.txt', 'depth2color.txt']:
        src = os.path.join(camera_path, config_file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(camera_output, config_file))

    if tcp_calculator is not None:
        del tcp_calculator

    print(f"\nAligned data saved to: {output_path}")
    print(f"  Total aligned samples: {valid_frame_idx}")
    if save_tcp_pose:
        print(f"  TCP poses saved to: {tcp_poses_output}")


def align_batch_data_by_timestamp(
    data_root: str,
    reference: str = 'camera',
    max_time_diff_ms: float = 50.0,
    camera_name: str = 'cam0',
    save_tcp_pose: bool = True
) -> None:
    if not os.path.exists(data_root):
        raise ValueError(f"Data root path not found: {data_root}")

    data_dirs = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            encoder_path = os.path.join(item_path, 'encoder')
            camera_path = os.path.join(item_path, 'camera')
            if os.path.exists(encoder_path) and os.path.exists(camera_path):
                if not item.startswith('record_'):
                    data_dirs.append(item_path)

    for i, data_dir in enumerate(data_dirs, 1):
        print(f"\n[{i}/{len(data_dirs)}] Processing: {os.path.basename(data_dir)}")

        align_data_by_timestamp(
            data_dir,
            reference=reference,
            max_time_diff_ms=max_time_diff_ms,
            camera_name=camera_name,
            save_tcp_pose=save_tcp_pose
        )

    print(f"\nBatch processing completed!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Align encoder and camera data by timestamp')
    parser.add_argument('--data_path', type=str, default='/media/ubuntu/B0A8C06FA8C0361E/Data/Origin_Data/records_1204_test')
    parser.add_argument('--reference', type=str, default='camera', choices=['camera', 'encoder'])
    parser.add_argument('--max_time_diff', type=float, default=50.0)
    parser.add_argument('--camera_name', type=str, default='cam_104122061018')
    parser.add_argument('--batch', action='store_true', default=True)
    parser.add_argument('--save_tcp_pose', action='store_true', default=True)

    args = parser.parse_args()

    if args.batch:
        align_batch_data_by_timestamp(
            data_root=args.data_path,
            reference=args.reference,
            max_time_diff_ms=args.max_time_diff,
            camera_name=args.camera_name,
            save_tcp_pose=args.save_tcp_pose
        )
    else:
        align_data_by_timestamp(
            data_path=args.data_path,
            reference=args.reference,
            max_time_diff_ms=args.max_time_diff,
            camera_name=args.camera_name,
            save_tcp_pose=args.save_tcp_pose
        )
