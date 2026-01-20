import os
import math
import glob
import sapien
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from utils.transform import matrix_to_pose
from utils.image_utils import generate_point_cloud


def extract_camera_params(file_path, save_path, height=800, width=800):
    """
    Extract camera parameters and generate COLMAP format data.

    Args:
        file_path: str, path to npz data (raw_data_dir/npzs)
        save_path: str, save path
        height: int, image height
        width: int, image width
    """
    os.makedirs(os.path.join(save_path, "sparse", "0"), exist_ok=True)
    files = glob.glob('{}/*npz'.format(file_path))
    if len(files) == 0:
        raise ValueError(f"[Error] No npz files found in {file_path}")

    # File name format: {phi}_{theta}_data.npz, sort by phi and theta
    def sort_key(file_path):
        filename = file_path.split("/")[-1].replace("_data.npz", "").replace(".npz", "")
        parts = filename.split("_")
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))  # (phi, theta)
        else:
            return (0, 0)  # If format is incorrect, put it at the front

    files = sorted(files, key=sort_key)
    camera_intrinsics = []
    camera_extrinsics = []
    view_ids = []

    tqdm.write(f"[Info] Starting to generate COLMAP dataset, {len(files)} files in total")

    # Load image width and height
    pbar_files = tqdm(files, desc="Loading camera parameters", unit="file")
    for file_path_item in pbar_files:
        data = np.load(file_path_item, allow_pickle=True)

        # Read data from raw_data_dir/npzs (using intrinsic_matrix and extrinsic_matrix)
        if 'intrinsic_matrix' not in data:
            raise ValueError(f"[Error] Missing camera intrinsic (intrinsic_matrix) in file {file_path_item}")

        if 'extrinsic_matrix' not in data:
            raise ValueError(f"[Error] Missing camera extrinsic (extrinsic_matrix) in file {file_path_item}")

        K = np.array(data['intrinsic_matrix']).reshape(3, 3)
        cam2world = np.array(data['extrinsic_matrix'])

        R_world2camera = cam2world[:3, :3]
        R_camera2world = R_world2camera.T
        t_camera2world = cam2world[:3, 3].reshape(1, 3)
        t_world2camera = (-R_camera2world @ t_camera2world.T).reshape(1, 3)
        camera_intrinsics.append(K)
        camera_extrinsics.append((R_camera2world, t_world2camera))

        # File name format: {phi}_{theta}_data.npz
        filename = file_path_item.split("/")[-1].replace("_data.npz", "").replace(".npz", "")
        parts = filename.split("_")
        if len(parts) >= 2:
            view_step = parts[0]  # phi
            camera_id = parts[1]  # theta
        else:
            # If format is incorrect, use default values
            view_step = "0"
            camera_id = "0"
        view_id = f"{view_step}_{camera_id}"
        view_ids.append(view_id)
        pbar_files.set_description(f"Loading camera parameters {view_id}")

    pbar_files.close()

    if len(camera_intrinsics) == 0:
        raise ValueError("[Error] Failed to load any valid camera parameters")

    # Check if all camera intrinsics are the same
    all_same = True
    first_K = camera_intrinsics[0]
    for K in camera_intrinsics[1:]:
        if not np.allclose(K, first_K, atol=1e-6):
            all_same = False
            break

    with open(os.path.join(save_path, "sparse", "0", "cameras.txt"), 'w') as cam_file:
        cam_file.write('# Camera list with one line of data per camera:\n')
        cam_file.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')

        if all_same:
            # All camera intrinsics are the same, write only one camera
            cam_file.write('# Number of cameras: 1\n')
            K = camera_intrinsics[0]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            cam_file.write('1 PINHOLE {} {} {} {} {} {}\n'.format(width, height, fx, fy, cx, cy))
            # Create camera ID mapping: all images use camera ID 1
            camera_id_mapping = {i: 1 for i in range(len(camera_intrinsics))}
        else:
            # Each camera has different intrinsics, create independent ID for each camera
            cam_file.write('# Number of cameras: {}\n'.format(len(camera_intrinsics)))
            camera_id_mapping = {}
            for idx, K in enumerate(camera_intrinsics):
                camera_id = idx + 1
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                cam_file.write('{} PINHOLE {} {} {} {} {} {}\n'.format(
                    camera_id, width, height, fx, fy, cx, cy))
                camera_id_mapping[idx] = camera_id
            tqdm.write(f"[Info] Detected {len(camera_intrinsics)} different camera intrinsics, "
                       f"creating independent ID for each camera")

    # Generate point cloud data for points3D.txt
    tqdm.write("[Info] Starting to generate point cloud and COLMAP format data...")
    all_points3d = []
    all_point_colors = []
    point_id_counter = 1
    image_point_mapping = {}  # Record point IDs corresponding to each image

    with open(os.path.join(save_path, "sparse", "0", "images.txt"), 'w') as img_file:
        # Save camera extrinsics
        img_file.write('# Image list with two lines of data per image:\n')
        img_file.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        img_file.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        img_file.write('# Number of images: {}\n'.format(len(camera_extrinsics)))
        camera_num = 0

        pbar_process = tqdm(enumerate(zip(files, view_ids, camera_extrinsics)),
                            total=len(files),
                            desc="Generating COLMAP data",
                            unit="image")

        for idx, (file_path_item, view_id, (r, t)) in pbar_process:
            pbar_process.set_description(f"Processing image {view_id} ({idx+1}/{len(files)})")
            qx, qy, qz, qw = Rotation.from_matrix(r).as_quat()
            tx, ty, tz = t[0]

            # Load data from npz file to generate point cloud
            data = np.load(file_path_item, allow_pickle=True)

            # Check required fields
            if 'rgb' not in data:
                raise ValueError(f"[Error] Missing rgb field in file {file_path_item}")
            if 'depth' not in data:
                raise ValueError(f"[Error] Missing depth field in file {file_path_item}")
            if 'part_mask' not in data:
                raise ValueError(f"[Error] Missing part_mask field in file {file_path_item}")
            if 'intrinsic_matrix' not in data:
                raise ValueError(f"[Error] Missing intrinsic_matrix field in file {file_path_item}")
            if 'extrinsic_matrix' not in data:
                raise ValueError(f"[Error] Missing extrinsic_matrix field in file {file_path_item}")

            # Generate point cloud data
            rgb = data['rgb']
            depth = data['depth']
            part_mask = data['part_mask']
            extrinsic_matrix = data['extrinsic_matrix']
            intrinsic_matrix = data['intrinsic_matrix']

            # Generate point cloud
            world_point_cloud, per_point_rgb, per_point_idx = generate_point_cloud(
                rgb=rgb,
                depth=depth,
                part_mask=part_mask,
                extrinsic_matrix=extrinsic_matrix,
                intrinsic_matrix=intrinsic_matrix,
                part_mask_id_min=2,
                view_cloud=False
            )

            # Check if point cloud is empty
            if len(world_point_cloud) == 0:
                raise ValueError(f"[Error] Point cloud generated for image {view_id} is empty")

            # Sample point cloud, limit to maximum 1000 points per image to avoid excessive data
            max_points_per_image = 1000
            if len(world_point_cloud) > max_points_per_image:
                indices = np.random.choice(len(world_point_cloud), max_points_per_image, replace=False)
                world_point_cloud = world_point_cloud[indices]
                per_point_rgb = per_point_rgb[indices]
                per_point_idx = per_point_idx[indices]

            # Record point cloud and corresponding image points
            # per_point_idx format: [N, 2] where [:, 0] is y coordinate, [:, 1] is x coordinate
            image_point_ids = []
            for i, (point_3d, point_rgb, point_idx) in enumerate(zip(world_point_cloud, per_point_rgb, per_point_idx)):
                all_points3d.append(point_3d)
                all_point_colors.append(point_rgb)
                # COLMAP format: (X, Y, POINT3D_ID), note that per_point_idx is [y, x]
                image_point_ids.append((point_idx[1], point_idx[0], point_id_counter))  # (x, y, point3d_id)
                point_id_counter += 1

            image_point_mapping[camera_num + 1] = image_point_ids

            # Get the camera ID corresponding to this image
            camera_id = camera_id_mapping[idx]

            # Write image information
            img_file.write('{} {} {} {} {} {} {} {} {} {}\n'.format(
                camera_num + 1, qw, qx, qy, qz, tx, ty, tz, camera_id, f'{view_id}_rgb.png'))

            # Write 2D feature points (required, otherwise error)
            if camera_num + 1 not in image_point_mapping:
                raise ValueError(f"[Error] Point cloud mapping missing for image {view_id}")

            points2d_list = image_point_mapping[camera_num + 1]
            if len(points2d_list) == 0:
                raise ValueError(f"[Error] No 2D feature points generated for image {view_id}")

            points2d_str = ' '.join([f'{x} {y} {pid}' for x, y, pid in points2d_list])
            img_file.write(f'{points2d_str}\n')

            camera_num += 1

        pbar_process.close()

    # Generate points3D.txt
    tqdm.write(f"[Info] Generating points3D.txt, {len(all_points3d)} 3D points in total...")
    with open(os.path.join(save_path, "sparse", "0", "points3D.txt"), 'w') as points_file:
        points_file.write('# 3D point list with one line of data per point:\n')
        points_file.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        points_file.write('# Number of points: {}\n'.format(len(all_points3d)))

        # Build mapping from point ID to image ID (track information)
        point_to_images = {}
        for img_id, point_list in image_point_mapping.items():
            for local_idx, (x, y, point_id) in enumerate(point_list):
                if point_id not in point_to_images:
                    point_to_images[point_id] = []
                # track format: (IMAGE_ID, POINT2D_IDX), POINT2D_IDX is the index in the points2d list of that image
                point_to_images[point_id].append((img_id, local_idx))

        # Write point cloud data
        pbar_points = tqdm(range(1, point_id_counter),
                            desc="Writing point cloud data",
                            unit="point",
                            total=len(all_points3d))
        for point_id in pbar_points:
            if point_id <= len(all_points3d):
                point_3d = all_points3d[point_id - 1]
                point_color = all_point_colors[point_id - 1]
                x, y, z = point_3d
                r, g, b = (point_color * 255).astype(np.uint8)

                # Build track information
                if point_id in point_to_images:
                    track_str = ' '.join([f'{img_id} {idx}' for img_id, idx in point_to_images[point_id]])
                else:
                    track_str = ''

                points_file.write('{} {} {} {} {} {} {} 1.0 {}\n'.format(
                    point_id, x, y, z, r, g, b, track_str))

        pbar_points.close()

    tqdm.write(f"[Info] COLMAP dataset generation completed!")
    tqdm.write(f"  - cameras.txt: {len(camera_intrinsics) if not all_same else 1} cameras")
    tqdm.write(f"  - images.txt: {len(camera_extrinsics)} images")
    tqdm.write(f"  - points3D.txt: {len(all_points3d)} 3D points")
    tqdm.write(f"  Save path: {os.path.join(save_path, 'sparse', '0')}")


def get_cam_pos_fix(theta, phi, distance):
    x = math.sin(math.pi / 180 * theta) * math.cos(math.pi / 180 * phi) * distance
    y = math.sin(math.pi / 180 * theta) * math.sin(math.pi / 180 * phi) * distance
    z = math.cos(math.pi / 180 * theta) * distance
    return np.array([x, y, z])


def get_camera_pos_mat(camera):
    Rtilt = camera.get_model_matrix()
    Rtilt_rot = Rtilt[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_trl = Rtilt[:3, 3]
    cam2_wolrd = np.eye(4)
    cam2_wolrd[:3, :3] = Rtilt_rot
    cam2_wolrd[:3, 3] = Rtilt_trl
    return cam2_wolrd
