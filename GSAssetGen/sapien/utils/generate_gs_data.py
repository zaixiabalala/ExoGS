import os
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from utils.image_utils import (
    change_image_background,
    generate_point_cloud,
    static_view_merge_pcd_data,
    voxel_sample_points
)
from utils.camera_utils import extract_camera_params

# Get the parent directory of current file, i.e., the sapien folder
current_dir = os.path.dirname(os.path.abspath(__file__))


def convert_gs_data(point_num, downsample_num, dynamic, camera_height, camera_width,
                     raw_data_dir, gs_data_dir):
    if os.path.exists(gs_data_dir):
        shutil.rmtree(gs_data_dir)
    os.makedirs(gs_data_dir, exist_ok=True)

    npz_files = glob.glob('{}/*npz'.format(raw_data_dir))

    for npz_file in npz_files:
        data_name = (f"{npz_file.split('/')[-1].split('.')[0].split('_')[0]}_"
                     f"{npz_file.split('/')[-1].split('.')[0].split('_')[1]}")
        npz_data = np.load(npz_file, allow_pickle=True)
        rgb = npz_data['rgb']  # [H, W, 3] numpy array
        depth = npz_data['depth']  # [H, W] numpy array
        part_mask = npz_data['part_mask']  # [H, W] numpy array
        intrinsic_matrix = npz_data['intrinsic_matrix']  # [3, 3] numpy array
        extrinsic_matrix = npz_data['extrinsic_matrix']  # [4, 4] numpy array

        # Check if part_mask contains objects other than background
        if np.unique(part_mask).shape[0] < 2:
            continue  # Skip if only background exists

        # Remove background
        no_background_image, depth_map = change_image_background(
            part_mask=part_mask,
            rgb=rgb,
            depth=depth,
            part_mask_id_min=2,
            image_show=False
        )

        # Generate point cloud
        world_point_cloud, per_point_rgb, _ = generate_point_cloud(
            rgb=rgb,
            depth=depth,
            part_mask=part_mask,
            extrinsic_matrix=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
            part_mask_id_min=2,
            view_cloud=False
        )

        # Downsample point cloud
        down_sampled_world_point_cloud, index_org = voxel_sample_points(
            points=world_point_cloud,
            point_number=point_num
        )

        # Get RGB color values of downsampled point cloud
        down_sampled_per_point_rgbs = per_point_rgb[index_org]
        # Check if the shape of RGB color values of downsampled point cloud is correct
        assert down_sampled_world_point_cloud.shape[0] == down_sampled_per_point_rgbs.shape[0]

        # Save data
        os.makedirs(gs_data_dir, exist_ok=True)
        os.makedirs(gs_data_dir + "/images", exist_ok=True)
        os.makedirs(gs_data_dir + "/depths", exist_ok=True)
        os.makedirs(gs_data_dir + "/infos", exist_ok=True)

        image = Image.fromarray(no_background_image)
        image.save(os.path.join(gs_data_dir, "images", f"{data_name}_rgb.png"))

        np.save(os.path.join(gs_data_dir, "depths", f"{data_name}.npy"), depth_map)

        np.savez(os.path.join(gs_data_dir, "infos", f"{data_name}.npz"),
                 per_point_coords=down_sampled_world_point_cloud,
                 per_point_rgbs=down_sampled_per_point_rgbs,
                 camera2world_matrix=extrinsic_matrix,
                 camera_intrinsic=intrinsic_matrix,
                 no_background_image=no_background_image,
                 depth_map=depth_map)

        if 'robot_qpos' in npz_data:
            os.makedirs(gs_data_dir + "/joint_states", exist_ok=True)
            joint_state = npz_data['robot_qpos']
            np.save(os.path.join(gs_data_dir, "joint_states", f"{data_name}.npy"), joint_state)

        print("finish save file: ", os.path.join(gs_data_dir, f"{data_name}"))

    # If static scene, merge point clouds
    if int(dynamic) == 0:
        static_view_merge_pcd_data(
            file_path=gs_data_dir + "/infos/",
            save_path=gs_data_dir,
            downsample_number=downsample_num
        )

    extract_camera_params(
        file_path=gs_data_dir + "/infos/",
        save_path=gs_data_dir,
        height=camera_height,
        width=camera_width
    )

    print("==========Finished!==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_name", type=str, default=None, help="Object name")
    parser.add_argument("--sapien_data_dir", type=str, default=None, help="SAPien data path")
    args = parser.parse_args()

    #####################################################
    obj_name = "dumbbell"
    point_num = 10000  # Number of points in point cloud
    downsample_num = 50000  # Downsample number
    dynamic = 0  # Whether it is a dynamic scene
    camera_height = 800  # Camera height
    camera_width = 800  # Camera width
    sapien_data_dir = "/media/robotflow/SSDWH/3DGS/Data/GS_Data/sapien_data"
    #####################################################

    if args.obj_name is not None:
        obj_name = args.obj_name
    if args.sapien_data_dir is not None:
        sapien_data_dir = args.sapien_data_dir

    raw_data_dir = os.path.join(sapien_data_dir, f"data_0/{obj_name}/raw")  # Raw data path
    gs_data_dir = os.path.join(sapien_data_dir, f"data_0/{obj_name}/gs")  # Generated data path

    convert_gs_data(point_num, downsample_num, dynamic, camera_height, camera_width,
                    raw_data_dir, gs_data_dir)
                    