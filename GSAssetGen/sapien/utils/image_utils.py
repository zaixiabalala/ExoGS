import os
import copy
import glob
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def change_image_background(part_mask, rgb, depth, part_mask_id_min=2, image_show=False,
                            background_color=[0.0, 0.0, 0.0]):
    valid_mask = part_mask >= part_mask_id_min
    no_background_image = copy.deepcopy(rgb)
    no_background_image[~valid_mask] = np.array(background_color)
    depth[~valid_mask] = 0.0
    if no_background_image.max() > 1.0:
        no_background_image = no_background_image / 255.0
    no_background_image = np.clip(no_background_image, 0.0, 1.0)
    if image_show:
        plt.imshow(no_background_image)
        plt.show()
    return (no_background_image * 255).astype(np.uint8), depth


def generate_point_cloud(rgb, depth, part_mask, extrinsic_matrix, intrinsic_matrix,
                         part_mask_id_min=2, view_cloud=True):
    """
    Generate point cloud.

    Args:
        rgb: numpy.ndarray, [H, W, 3]
        depth: numpy.ndarray, [H, W]
        part_mask: numpy.ndarray, [H, W]
        extrinsic_matrix: numpy.ndarray, [4, 4]
        intrinsic_matrix: numpy.ndarray, [3, 3]
        part_mask_id_min: int, minimum segmentation id
        view_cloud: bool, whether to visualize point cloud

    Returns:
        world_point_cloud: numpy.ndarray, [N, 3]
            World coordinate 3D point cloud coordinates
        per_point_rgb: numpy.ndarray, [N, 3]
            RGB color values corresponding to each point
        per_point_idx: numpy.ndarray, [N, 2]
            Pixel indices of each point in the original image
    """
    width, height = rgb.shape[0], rgb.shape[1]
    K = np.array(intrinsic_matrix).reshape(3, 3)
    y_coords, x_coords = np.indices((width, height))
    z_new = depth.astype(float)
    valid_mask = part_mask >= part_mask_id_min
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    z_new = z_new[valid_mask]
    x_new = (x_coords - K[0, 2]) * z_new / K[0, 0]
    y_new = (y_coords - K[1, 2]) * z_new / K[1, 1]
    point_cloud = np.stack((x_new, y_new, z_new), axis=-1)
    per_point_rgb = rgb[y_coords, x_coords]
    per_point_idx = np.stack((y_coords, x_coords), axis=-1)
    camera_point_cloud = np.array(point_cloud)
    per_point_rgb = np.array(per_point_rgb)
    per_point_idx = np.array(per_point_idx)

    if per_point_rgb.max() > 1.0:
        per_point_rgb = per_point_rgb / 255.0

    per_point_rgb = np.clip(per_point_rgb, 0.0, 1.0)

    world_point_cloud = point_cloud_camera_to_world(camera_point_cloud, extrinsic_matrix)
    if view_cloud:
        view_point_clouds(camera_point_cloud, per_point_rgb)
        view_point_clouds(world_point_cloud, per_point_rgb)
        view_point_clouds(world_point_cloud, per_point_rgb, [extrinsic_matrix])

    return world_point_cloud, per_point_rgb, per_point_idx


def view_point_clouds(point_cloud, rgbs=None, cam2world_matrixs=None):
    """
    Visualize point cloud.

    Args:
        point_cloud: numpy.ndarray, [N, 3]
        rgbs: numpy.ndarray, [N, 3]
        cam2world_matrixs: numpy.ndarray, [N, 4, 4]
    """
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    if rgbs is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgbs)
    if cam2world_matrixs is not None:
        lines_pcds = []
        for camera_pose in cam2world_matrixs:
            polygon_points = np.array(
                [camera_pose[:3, 3].tolist(),
                 (camera_pose[:3, 3] + camera_pose[:3, 0] * 0.15).tolist()])
            lines = [[0, 1]]
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector([np.array([1, 0, 0])])
            lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
            lines_pcds.append(lines_pcd)
            polygon_points = np.array(
                [camera_pose[:3, 3].tolist(),
                 (camera_pose[:3, 3] + camera_pose[:3, 1] * 0.15).tolist()])
            lines = [[0, 1]]
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector([np.array([0, 1, 0])])
            lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
            lines_pcds.append(lines_pcd)
            polygon_points = np.array(
                [camera_pose[:3, 3].tolist(),
                 (camera_pose[:3, 3] + camera_pose[:3, 2] * 0.15).tolist()])
            lines = [[0, 1]]
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector([np.array([0, 0, 1])])
            lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
            lines_pcds.append(lines_pcd)
        o3d.visualization.draw_geometries([cloud, *lines_pcds, axis_pcd])
    else:
        o3d.visualization.draw_geometries([cloud, axis_pcd])


def point_cloud_camera_to_world(point_cloud, extrinsic):
    """
    Convert point cloud from camera coordinate system to world coordinate system.

    Args:
        point_cloud: numpy.ndarray, [N, 3]
        extrinsic: numpy.ndarray, [4, 4]

    Returns:
        point_cloud: numpy.ndarray, [N, 3]
    """
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    point_cloud = (R @ point_cloud.T).T + T
    return point_cloud


def voxel_sample_points(points, method='voxel', point_number=4096, voxel_size=0.0005):
    """
    Downsample point cloud.

    Args:
        points: numpy.ndarray, [N, 3]
        method: 'voxel'/'random'
        point_number: output point number
        voxel_size: grid size used in voxel_down_sample

    Returns:
        points: numpy.ndarray, [N, 3]
            Downsampled point cloud
        index_org: numpy.ndarray, [N]
            Indices of downsampled point cloud in original point cloud
    """
    assert (method in ['voxel', 'random'])
    if method == 'voxel':
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud, trace, _ = cloud.voxel_down_sample_and_trace(
            voxel_size=voxel_size,
            min_bound=cloud.get_min_bound() + 1,
            max_bound=cloud.get_max_bound() + 1)
        to_index_org = np.max(trace, 1)
        points = np.array(cloud.points)
    if len(points) >= point_number:
        idxs = np.random.choice(len(points), point_number, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), point_number - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2])
    points = points[idxs]
    index_org = to_index_org[idxs]
    return points, index_org


def static_view_merge_pcd_data(file_path, save_path, downsample_number=50000):
    """
    Merge point cloud data.

    Args:
        file_path: str, point cloud data path
        save_path: str, save path
        downsample_number: int, number of downsampled points

    Returns:
        world_point_clouds: numpy.ndarray, [N, 3]
        point_rgbs: numpy.ndarray, [N, 3]
    """
    pcd_data_paths = glob.glob('{}/*npz'.format(file_path))
    world_point_clouds = []
    point_rgbs = []
    for data_path in pcd_data_paths:
        file = np.load(data_path, allow_pickle=True)
        per_point_coords = file["per_point_coords"]
        per_point_rgbs = file["per_point_rgbs"]
        world_point_clouds.append(per_point_coords)
        point_rgbs.append(per_point_rgbs)
    world_point_clouds = np.concatenate(world_point_clouds, axis=0)
    point_rgbs = np.concatenate(point_rgbs, axis=0)

    if point_rgbs.max() > 1.0:
        point_rgbs = point_rgbs / 255.0

    point_rgbs = np.clip(point_rgbs, 0.0, 1.0)

    if downsample_number > 0:
        world_point_clouds, index_org = voxel_sample_points(world_point_clouds,
                                                             point_number=downsample_number)
        point_rgbs = point_rgbs[index_org]
    ply_dir = os.path.join(save_path, "sparse", "0")
    os.makedirs(ply_dir, exist_ok=True)
    ply_path = os.path.join(ply_dir, "points3D.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_point_clouds)
    pcd.colors = o3d.utility.Vector3dVector(point_rgbs)
    o3d.io.write_point_cloud(ply_path, pcd)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
