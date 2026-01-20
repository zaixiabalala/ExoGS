import torch
import numpy as np
import open3d as o3d
from plyfile import PlyData

from .sh_utils import SH2RGB


def detect_sh_degree_from_ply(ply_path: str) -> int:
    """
    Detect spherical harmonics degree from PLY file.
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        sh_degree: Detected spherical harmonics degree (0, 1, 2, 3, ...)
    """
    plydata = PlyData.read(ply_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    num_f_rest = len(extra_f_names)
    
    # Infer sh_degree from f_rest count
    # num_f_rest = 3 * (sh_degree + 1)^2 - 3
    # Therefore: (sh_degree + 1)^2 = (num_f_rest + 3) / 3
    if num_f_rest == 0:
        return 0
    
    sh_degree_plus_one_squared = (num_f_rest + 3) / 3
    sh_degree_plus_one = int(np.sqrt(sh_degree_plus_one_squared))
    sh_degree = sh_degree_plus_one - 1
    
    # Verify
    expected_num = 3 * (sh_degree + 1) ** 2 - 3
    if expected_num != num_f_rest:
        raise ValueError(f"Cannot detect sh_degree from PLY file: f_rest count={num_f_rest}, cannot match valid sh_degree")
    
    return sh_degree


def extract_and_save_point_cloud(gaussians_model, output_path, target_points=10000):
    """
    Extract point cloud from Gaussian model and downsample to target number of points.
    
    Args:
        gaussians_model: GaussianModel instance
        output_path: Output point cloud file path
        target_points: Target number of points
    """
    # Extract 3D coordinates
    points = gaussians_model._xyz.detach().cpu().numpy()
    
    # Extract colors (from spherical harmonics DC component)
    features_dc = gaussians_model._features_dc.detach().cpu().numpy()
    colors = SH2RGB(torch.from_numpy(features_dc.squeeze())).numpy()
    # Ensure color values in [0, 1] range
    colors = np.clip(colors, 0.0, 1.0)
    
    # Extract opacity
    opacities = gaussians_model._opacity.detach().cpu().numpy()
    
    # Filter points with low opacity
    valid_mask = opacities.squeeze() > 0.1
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Downsample to target number of points
    if len(points) > target_points:
        pcd = pcd.random_down_sample(target_points / len(points))
    
    # Save point cloud
    o3d.io.write_point_cloud(output_path, pcd)
