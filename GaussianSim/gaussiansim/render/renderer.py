import math
import torch

from ..gaussiansim.model import GaussianModel

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError as e:
    raise ImportError("Cannot import diff_gaussian_rasterization module, please ensure it is properly installed and in PYTHONPATH.") from e


def render(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier: float = 0.8,
           override_color=None, separate_sh: bool = False):
    """
    Render Gaussian Splatting scene from a camera viewpoint.
    
    Args:
        viewpoint_camera: Camera object with transformation matrices and FOV
        pc: GaussianModel containing 3D Gaussians
        bg_color: Background color tensor
        scaling_modifier: Scale modifier for Gaussian sizes
        override_color: Optional precomputed colors to override spherical harmonics
        separate_sh: Whether to use separate DC and rest spherical harmonics
    
    Returns:
        Dictionary containing rendered image, depth, alpha, and radii
    """
    device = pc.get_xyz.device
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    
    # Calculate field of view tangents
    tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
    tanfovy = math.tan(viewpoint_camera.FovY * 0.5)
    
    # Setup rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color.to(device=pc.get_xyz.device, dtype=torch.float32),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Extract Gaussian parameters
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    # Prepare color features
    cov3D_precomp = None
    shs = None
    colors_precomp = None
    
    if override_color is None:
        if separate_sh:
            dc = pc.get_features_dc
            shs = pc.get_features_rest
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # Perform rasterization
    if separate_sh:
        rendered_image, radii, depth_image, alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )
    else:
        rendered_image, radii, depth_image, alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )
    
    return {
        "render": rendered_image.clamp(0, 1),
        "depth": depth_image,
        "alpha": alpha,
        "radii": radii
    }
