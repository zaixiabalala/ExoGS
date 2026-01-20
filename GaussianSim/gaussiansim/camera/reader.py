import json
import numpy as np
from typing import NamedTuple
from ..utils.graphics_utils import focal2fov


class CameraInfo(NamedTuple):
    """
    Camera information container with intrinsic and extrinsic parameters.
    """
    uid: int
    R: np.ndarray  # Rotation matrix (3x3)
    T: np.ndarray  # Translation vector (3,)
    FovY: float    # Field of view in Y direction
    FovX: float    # Field of view in X direction
    width: int     # Image width
    height: int    # Image height


def readCamerasFromTransforms(transformsfile_path, image_width, image_height):
    """
    Read camera information from a transforms JSON file.
    
    Args:
        transformsfile_path: Path to the transforms JSON file
        image_width: Width of the images
        image_height: Height of the images
    
    Returns:
        List of CameraInfo objects
    """
    cam_infos = []
    
    # Load JSON file
    with open(transformsfile_path) as json_file:
        contents = json.load(json_file)
    
    # Process each frame in the transforms file
    for idx, frame in enumerate(contents["frames"]):
        # Extract camera-to-world transformation matrix
        c2w = np.array(frame["transform_matrix"])
        
        # Handle different matrix shapes
        if c2w.ndim == 3 and c2w.shape[0] == 1:
            c2w = c2w[0]
        elif c2w.ndim > 2:
            c2w = c2w.squeeze()
        
        # Convert 3x4 to 4x4 if needed
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        elif c2w.shape != (4, 4):
            raise ValueError(f"Expected transform_matrix to be 3x4 or 4x4, got {c2w.shape}")
        
        # Compute world-to-camera transformation
        w2c = np.linalg.inv(c2w)
        R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]
        
        # Convert focal length to field of view
        FovX, FovY = focal2fov(contents["fx"], image_width), focal2fov(contents["fy"], image_height)
        
        # Create and append camera info
        cam_infos.append(CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            width=image_width,
            height=image_height
        ))
    
    return cam_infos


# Callback registry for different camera load types
caminfoLoadTypeCallbacks = {"Blender": readCamerasFromTransforms}
