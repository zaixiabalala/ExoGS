import json
import os
import numpy as np
from typing import List

from ..utils.camera_utils import generate_spherical_camera_array, visualize_camera_positions
from .constants import FX, FY, CX, CY


class CamerasJSONGenerator():
    """
    Generator for creating camera JSON files with camera parameters and poses.
    """
    
    def __init__(self, json_output_path, fx=FX, fy=FY, cx=CX, cy=CY):
        """
        Initialize the camera JSON generator.
        
        Args:
            json_output_path: Path to output JSON file
            fx: Focal length in x direction (default from constants)
            fy: Focal length in y direction (default from constants)
            cx: Principal point x coordinate (default from constants)
            cy: Principal point y coordinate (default from constants)
        """
        assert json_output_path is not None, "[Error] json_output_path cannot be empty"
        
        self.json_output_path = json_output_path
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Initialize transforms dictionary with camera intrinsics
        self.transforms = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "frames": []
        }

    def add_camera(self, camera_extrinsic_matrix: np.ndarray):
        """
        Add a single camera pose to the transforms.
        
        Args:
            camera_extrinsic_matrix: 4x4 camera extrinsic transformation matrix
        """
        assert camera_extrinsic_matrix.shape == (4, 4), "[Error] camera_extrinsic_matrix must be 4x4 matrix"
        
        self.transforms["frames"].append({
            "transform_matrix": camera_extrinsic_matrix.tolist()
        })

    def save_json(self):
        """
        Save the camera transforms to a JSON file.
        """
        assert self.json_output_path.endswith(".json"), "[Error] json_output_path must be JSON file"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.json_output_path):
            os.makedirs(os.path.dirname(self.json_output_path), exist_ok=True)
        
        # Write JSON file with indentation
        with open(self.json_output_path, 'w') as f:
            json.dump(self.transforms, f, indent=4)
        
        print(f"[Info] JSON file {self.json_output_path.split('/')[-1]} saved to: {self.json_output_path}")
        print(f"[Info] Total cameras: {len(self.transforms['frames'])}")

    def add_spherical_cameras(
        self,
        camera_extrinsic_matrix: np.ndarray,
        phi_values: List[float],
        camera_counts: List[int]
    ):
        """
        Add multiple cameras arranged in a spherical pattern.
        
        Args:
            camera_extrinsic_matrix: Base 4x4 camera extrinsic transformation matrix
            phi_values: List of phi (elevation) angles for camera positions
            camera_counts: List of camera counts for each phi value
        """
        assert camera_extrinsic_matrix.shape == (4, 4), "[Error] camera_extrinsic_matrix must be 4x4 matrix"
        assert len(phi_values) == len(camera_counts), "[Error] phi_values and camera_counts must have same length"
        
        # Generate and add spherical camera array
        for camera_pose in generate_spherical_camera_array(camera_extrinsic_matrix, phi_values, camera_counts):
            self.add_camera(camera_pose)

    def visualize_cameras(self):
        """
        Visualize all camera positions in 3D space.
        """
        print(f"[Info] Starting visualization")
        
        # Extract camera poses from transforms
        camera_poses = [np.array(frame["transform_matrix"]) for frame in self.transforms["frames"]]
        
        visualize_camera_positions(camera_poses)
        print(f"[Info] Visualization complete")
