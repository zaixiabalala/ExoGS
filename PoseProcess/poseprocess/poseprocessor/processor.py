import os
import logging
import numpy as np
from typing import List, Optional
from scipy.spatial.transform import Rotation
from poseprocess.poseprocessor.base import PoseProcessorBase
from poseprocess.utils.grasp_utils import detect_grasp_frame, fix_object_pose_to_eef
logger = logging.getLogger(__name__)


class PoseProcessorProcessor(PoseProcessorBase):
    def __init__(self,
        object_name_list: List[str],
        object_pose_dir_list: List[str],
        pose_file_type: str = "txt",
        pose_init_reference: str = "camera",
        tcp_pose_dir: Optional[str] = None,
        tcp_offset = np.array([0.0, 0.0, 0.0]),
        gripper_width_list: Optional[List[float]] = None,
        eef_type: str = "flexiv",
    ):
        super().__init__(
            object_name_list=object_name_list,
            object_pose_dir_list=object_pose_dir_list,
            pose_file_type=pose_file_type,
            pose_init_reference=pose_init_reference,
            tcp_pose_dir=tcp_pose_dir,
            tcp_offset=tcp_offset,
            gripper_width_list=gripper_width_list,
            eef_type=eef_type,
        )

    def pose_overwrite(self, object_name_list: List[str], overwrite_matrix: Optional[np.ndarray] = None):
        """
        Overwrite the poses of the objects with the first pose in the pose list
        """
        self.check_object_name_list(object_name_list)
        for object_name in object_name_list:
            object_data = self.object_dict[object_name]
            if len(object_data["pose_list"]) == 0:
                e = f"Object {object_name} has empty pose list, skipping overwrite";logger.error(e);raise ValueError(e) from None

            if overwrite_matrix is not None:
                first_pose = overwrite_matrix.copy()
            else:
                first_pose = object_data["pose_list"][0].copy()
            object_data["pose_list"] = [first_pose.copy() for _ in range(len(object_data["pose_list"]))]
            self.object_dict[object_name] = object_data
            logger.info(f"Overwrote the poses of {object_name}")

    def pose_offset(self, object_name_list: List[str], offset_vector: Optional[np.ndarray] = np.array([0.0, 0.0, 0.0])):
        """
        Offset the poses of the objects in the pose list
        """
        self.check_object_name_list(object_name_list)
        if offset_vector is not None and (not isinstance(offset_vector, np.ndarray) or offset_vector.shape != (3,)):
            logger.error(f"Offset vector is not a 3D vector: {offset_vector.shape}");raise ValueError(f"Offset vector is not a 3D vector: {offset_vector.shape}") from None

        for object_name in object_name_list:
            object_data = self.object_dict[object_name]
            # Offset must be in base coordinate system
            if object_data["pose_reference"] != "base":
                logger.error(f"Object {object_name} is not in base coordinate system to offset");raise ValueError(f"Object {object_name} is not in base coordinate system to offset") from None
            # Apply offset only to the translation part (pose[:3, 3]) of each 4x4 pose matrix
            for pose in object_data["pose_list"]:
                pose[:3, 3] += offset_vector
            logger.info(f"Offset the poses of {object_name} by {offset_vector}")

    def pose_freeze(self, 
            object_name_list: List[str], 
            ):
        """
        Freeze the poses of the objects in the pose list
        """
        self.check_object_name_list(object_name_list)
        if not self.container_dict:
            e = "Container dict is not set";logger.error(e);raise ValueError(e) from None

        container_data = list(self.container_dict.values())[0].copy()
        bbox_center = container_data["bbox_center"]
        bbox_min = bbox_center + container_data["bbox_size_min"]
        bbox_max = bbox_center + container_data["bbox_size_max"]

        for object_name in object_name_list:
            object_data = self.object_dict[object_name].copy()
            if not object_data["pose_reference"] == "base":
                e = f"Object {object_name} is not in base coordinate system to freeze";logger.error(e);raise ValueError(f"Object {object_name} is not in base coordinate system to freeze") from None

            first_entry_idx = None
            for idx, pose in enumerate(object_data["pose_list"]):                
                in_bbox = np.all(pose[:3, 3] >= bbox_min) and np.all(pose[:3, 3] <= bbox_max)
                if in_bbox and first_entry_idx is None:
                    first_entry_idx = idx
                    break
            
            # add first entry index to object_dict
            if first_entry_idx is not None:
                self.object_dict[object_name]["freeze_index"] = first_entry_idx
                logger.info(f"Object {object_name} first entered bbox at pose index {first_entry_idx}")
            else:
                self.object_dict[object_name]["freeze_index"] = None
                logger.warning(f"Object {object_name} never entered bbox")
            
            # freeze the poses in the bbox
            if first_entry_idx is not None:
                freeze_pose = object_data["pose_list"][first_entry_idx].copy()
                for i in range(first_entry_idx, len(object_data["pose_list"])):
                    object_data["pose_list"][i] = freeze_pose.copy()
                self.object_dict[object_name] = object_data
                logger.debug(f"Froze poses of {object_name} from index {first_entry_idx} to {len(object_data['pose_list'])-1}")
            
    def pose_fix(self, 
            object_name_list: List[str],
            grasp_delay: int=0,
            release_delay: int=0):
        """
        Fix the poses of the objects to the tcp poses
        """
        self.check_object_name_list(object_name_list)
        if not self.tcp_dict:
            e = "TCP dict is not set";logger.error(e);raise ValueError(e) from None
        if not self.gripper_width_list:
            e = "Gripper width list is not set";logger.error(e);raise ValueError(e) from None
        if len(self.grasp_frame_idx_list) != len(self.release_frame_idx_list):
            e = "Grasp frame index list and release frame index list have different lengths";logger.error(e);raise ValueError(e) from None
        if len(object_name_list) != len(self.grasp_frame_idx_list):
            e = f"Object name list and grasp frame index list have different lengths, object_name_list={object_name_list}, grasp_frame_idx_list={self.grasp_frame_idx_list}";logger.error(e);raise ValueError(e) from None
        
        
        for object_name in object_name_list:
            min_distance = float('inf')
            grasp_release_frame_list_idx = -1

            # find the grasp and release frame with the smallest distance
            for idx, grasp_frame_idx in enumerate(self.grasp_frame_idx_list):
                distance = np.linalg.norm(self.object_dict[object_name]["pose_list"][grasp_frame_idx][:3, 3] - self.tcp_dict["pose_offset_list"][grasp_frame_idx][:3, 3])
                if distance < min_distance:
                    min_distance = distance
                    grasp_release_frame_list_idx = idx
            logger.info(f"Object {object_name} with the smallest distance: {min_distance} with the {grasp_release_frame_list_idx}.th grasp")

            # fix the pose of the object to the tcp offset pose
            grasp_idx = self.grasp_frame_idx_list[grasp_release_frame_list_idx]
            release_idx = self.release_frame_idx_list[grasp_release_frame_list_idx]
            self.object_dict[object_name]["pose_list"] = fix_object_pose_to_eef(
                self.object_dict[object_name]["pose_list"],
                self.tcp_dict["pose_offset_list"],
                reference_frame_idx=grasp_idx,
                fix_start_frame_idx=grasp_idx + grasp_delay,
                fix_end_frame_idx=release_idx + release_delay,
            )
            logger.info(f"Fixed the pose of {object_name} to the tcp offset pose")
    
    def pose_rotate(self, object_name_list: List[str], axis: str = "z", angle_deg: float = 0.0):
        """
        Rotate all poses in the sequence around the object's local axis.
        """        
        self.check_object_name_list(object_name_list)
        if axis not in ["x", "y", "z"]:
            e = f"Invalid axis '{axis}', must be one of 'x', 'y', 'z'";logger.error(e);raise ValueError(e) from None
        
        angle_rad = np.deg2rad(angle_deg)
        if axis == "x": euler = [angle_rad, 0, 0]
        elif axis == "y": euler = [0, angle_rad, 0]
        else: euler = [0, 0, angle_rad]
        R_local = Rotation.from_euler('xyz', euler).as_matrix()
        
        for object_name in object_name_list:
            object_data = self.object_dict[object_name].copy()
            if object_data["pose_reference"] != "base":
                e = f"Object {object_name} is not in base coordinate system to rotate";logger.error(e);raise ValueError(e) from None
            
            for pose in object_data["pose_list"]:
                pose[:3, :3] = pose[:3, :3] @ R_local # new_R = R_original @ R_local
            logger.info(f"Rotated poses of {object_name} by {angle_deg} degrees around local {axis}-axis")

            self.object_dict[object_name] = object_data

    def pose_drop(self, object_name: str, final_z: float, drop_duration: int = 10):
        """
        Generate the drop trajectory of the object from the release frame
        
        Args:
            object_name: object name
            final_z: the final z value of the object
            drop_duration: the duration of the drop, default is 10 frames
        """
        self.check_object_name_list([object_name])
        if not self.tcp_dict:
            e = "TCP dict is not set";logger.error(e);raise ValueError(e) from None
        if not self.gripper_width_list:
            e = "Gripper width list is not set";logger.error(e);raise ValueError(e) from None
        if len(self.grasp_frame_idx_list) != len(self.release_frame_idx_list):
            e = "Grasp frame index list and release frame index list have different lengths";logger.error(e);raise ValueError(e) from None
        
        object_data = self.object_dict[object_name]
        if object_data["pose_reference"] != "base":
            e = f"Object {object_name} is not in base coordinate system to drop";logger.error(e);raise ValueError(e) from None
        
        # Find the release frame of the object
        release_frame_idx = self.release_frame_idx_list[0]
        
        # Check if release frame is valid
        if release_frame_idx >= len(object_data["pose_list"]):
            e = f"Release frame index {release_frame_idx} exceeds total sequence length {len(object_data['pose_list'])}";logger.error(e);raise ValueError(e) from None
        
        # Get the reference pose from 15 frames before the release frame
        reference_frame_idx = max(0, release_frame_idx - 30)
        reference_pose = object_data["pose_list"][reference_frame_idx].copy()
        reference_z = reference_pose[2, 3]
        
        # Calculate the actual drop end frame, avoiding exceeding total sequence length
        total_frames = len(object_data["pose_list"])
        final_down_frame_idx = min(total_frames - 1, release_frame_idx + drop_duration)
        
        # Calculate actual drop duration
        actual_drop_duration = final_down_frame_idx - release_frame_idx
        if actual_drop_duration <= 0:
            e = f"Cannot drop: release frame {release_frame_idx} is at or beyond the end of sequence (total: {total_frames})";logger.error(e);raise ValueError(e) from None
        
        # Generate drop trajectory points (using quadratic function for accelerated drop)
        z_diff = final_z - reference_z
        for i in range(release_frame_idx, final_down_frame_idx + 1):
            # Calculate normalized time parameter [0, 1]
            t = (i - release_frame_idx) / actual_drop_duration if actual_drop_duration > 0 else 0
            
            # Use quadratic function to simulate accelerated drop (t^2 makes it slow at start, fast at end)
            z_current = reference_z + z_diff * (t ** 2)
            
            # Copy the reference pose and update only the z value
            pose_i = reference_pose.copy()
            pose_i[2, 3] = z_current
            object_data["pose_list"][i] = pose_i
        
        self.object_dict[object_name] = object_data
        logger.info(f"Generated drop trajectory for {object_name} from frame {release_frame_idx} to {final_down_frame_idx}, "
                f"z: {reference_z:.4f} -> {final_z:.4f} over {actual_drop_duration + 1} frames")