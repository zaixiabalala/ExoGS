import os
import shutil
import logging
import numpy as np
from typing import List, Union, Dict, Optional
from pathlib import Path
from poseprocess.utils.pose_utils import read_pose_matrices_from_folder, write_pose_matrices_to_folder, compute_finger_poses
from poseprocess.utils.transform import transform_poses_cam_to_base, align_pose_z_axis_to_reference
from poseprocess.utils.grasp_utils import detect_grasp_frame
from poseprocess.vis.pose_mesh_vis import PoseMeshViewer
logger = logging.getLogger(__name__)
class PoseProcessorBase:
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
        self.pose_init_reference = pose_init_reference
        self.eef_type = eef_type
        self.object_dict = {}
        self.container_dict = {}
        self.tcp_dict = {}
        self.gripper_width_list = []
        self.grasp_frame_idx_list = []
        self.release_frame_idx_list = []
        assert len(object_name_list) == len(object_pose_dir_list), f"Different length: object_name_list={len(object_name_list)}, object_pose_dir_list={len(object_pose_dir_list)}"
        for object_name, pose_dir in zip(object_name_list, object_pose_dir_list):
            self.object_dict[object_name] = {
                "pose_dir": pose_dir,
                "pose_list": read_pose_matrices_from_folder(pose_dir),
                "pose_reference": pose_init_reference,
            }

        # transform poses to base coordinate system
        if self.pose_init_reference == "camera":
            for object_name, object_data in self.object_dict.items():
                object_data["pose_list"] = transform_poses_cam_to_base(object_data["pose_list"])
                object_data["pose_reference"] = "base"
        
        logger.info(f"Initialized PoseProcessorBase with {len(self.object_dict)} objects")

        if tcp_pose_dir:
            tcp_pose_list = read_pose_matrices_from_folder(tcp_pose_dir)
            tcp_offset_list = [pose.copy() for pose in tcp_pose_list]
            for idx, pose in enumerate(tcp_offset_list): 
                offset_matrix = np.eye(4)
                offset_matrix[:3, 3] = tcp_offset
                tcp_offset_list[idx] = pose @ offset_matrix
            self.tcp_dict = {"pose_list": tcp_pose_list, "pose_offset_list": tcp_offset_list, "tcp_offset": tcp_offset}
            logger.info(f"Initialized TCPProcessorBase with {len(tcp_offset_list)} TCPs")

        if gripper_width_list:
            self.gripper_width_list = gripper_width_list
            # grasp_frame_info = detect_grasp_frame(self.gripper_width_list)
            # self.grasp_frame_idx_list = grasp_frame_info['grasp_frame_idx_list']
            # self.release_frame_idx_list = grasp_frame_info['release_frame_idx_list']
            # logger.info(f"Grasp frame index: {self.grasp_frame_idx_list}, Release frame index: {self.release_frame_idx_list}")
            
            # Compute finger poses if tcp_dict is available
            if self.tcp_dict:
                finger_poses = compute_finger_poses(
                    self.tcp_dict["pose_list"], 
                    self.gripper_width_list,
                    eef_type=self.eef_type
                )
                self.tcp_dict["left_finger_pose_list"] = finger_poses["left_finger_pose_list"]
                self.tcp_dict["right_finger_pose_list"] = finger_poses["right_finger_pose_list"]
                logger.info(f"Computed finger poses for {self.eef_type}: {len(finger_poses['left_finger_pose_list'])} frames")


    def check_object_name_list(self, object_name_list: List[str]):
        """
        Check if the object name list is valid
        """
        if not object_name_list:
            logger.error("object_name_list is empty");raise ValueError("object_name_list is empty") from None
        missing_objects = set(object_name_list) - set(self.object_dict.keys())
        if missing_objects:
            logger.error(f"Objects not found in object_dict: {missing_objects}. Available objects: {set(self.object_dict.keys())}");raise ValueError(f"Objects not found in object_dict: {missing_objects}. Available objects: {set(self.object_dict.keys())}") from None
        logger.debug(f"Object name list is valid: {object_name_list}")

    def show_states(self):
        logger.info("============ Showing states of PoseProcessorBase ============")
        for object_name, object_data in self.object_dict.items():
            logger.info(f"Object: {object_name}")
            logger.info(f"Pose reference: {object_data['pose_reference']}")
            logger.info(f"Pose list [0]: \n{object_data['pose_list'][0]}")
        
        if self.tcp_dict:
            logger.info(f"TCP offset: {self.tcp_dict['tcp_offset']}")
            logger.info(f"TCP list [0]: \n{self.tcp_dict['pose_list'][0]}")
            logger.info(f"TCP offset list [0]: \n{self.tcp_dict['pose_offset_list'][0]}")
        
        if self.gripper_width_list:
            logger.info(f"Gripper width list[0]: {self.gripper_width_list[0]}")
    
        if self.container_dict:
            logger.info(f"Container dict: {self.container_dict}")
        logger.info("============ End of Showing states of PoseProcessorBase ============")

    def set_container_bbox(self, 
            container_name: str,
            bbox_size_max: np.ndarray = np.array([0.07, 0.07, 0.0225 + 0.02]),  # x,y,z in meters
            bbox_size_min: np.ndarray = np.array([-0.07, -0.07, -0.0225 - 0.02]),
            reference_up_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),  # Reference upward direction (default: base coordinate system z-axis)
            ):
        """
        Set bounding box for a container object. 
        Automatically aligns the container's z-axis (or closest axis) with the reference upward axis.
        
        Args:
            container_name: Name of the container object
            bbox_size_max: Maximum bounding box size [x, y, z] in meters
            bbox_size_min: Minimum bounding box size [x, y, z] in meters
            reference_up_axis: Reference upward direction (default: [0, 0, 1] for base coordinate system z-axis)
        """
        self.check_object_name_list([container_name])
        container_data = self.object_dict[container_name]
        if container_data["pose_reference"] != "base":
            e = f"Container {container_name} is not in base coordinate system to set bbox";logger.error(e);raise ValueError(e) from None
        if len(container_data["pose_list"]) == 0:
            e = f"Container {container_name} has no poses";logger.error(e);raise ValueError(e) from None
        
        # Align first pose to detect the axis and get rotation matrix
        first_pose_aligned, detected_axis, R_align = align_pose_z_axis_to_reference(
            container_data["pose_list"][0],
            reference_up_axis=reference_up_axis
        )
        logger.debug(f"Aligned container '{container_name}' poses: detected axis '{detected_axis}' aligned with reference upward axis")

        self.container_dict = {}
        self.container_dict[container_name] = {
            "bbox_center": first_pose_aligned[:3, 3],
            "bbox_size_max": bbox_size_max,
            "bbox_size_min": bbox_size_min,
            "pose_reference": "base",
        }
        logger.info(f"Set container bbox for '{container_name}' with detected axis '{detected_axis}'")

    def pose_axis_align(self, 
            object_name_list: List[str], 
            reference_up_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
            ):
        """
        Align the pose of the object to the reference upward axis
        """
        self.check_object_name_list(object_name_list)
        if len(object_name_list) == 0:
            e = "object_name_list is empty";logger.error(e);raise ValueError(e) from None
        if len(reference_up_axis) != 3:
            e = "reference_up_axis must be a 3D vector";logger.error(e);raise ValueError(e) from None
        
        for object_name in object_name_list:
            object_data = self.object_dict[object_name].copy()
            first_pose_aligned, detected_axis, R_align = align_pose_z_axis_to_reference(
                object_data["pose_list"][0],
                reference_up_axis=reference_up_axis
            )
            for idx, pose in enumerate(object_data["pose_list"]):
                aligned_pose = pose.copy()
                aligned_pose[:3, :3] = R_align @ pose[:3, :3]
                object_data["pose_list"][idx] = aligned_pose
            self.object_dict[object_name] = object_data
            logger.info(f"Aligned the pose of {object_name} to the reference upward axis: {detected_axis}")

    def pose_visualize(
        self,
        axes_size: float = 0.1,
        line_width: float = 2.0,
    ):
        """
        Visualize pose sequences with mesh models.
        
        Args:
            pose_dict: Dictionary mapping object/eef names to pose lists (each pose is 4x4 matrix).
                      Keys can be object names from object_dict or "eef"/"tcp" for end-effector.
                      All poses must be in base coordinate system.
            axes_size: Size of coordinate frame axes
            line_width: Width of trajectory lines
        """
        pose_dict = {}
        for object_name, object_data in self.object_dict.items():
            pose_dict[object_name] = object_data["pose_list"].copy()
        if self.tcp_dict:
            # 根据末端执行器类型设置不同的名称
            if self.eef_type == "flexiv":
                hand_name = "xense_hand"
                left_finger_name = "xense_leftfinger"
                right_finger_name = "xense_rightfinger"
            elif self.eef_type == "franka" or self.eef_type == "panda":
                hand_name = "panda_hand"
                left_finger_name = "panda_leftfinger"
                right_finger_name = "panda_rightfinger"
            else:
                logger.warning(f"Unknown eef_type: {self.eef_type}, using flexiv defaults")
                hand_name = "xense_hand"
                left_finger_name = "xense_leftfinger"
                right_finger_name = "xense_rightfinger"
            
            pose_dict[hand_name] = self.tcp_dict["pose_list"].copy()
            if "left_finger_pose_list" in self.tcp_dict:
                pose_dict[left_finger_name] = self.tcp_dict["left_finger_pose_list"].copy()
            if "right_finger_pose_list" in self.tcp_dict:
                pose_dict[right_finger_name] = self.tcp_dict["right_finger_pose_list"].copy()
        
        # Create and run viewer
        viewer = PoseMeshViewer(
            pose_dict=pose_dict,
            axes_size=axes_size,
            line_width=line_width,
        )
        logger.info(f"Starting visualization for: {list(pose_dict.keys())}")
        viewer.run()

    def pose_writeback(self, 
            object_name_list: List[str], 
            pose_output_dir_list: List[str],
            force_overwrite: bool = True,
            file_name_format: str = "{i:016d}.txt",
            ):
        """
        Write back the poses to the file
        """
        self.check_object_name_list(object_name_list)
        if len(object_name_list) == 0:
            e = "object_name_list is empty";logger.error(e);raise ValueError(e) from None
        if len(pose_output_dir_list) == 0:
            e = "pose_output_dir_list is empty";logger.error(e);raise ValueError(e) from None
        if len(object_name_list) != len(pose_output_dir_list):
            e = f"Different length: object_name_list={len(object_name_list)}, pose_output_dir_list={len(pose_output_dir_list)}";logger.error(e);raise ValueError(e) from None

        for object_name, pose_output_dir in zip(object_name_list, pose_output_dir_list):
            if force_overwrite:
                shutil.rmtree(pose_output_dir, ignore_errors=True)
            os.makedirs(pose_output_dir, exist_ok=True)
            object_data = self.object_dict[object_name]["pose_list"].copy()
            write_pose_matrices_to_folder(pose_list=object_data, folder_path=pose_output_dir, file_name_format=file_name_format)
            logger.info(f"Written back the poses of {object_name} to {pose_output_dir}")
