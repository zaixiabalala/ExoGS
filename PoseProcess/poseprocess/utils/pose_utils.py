import os
import logging
import numpy as np
import shutil
from typing import List
from assets.camera_constants import CAM_TO_BASE_MATRIX_319522062799, CAM_TO_BASE_MATRIX_327322062498
from poseprocess.utils.transform import xyzquat_to_matrix

# 创建 logger，使用模块名作为 logger 名称
logger = logging.getLogger(__name__)

def read_pose_matrices_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    read all 4x4 pose matrices from a folder(txt or npy files), 
    make sure the pose matrices are 4x4 numpy arrays and more than 0
    
    Args:
        folder_path: the path of the folder containing the 4x4 pose matrices
    
    Returns:
        a list of 4x4 numpy arrays
    """
    if not os.path.exists(folder_path):
        logger.error(f"Folder not exists: {folder_path}");raise FileNotFoundError(f"Folder not exists: {folder_path}") from None
    if not os.path.isdir(folder_path):
        logger.error(f"Path is not a folder: {folder_path}");raise ValueError(f"Path is not a folder: {folder_path}") from None

    pose_matrices: List[np.ndarray] = []
    
    # get all txt files and sort by file name
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    if not txt_files and not npy_files:
        logger.error(f"No txt or npy files found in folder: {folder_path}");raise ValueError(f"No txt or npy files found in folder: {folder_path}") from None
    
    files_list = []
    file_type = None
    if len(txt_files) > 0 and len(npy_files) == 0: file_type = "txt"; files_list = txt_files
    elif len(txt_files) == 0 and len(npy_files) > 0: file_type = "npy"; files_list = npy_files
    elif len(txt_files) > 0 and len(npy_files) > 0: e = "Both txt and npy files found in folder: {folder_path}";logger.error(e);raise ValueError(e) from None
    elif len(txt_files) == 0 and len(npy_files) == 0: e = "No txt or npy files found in folder: {folder_path}";logger.error(e);raise ValueError(e) from None

    logger.debug(f"Found {len(files_list)} {file_type} files, starting to read...")
    
    # read each txt file
    for file in files_list:
        file_path = os.path.join(folder_path, file)
        try:
            # read 4x4 matrix
            if file_type == "txt": matrix = np.loadtxt(file_path)
            elif file_type == "npy": matrix = np.load(file_path)
            else: e = f"Unknown file type: {file_type}";logger.error(e);raise ValueError(e) from None
            if matrix.shape != (4, 4):
                if matrix.shape == (8,):
                    matrix = xyzquat_to_matrix(matrix[:7])
                else:
                    e = f"{file} is not a 4x4 matrix, shape is {matrix.shape}";logger.error(e);raise ValueError(e) from None
            pose_matrices.append(matrix)
        except Exception as e:
            e = f"Failed to read file {file}: {e}";logger.error(e);raise ValueError(e) from None
    
    logger.debug(f"Successfully read {len(pose_matrices)} pose matrices")
    
    return pose_matrices


def write_pose_matrices_to_folder(pose_list: List[np.ndarray], 
                                  folder_path: str,
                                  file_name_format: str = "{i:016d}.txt"):
    """
    Write the pose matrices to the folder

    Args:
        pose_list: the list of pose matrices
        folder_path: the path of the folder to write the pose matrices
        force_overwrite: whether to force overwrite the existing files
        file_name_format: the format of the file name
    """
    if len(pose_list) == 0:
        e = "pose_list is empty";logger.error(e);raise ValueError(e) from None
    if not os.path.exists(folder_path):
        e = f"Folder not exists: {folder_path}";logger.error(e);raise FileNotFoundError(e) from None
    if not os.path.isdir(folder_path):
        e = f"Path is not a folder: {folder_path}";logger.error(e);raise ValueError(e) from None

    logger.debug(f"Writing {len(pose_list)} pose matrices to the folder: {folder_path}")
    
    saved_count = 0
    for i, pose in enumerate(pose_list):
        file_name = file_name_format.format(i=i)
        file_path = os.path.join(folder_path, file_name)
        try:
            np.savetxt(file_path, pose)
            saved_count += 1
        except Exception as e:
            e = f"Failed to write file {file_path}: {e}";logger.error(e);raise ValueError(e) from None
    
    logger.debug(f"Successfully written {saved_count} pose matrices")


def compute_finger_poses(
    tcp_pose_list: List[np.ndarray],
    gripper_width_list: List[float],
    eef_type: str = "flexiv",
) -> dict:
    """
    Compute left and right finger poses from tcp poses and gripper widths.
    
    Args:
        tcp_pose_list: TCP pose list (List of 4x4 matrices)
        gripper_width_list: Gripper width list (total width in meters)
        eef_type: End-effector type, either "flexiv" or "franka"/"panda" (default: "flexiv")
    
    Returns:
        dict with 'left_finger_pose_list' and 'right_finger_pose_list'
    """
    if len(tcp_pose_list) != len(gripper_width_list):
        e = f"Length mismatch: tcp_pose_list={len(tcp_pose_list)}, gripper_width_list={len(gripper_width_list)}"
        logger.error(e);raise ValueError(e) from None
    
    left_finger_pose_list = []
    right_finger_pose_list = []
    
    for tcp_pose, width in zip(tcp_pose_list, gripper_width_list):
        half_width = width / 2.0
        
        if eef_type == "flexiv":
            # Flexiv/Xense: Finger origins coincide with hand origin, offset in local x-axis
            # Left finger: offset in +x direction
            left_pose = tcp_pose.copy()
            left_pose[:3, 3] -= tcp_pose[:3, 0] * half_width
            left_finger_pose_list.append(left_pose)
            
            # Right finger: offset in -x direction
            right_pose = tcp_pose.copy()
            right_pose[:3, 3] += tcp_pose[:3, 0] * half_width
            right_finger_pose_list.append(right_pose)
            
        elif eef_type == "franka" or eef_type == "panda":
            
            left_pose = tcp_pose.copy()
            left_pose[:3, 3] += tcp_pose[:3, 1] * half_width
            left_pose[:3, 3] += tcp_pose[:3, 2] * 0.0584
            left_finger_pose_list.append(left_pose)
            
            right_pose = tcp_pose.copy()
            right_pose[:3, 3] -= tcp_pose[:3, 1] * half_width
            right_pose[:3, 3] += tcp_pose[:3, 2] * 0.0584
            right_finger_pose_list.append(right_pose)
            
            logger.debug(f"Using flexiv-style finger pose calculation for {eef_type} (TODO: update with actual panda model)")
        else:
            logger.warning(f"Unknown eef_type: {eef_type}, using flexiv defaults")
            # Fallback to flexiv calculation
            left_pose = tcp_pose.copy()
            left_pose[:3, 3] -= tcp_pose[:3, 0] * half_width
            left_finger_pose_list.append(left_pose)
            
            right_pose = tcp_pose.copy()
            right_pose[:3, 3] += tcp_pose[:3, 0] * half_width
            right_finger_pose_list.append(right_pose)
    
    logger.debug(f"Computed {len(left_finger_pose_list)} finger poses for {eef_type}")
    return {
        'left_finger_pose_list': left_finger_pose_list,
        'right_finger_pose_list': right_finger_pose_list,
    }

