import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import logging
import numpy as np
from tap import Tap
from pathlib import Path
from typing import List, Optional
from poseprocess.utils.pose_utils import read_pose_matrices_from_folder, write_pose_matrices_to_folder
from poseprocess.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def get_cam_to_base_matrix(cam_suffix: str):
    """
    Get CAM_TO_BASE_MATRIX from camera_constants module based on cam folder suffix.
    
    Args:
        cam_suffix: The suffix of cam folder (e.g., "319522062799" from "cam_319522062799")
    
    Returns:
        4x4 transformation matrix from camera to base coordinate system
    """
    try:
        from assets import camera_constants
        matrix_name = f"CAM_TO_BASE_MATRIX_{cam_suffix}"
        if hasattr(camera_constants, matrix_name):
            matrix = getattr(camera_constants, matrix_name)
            logger.debug(f"Found camera matrix {matrix_name} for cam suffix {cam_suffix}")
            return matrix
        else:
            e = f"Camera matrix {matrix_name} not found in camera_constants for cam suffix {cam_suffix}";logger.error(e);raise ValueError(e) from None
    except Exception as e:
        e = f"Failed to get camera matrix for cam suffix {cam_suffix}: {e}";logger.error(e);raise ValueError(e) from None


def transform_poses_cam_to_base(pose_list: List[np.ndarray], cam_to_base_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Transform poses from camera coordinate system to base coordinate system.
    
    Args:
        pose_list: List of 4x4 pose matrices in camera coordinate system
        cam_to_base_matrix: 4x4 transformation matrix from camera to base
    
    Returns:
        List of 4x4 pose matrices in base coordinate system
    """
    if len(pose_list) == 0:
        e = "pose list is empty";logger.error(e);raise ValueError(e) from None
    
    return [cam_to_base_matrix @ pose for pose in pose_list]


def average_poses(pose_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Average multiple pose lists frame by frame.
    For each frame index, average the translation (xyz position) and use the rotation from the first camera.
    
    Args:
        pose_lists: List of pose lists from different cameras
    
    Returns:
        List of averaged 4x4 pose matrices
    """
    if len(pose_lists) == 0:
        e = "pose_lists is empty";logger.error(e);raise ValueError(e) from None
    
    # Find the minimum length to ensure all lists have the same number of frames
    min_length = min(len(pose_list) for pose_list in pose_lists)
    if min_length == 0:
        e = "One or more pose lists are empty";logger.error(e);raise ValueError(e) from None
    
    averaged_poses = []
    for frame_idx in range(min_length):
        # Collect poses from all cameras for this frame
        frame_poses = [pose_list[frame_idx] for pose_list in pose_lists]
        
        # Average translation (position)
        translations = np.array([pose[:3, 3] for pose in frame_poses])
        avg_translation = np.mean(translations, axis=0)
        
        # Use rotation matrix from the first camera
        first_rotation = frame_poses[0][:3, :3]
        
        # Construct averaged pose matrix
        avg_pose = np.eye(4, dtype=np.float64)
        avg_pose[:3, :3] = first_rotation
        avg_pose[:3, 3] = avg_translation
        
        averaged_poses.append(avg_pose)
    
    logger.debug(f"Averaged {len(pose_lists)} pose lists with {min_length} frames")
    return averaged_poses


def process_record_folder(record_path: Path, base_dir: Path, force_overwrite: bool = False):
    """
    Process a single record folder.
    
    Args:
        record_path: Path to the record folder
        base_dir: Base directory containing all record folders
        force_overwrite: If True, remove existing poses folder before processing
    """
    logger.info(f"Processing record folder: {record_path.name}")
    
    # Remove existing poses folder if force_overwrite is enabled
    poses_dir = record_path / "poses"
    if force_overwrite and poses_dir.exists() and poses_dir.is_dir():
        logger.info(f"Removing existing poses folder: {poses_dir}")
        try:
            shutil.rmtree(poses_dir)
            logger.info(f"Successfully removed poses folder: {poses_dir}")
        except Exception as e:
            logger.error(f"Failed to remove poses folder {poses_dir}: {e}")
            raise
    
    # Find all cam_* folders in the record directory
    cam_folders = sorted([d for d in record_path.iterdir() if d.is_dir() and d.name.startswith("cam_")])
    if len(cam_folders) == 0:
        logger.warning(f"No cam_* folders found in {record_path}")
        return
    
    logger.info(f"Found {len(cam_folders)} camera folders: {[f.name for f in cam_folders]}")
    
    # Extract cam suffixes and get corresponding transformation matrices
    cam_matrices = {}
    for cam_folder in cam_folders:
        cam_suffix = cam_folder.name.replace("cam_", "")
        try:
            cam_matrices[cam_folder.name] = get_cam_to_base_matrix(cam_suffix)
        except ValueError as e:
            logger.warning(f"Skipping {cam_folder.name}: {e}")
            continue
    
    if len(cam_matrices) == 0:
        logger.warning(f"No valid camera matrices found for record {record_path.name}")
        return
    
    # Find organized folders in each cam folder
    organized_folders = {}
    for cam_folder in cam_folders:
        if cam_folder.name not in cam_matrices:
            continue
        organized_path = cam_folder / "organized"
        if organized_path.exists() and organized_path.is_dir():
            organized_folders[cam_folder.name] = organized_path
        else:
            logger.warning(f"Organized folder not found in {cam_folder}")
    
    if len(organized_folders) == 0:
        logger.warning(f"No organized folders found in record {record_path.name}")
        return
    
    # Find all tracking_* folders (objects) in the first organized folder
    first_organized = list(organized_folders.values())[0]
    tracking_folders = sorted([d for d in first_organized.iterdir() 
                               if d.is_dir() and d.name.startswith("tracking_")])
    
    if len(tracking_folders) == 0:
        logger.warning(f"No tracking_* folders found in {first_organized}")
        return
    
    logger.info(f"Found {len(tracking_folders)} object folders: {[f.name for f in tracking_folders]}")
    
    # Process each object (tracking folder)
    for tracking_folder in tracking_folders:
        object_name = tracking_folder.name  # e.g., "tracking_1"
        logger.info(f"Processing object: {object_name}")
        
        # Collect pose lists from all cameras for this object
        object_pose_lists_in_base = {}
        
        for cam_name, organized_path in organized_folders.items():
            ob_in_cam_path = organized_path / object_name / "ob_in_cam"
            
            if not ob_in_cam_path.exists() or not ob_in_cam_path.is_dir():
                logger.warning(f"ob_in_cam folder not found: {ob_in_cam_path}")
                continue
            
            try:
                # Read poses in camera coordinate system
                pose_list_cam = read_pose_matrices_from_folder(str(ob_in_cam_path))
                logger.debug(f"Read {len(pose_list_cam)} poses from {cam_name}/{object_name}")
                
                # Transform to base coordinate system
                cam_to_base = cam_matrices[cam_name]
                pose_list_base = transform_poses_cam_to_base(pose_list_cam, cam_to_base)
                object_pose_lists_in_base[cam_name] = pose_list_base
                
            except Exception as e:
                logger.error(f"Failed to process {cam_name}/{object_name}: {e}")
                continue
        
        if len(object_pose_lists_in_base) == 0:
            logger.warning(f"No valid pose data found for object {object_name}")
            continue
        
        # Average poses from all cameras
        try:
            pose_lists = list(object_pose_lists_in_base.values())
            averaged_poses = average_poses(pose_lists)
            logger.info(f"Averaged poses from {len(pose_lists)} cameras for {object_name}, got {len(averaged_poses)} frames")
        except Exception as e:
            logger.error(f"Failed to average poses for {object_name}: {e}")
            continue
        
        # Determine object index from tracking folder name (e.g., tracking_1 -> 1)
        try:
            object_idx = int(object_name.replace("tracking_", ""))
        except ValueError:
            logger.warning(f"Could not extract object index from {object_name}, using default")
            object_idx = 1
        
        # Create output directory: record/poses/object_x_org
        output_dir = record_path / "poses" / f"object_{object_idx}_org"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write averaged poses to output directory
        try:
            write_pose_matrices_to_folder(
                pose_list=averaged_poses,
                folder_path=str(output_dir),
                file_name_format="{i:016d}.txt"
            )
            logger.info(f"Saved {len(averaged_poses)} averaged poses to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to write poses for {object_name} to {output_dir}: {e}")


def main(args):
    """
    Main function to process all record folders.
    
    Args:
        args: ArgumentParser instance with configuration
    """
    base_path = Path(args.base_dir)
    if not base_path.exists() or not base_path.is_dir():
        e = f"Base directory does not exist or is not a directory: {args.base_dir}";logger.error(e);raise ValueError(e) from None
    
    # Find all record_* folders
    record_folders = sorted([d for d in base_path.iterdir() 
                            if d.is_dir() and d.name.startswith("record_")])
    
    if len(record_folders) == 0:
        logger.warning(f"No record_* folders found in {args.base_dir}")
        return
    
    logger.info(f"Found {len(record_folders)} record folders to process")
    
    # Process each record folder
    for record_folder in record_folders:
        try:
            process_record_folder(record_folder, base_path, force_overwrite=args.force_overwrite)
        except Exception as e:
            logger.error(f"Failed to process record folder {record_folder.name}: {e}")
            continue
    
    logger.info("Finished processing all record folders")


class ArgumentParser(Tap):
    base_dir: str = "/home/ubuntu/data/Origin_Data/records_unscrew_cap_1221"  # Base directory containing all record_* folders
    log_level: str = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR
    log_file: Optional[str] = None  # Log file path (optional)
    force_overwrite: bool = True  # If True, remove existing poses folder before processing


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    
    # Setup logging
    setup_logging(
        log_level=getattr(logging, args.log_level),
        log_file=args.log_file
    )
    
    # Run main function
    main(args)

