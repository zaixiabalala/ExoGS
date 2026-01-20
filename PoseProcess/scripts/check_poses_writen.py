import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from tap import Tap
from typing import List
from pathlib import Path
from poseprocess.utils.logging_config import setup_logging


def check_poses_folders(records_dir: str, folder_names: List[str]):
    """
    check if the poses folders exist in each record and the txt file count is correct.
    
    Args:
        records_dir: the base directory containing all record folders
        folder_names: the list of folder names to check (e.g. ["object_1", "object_2"])
    
    Raises:
        ValueError: if the folder does not exist, the txt file count is 0 or the txt file count is not consistent between folders
    """
    logger = logging.getLogger(__name__)
    
    records_path = Path(records_dir)
    if not records_path.exists() or not records_path.is_dir():
        e = f"Records directory does not exist or is not a directory: {records_dir}"
        logger.error(e)
        raise ValueError(e)
    
    record_folders = sorted([d for d in records_path.iterdir() 
                            if d.is_dir() and d.name.startswith("record_")])
    
    if len(record_folders) == 0:
        logger.warning(f"No record_* folders found in {records_dir}")
        return
    
    logger.info(f"Found {len(record_folders)} record folders")
    
    for record_folder in record_folders:
        logger.info(f"Checking record folder: {record_folder.name}")
        
        poses_dir = record_folder / "poses"
        if not poses_dir.exists() or not poses_dir.is_dir():
            e = f"poses folder does not exist: {poses_dir}"
            logger.error(e)
            raise ValueError(e)
        
        folder_txt_counts = {}
        missing_folders = []
        
        for folder_name in folder_names:
            folder_path = poses_dir / folder_name
            
            if not folder_path.exists() or not folder_path.is_dir():
                missing_folders.append(folder_name)
                continue
            
            txt_files = list(folder_path.glob("*.txt"))
            txt_count = len(txt_files)
            folder_txt_counts[folder_name] = txt_count
            
            logger.info(f"  {folder_name}: {txt_count} txt files")
        
        if missing_folders:
            e = f"The following folders are missing in {record_folder.name}/poses: {missing_folders}"
            logger.error(e)
            raise ValueError(e)
        
        if len(folder_txt_counts) == 0:
            e = f"No specified folders found in {record_folder.name}/poses"
            logger.error(e)
            raise ValueError(e)
        
        zero_count_folders = [name for name, count in folder_txt_counts.items() if count == 0]
        if zero_count_folders:
            e = f"The following folders have 0 txt files in {record_folder.name}/poses: {zero_count_folders}"
            logger.error(e)
            raise ValueError(e)
        
        txt_counts = list(folder_txt_counts.values())
        if len(set(txt_counts)) > 1:
            e = f"The txt file counts are not consistent between folders in {record_folder.name}/poses: {folder_txt_counts}"
            logger.error(e)
            raise ValueError(e)
        
        logger.info(f"  {record_folder.name} check passed: all folders have {txt_counts[0]} txt files")
    
    logger.info("All record folders checked")


def main(args):
    """
    main function
    
    Args:
        args: ArgumentParser instance, containing configuration parameters
    """
    setup_logging(
        log_level=logging.INFO,
        log_file=args.log_file
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting to check poses folders")
    
    try:
        check_poses_folders(
            records_dir=args.records_dir,
            folder_names=args.folder_names
        )
        logger.info("Check completed, all record folders passed")
    except ValueError as e:
        logger.error(f"Check failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error occurred during check: {e}")
        raise


class ArgumentParser(Tap):
    records_dir: str = "/media/ubuntu/B0A8C06FA8C0361E/Data/Origin_Data/records_dumbbell_1017"
    folder_names: List[str] = ["object_1", "object_2"]
    log_file: str = "logs/check_poses.log"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)

