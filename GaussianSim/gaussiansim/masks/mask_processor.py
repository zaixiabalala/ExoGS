"""
Mask processor for batch cleaning masks in Gaussian Splatting records.
Provides utilities for cleaning mask folders across multiple records and cameras.
"""

import os
import cv2
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import glob
from .mask_cleaner import MaskCleaner


class MaskProcessor:
    """
    Mask processor for batch processing masks in Gaussian Splatting records.
    Handles cleaning operations across multiple records, cameras, and mask folders.
    """
    
    def __init__(self, GS_dir: str, records_name: str):
        """
        Initialize mask processor.
        
        Args:
            GS_dir: Gaussian Splatting data directory
            records_name: Records folder name
        """
        self.GS_records_dir = os.path.join(GS_dir, records_name)
    
    def clean_mask_folder(
        self,
        record_name: str,
        cam_name: str,
        rgb_folder_name: str,
        target_colors: List[Tuple[int, int, int]],
        output_suffix: str = '_clean',
        **kwargs
    ):
        """
        Clean mask folder for a specific record and camera.
        
        Args:
            record_name: Record name
            cam_name: Camera name, e.g., 'cam_0'
            rgb_folder_name: RGB folder name, e.g., 'rgb_mask_object'
            target_colors: Target color list (BGR format, 0-255)
            output_suffix: Output folder suffix
                - '_clean': rgb_mask_object → rgb_mask_object_clean
                - '': In-place replacement (overwrite original files)
            **kwargs: Additional parameters
                - min_object_size: Minimum object size (pixels)
                - kernel_size: Morphological kernel size
                - tolerance: Color tolerance
        """
        # Input and output folders
        input_folder = os.path.join(
            self.GS_records_dir, record_name, cam_name, rgb_folder_name
        )
        if output_suffix:
            output_folder_name = rgb_folder_name + output_suffix
        else:
            output_folder_name = rgb_folder_name
        
        output_folder = os.path.join(
            self.GS_records_dir, record_name, cam_name, output_folder_name
        )
        if not os.path.exists(input_folder):
            print(f"[Warning] Input folder does not exist: {input_folder}")
            return
        
        self.batch_clean_masks(
            input_folder, output_folder,
            target_colors,
            file_pattern='*.png',
            **kwargs
        )
        
        print(f"Mask cleaned: {output_folder_name}")
    
    def clean_all_masks_in_record(
        self,
        record_name: str,
        mask_configs: List[Dict],
        cameras: Optional[List[str]] = None
    ):
        """
        Clean all masks in a record according to configuration.
        
        Args:
            record_name: Record name
            mask_configs: List of mask configuration dictionaries
            cameras: List of camera names, None means all cameras
        """
        if cameras is None:
            record_dir = os.path.join(self.GS_records_dir, record_name)
            cameras = [d for d in os.listdir(record_dir) if d.startswith('cam_')]
        
        for cam_name in cameras:
            print(f"\n[Info] Processing camera: {cam_name}")
            for config in mask_configs:
                rgb_folder = config['rgb_folder']
                target_colors = config['target_colors']
                method = config.get('method', 'nearest')
                output_suffix = config.get('output_suffix', '_clean')
                # Extract additional parameters
                extra_params = {k: v for k, v in config.items() 
                               if k not in ['rgb_folder', 'target_colors', 'method', 'output_suffix']}
                self.clean_mask_folder(
                    record_name, cam_name, rgb_folder,
                    target_colors, output_suffix,
                    **extra_params
                )
    
    def clean_all_records(
        self,
        mask_configs: List[Dict],
        cameras: Optional[List[str]] = None,
        record_indices: Optional[List[int]] = None
    ):
        """
        Clean masks in all records according to configuration.
        
        Args:
            mask_configs: List of mask configuration dictionaries
            cameras: List of camera names, None means all cameras
            record_indices: List of record indices to process, None means all records
        """
        all_records = sorted([
            d for d in os.listdir(self.GS_records_dir)
            if d.startswith('record_')
        ])
        
        if record_indices is not None:
            records_to_process = [all_records[i] for i in record_indices]
        else:
            records_to_process = all_records
        
        print(f"[Info] Starting batch cleaning, {len(records_to_process)} records")
        
        for record_name in tqdm(records_to_process, desc="Processing records", unit="record"):
            self.clean_all_masks_in_record(record_name, mask_configs, cameras)
        
        print("[Info] All record masks cleaned")

    def batch_clean_masks(
        self,
        input_folder: str,
        output_folder: str,
        target_colors: List[Tuple[int, int, int]],
        file_pattern: str = '*.png',
        **kwargs
    ):
        """
        Batch clean masks in a folder.
        
        Args:
            input_folder: Input folder path
            output_folder: Output folder path
            target_colors: Target color list
            file_pattern: File matching pattern
            **kwargs: Additional parameters passed to cleaning functions
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        pattern = os.path.join(input_folder, file_pattern)
        image_files = sorted(glob.glob(pattern))
        
        print(f"[Info] Starting batch mask cleaning, {len(image_files)} images")
        print(f"[Info] Target colors: {target_colors}")
        
        cleaner = MaskCleaner(target_colors)
        
        for img_path in tqdm(image_files, desc="Cleaning masks", unit="image"):
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Cannot read image: {img_path}")
                continue
            
            # Clean
            cleaned = cleaner.nearest_color_mapping(img, target_colors)
            kernel_size = kwargs.get('kernel_size', 3)
            if kernel_size > 0:
                cleaned = cleaner.morphology_clean(cleaned, kernel_size, 'open-close')

            # Save
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cleaned)
        
        print(f"[Info] Batch cleaning completed, output to: {output_folder}")

if __name__ == "__main__":
    """Usage example"""
    
    # Example 1: Clean single mask folder
    processor = MaskProcessor(
        GS_dir="/home/student/new_data/GS_Data",
        records_name="records_1024_test"
    )
    
    processor.clean_mask_folder(
        record_name='record_0',
        cam_name='cam_0',
        rgb_folder_name='rgb_mask_object',
        target_colors=[(0, 0, 0), (255, 255, 255)],  # Black and white
        min_object_size=100,
        kernel_size=5
    )
    
    # Example 2: Batch clean multiple masks
    mask_configs = [
        {
            'rgb_folder': 'rgb_mask_object',
            'target_colors': [(0, 0, 0), (255, 255, 255)],
            'min_object_size': 100,
            'kernel_size': 5
        },
        {
            'rgb_folder': 'rgb_mask_instance',
            'target_colors': [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)],
            'min_object_size': 50,
            'kernel_size': 3
        }
    ]
    
    processor.clean_all_masks_in_record(
        'record_0',
        mask_configs,
        cameras=['cam_0', 'cam_1']
    )
    
    print("\n✅ Test completed")

