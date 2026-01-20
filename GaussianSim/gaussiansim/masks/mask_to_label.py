"""
Mask color to label conversion utility.
Convert solid color masks to single-channel numeric label maps.

Color mapping rules:
- Background (black) → 0
- Robot (blue) → 1
- object_0 (first object) → 2
- object_1 (second object) → 3
- object_n → n+2
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class MaskToLabelConverter:
    """
    Mask to label converter.
    Convert RGB color masks to single-channel numeric labels.
    """
    
    def __init__(
        self,
        color_label_config: Dict[Tuple[int, int, int], int]
    ):
        """
        Initialize converter.
        
        Args:
            color_label_config: Color to label mapping dictionary, format {(B, G, R): label_value}
                               Keys are BGR format color tuples (0-255 range), values are corresponding label values
        
        Example:
            converter = MaskToLabelConverter(
                color_label_config={
                    (0, 0, 0): 0,        # Black → label 0
                    (255, 0, 0): 1,      # Blue → label 1
                    (0, 0, 255): 5,      # Red → label 5
                    (0, 255, 0): 10,     # Green → label 10
                }
            )
        """
        self.color_label_config = color_label_config
        
        # 构建完整的颜色到标签映射
        self._build_color_map()
    
    def _build_color_map(self):
        """Build color to label mapping dictionary."""
        self.color_to_label = {}
        
        # Build mapping directly from color_label_config
        for color_bgr, label in self.color_label_config.items():
            # Ensure color is tuple format
            if isinstance(color_bgr, (list, np.ndarray)):
                color_bgr = tuple(color_bgr)
            self.color_to_label[color_bgr] = label
        
        # Reverse mapping (label → color)
        self.label_to_color = {v: k for k, v in self.color_to_label.items()}
        
        print(f"[Info] Color mapping table built:")
        # Display sorted by label value
        sorted_items = sorted(self.color_to_label.items(), key=lambda x: x[1])
        for color_bgr, label in sorted_items:
            print(f"  {label}: BGR{color_bgr}")
    
    def convert_single_image(
        self,
        rgb_mask: np.ndarray,
        unknown_as_background: bool = True
    ) -> np.ndarray:
        """
        Convert single RGB mask to single-channel label map.
        
        Args:
            rgb_mask: Input RGB mask (BGR format, uint8, shape=(H,W,3))
            unknown_as_background: Whether unknown colors should be treated as background (0), otherwise keep as 255
        
        Returns:
            Single-channel label map (uint8, shape=(H,W))
        
        Example:
            Input: RGB image with red box, green dumbbell, blue circle
            Output: Grayscale image with value 2 regions, value 3 regions, value 1 regions
        """
        h, w = rgb_mask.shape[:2]
        label_map = np.zeros((h, w), dtype=np.uint8)
        
        # Map each pixel to label
        for color_bgr, label in self.color_to_label.items():
            # Create mask for this color
            color_arr = np.array(color_bgr, dtype=np.uint8)
            mask = np.all(rgb_mask == color_arr, axis=2)
            label_map[mask] = label
        
        # Handle unknown colors
        if not unknown_as_background:
            # Find all unmapped pixels
            mapped_mask = np.zeros((h, w), dtype=bool)
            for color_bgr in self.color_to_label.keys():
                color_arr = np.array(color_bgr, dtype=np.uint8)
                mapped_mask |= np.all(rgb_mask == color_arr, axis=2)
            
            # Mark unmapped pixels as 255
            label_map[~mapped_mask] = 255
        
        return label_map
    
    def convert_single_image_fast(
        self,
        rgb_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fast conversion using vectorized operations.
        
        Args:
            rgb_mask: Input RGB mask
        
        Returns:
            Single-channel label map
        """
        h, w = rgb_mask.shape[:2]
        
        # Reshape RGB image to pixel list
        pixels = rgb_mask.reshape(-1, 3)
        
        # Initialize label array (default 0=background)
        labels = np.zeros(pixels.shape[0], dtype=np.uint8)
        
        # Assign labels for each color
        for color_bgr, label in self.color_to_label.items():
            color_arr = np.array(color_bgr, dtype=np.uint8)
            # Find all pixels matching this color
            matches = np.all(pixels == color_arr, axis=1)
            labels[matches] = label
        
        # Reshape back to image shape
        label_map = labels.reshape(h, w)
        
        return label_map
    
    def convert_and_save(
        self,
        input_path: str,
        output_path: str,
        use_fast: bool = True
    ):
        """
        Read RGB mask, convert and save as label map.
        
        Args:
            input_path: Input RGB mask path
            output_path: Output label map path
            use_fast: Whether to use fast method
        """
        # Read
        rgb_mask = cv2.imread(input_path)
        if rgb_mask is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")
        
        # Convert
        if use_fast:
            label_map = self.convert_single_image_fast(rgb_mask)
        else:
            label_map = self.convert_single_image(rgb_mask)
        
        # Save as grayscale (single channel)
        cv2.imwrite(output_path, label_map)
    
    def batch_convert_folder(
        self,
        input_folder: str,
        output_folder: str,
        file_pattern: str = '*.png'
    ):
        """
        Batch convert all masks in folder.
        
        Args:
            input_folder: Input folder path
            output_folder: Output folder path
            file_pattern: File matching pattern
        """
        import glob
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        pattern = os.path.join(input_folder, file_pattern)
        image_files = sorted(glob.glob(pattern))
        
        if len(image_files) == 0:
            print(f"[Warning] No image files found: {input_folder}")
            return
        
        print(f"[Info] Starting conversion, {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc="Converting to labels", unit="img"):
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, filename)
            
            self.convert_and_save(img_path, output_path, use_fast=True)
        
        print(f"[Info] Batch conversion complete, output to: {output_folder}")
    
    def get_label_statistics(self, label_map: np.ndarray) -> Dict[int, int]:
        """
        Get label map statistics.
        
        Args:
            label_map: Label map
        
        Returns:
            Dictionary {label: pixel_count}
        """
        unique, counts = np.unique(label_map, return_counts=True)
        stats = dict(zip(unique.tolist(), counts.tolist()))
        
        return stats
    
    def visualize_label_map(
        self,
        label_map: np.ndarray,
        colormap: str = 'viridis'
    ) -> np.ndarray:
        """
        Visualize label map as pseudo-color image (for debugging).
        
        Args:
            label_map: Single-channel label map
            colormap: OpenCV colormap name
        
        Returns:
            Pseudo-color image
        """
        # Normalize to 0-255
        if label_map.max() > 0:
            normalized = (label_map * (255 // label_map.max())).astype(np.uint8)
        else:
            normalized = label_map.astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored


class MaskLabelBatchProcessor:
    """
    Batch processor for automatically processing mask folders in all records.
    """
    
    def __init__(
        self,
        GS_dir: str,
        records_name: str,
        color_label_config: Dict[Tuple[int, int, int], int]
    ):
        """
        Initialize batch processor.
        
        Args:
            GS_dir: GS data directory
            records_name: Records name
            color_label_config: Color to label mapping dictionary, format {(B, G, R): label_value}
                               Keys are BGR format color tuples (0-255 range), values are corresponding label values
        
        Example:
            processor = MaskLabelBatchProcessor(
                GS_dir, records_name,
                color_label_config={
                    (0, 0, 0): 0,        # Black → label 0
                    (255, 0, 0): 1,      # Blue → label 1
                    (0, 0, 255): 5,      # Red → label 5
                    (0, 255, 0): 10,     # Green → label 10
                }
            )
        """
        self.GS_records_dir = os.path.join(GS_dir, records_name)
        self.converter = MaskToLabelConverter(color_label_config=color_label_config)
    
    def process_single_folder(
        self,
        record_name: str,
        cam_name: str,
        rgb_folder_name: str,
        output_suffix: str = '_labels'
    ):
        """
        Process single mask folder.
        
        Args:
            record_name: Record name, e.g., 'record_0'
            cam_name: Camera name, e.g., 'cam_0'
            rgb_folder_name: RGB mask folder name, e.g., 'masks_clean'
            output_suffix: Output folder suffix, e.g., '_labels'
        
        Example:
            processor.process_single_folder(
                'record_0', 'cam_0', 'masks_clean'
            )
            # Output to: cam_0/masks_clean_labels/
        """
        input_folder = os.path.join(
            self.GS_records_dir, record_name, cam_name, rgb_folder_name
        )
        
        output_folder_name = rgb_folder_name + output_suffix
        output_folder = os.path.join(
            self.GS_records_dir, record_name, cam_name, output_folder_name
        )
        
        if not os.path.exists(input_folder):
            print(f"[Warning] Input folder not found: {input_folder}")
            return
        
        self.converter.batch_convert_folder(input_folder, output_folder)
        
        # Verify first frame
        first_file = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])[0]
        first_label = cv2.imread(os.path.join(output_folder, first_file), cv2.IMREAD_GRAYSCALE)
        
        stats = self.converter.get_label_statistics(first_label)
        print(f"[Info] First frame label distribution: {stats}")
        print(f"  ✅ Output to: {output_folder_name}")
    
    def process_record(
        self,
        record_name: str,
        cam_names: List[str],
        rgb_folder_names: List[str],
        output_suffix: str = '_labels'
    ):
        """
        Process multiple mask folders for multiple cameras in a record.
        
        Args:
            record_name: Record name
            cam_names: List of camera names, e.g., ['cam_0', 'cam_1']
            rgb_folder_names: List of RGB mask folder names to process
            output_suffix: Output suffix
        
        Example:
            processor.process_record(
                'record_0',
                cam_names=['cam_0', 'cam_1'],
                rgb_folder_names=['masks_clean', 'mask_object_0_clean']
            )
        """
        print(f"\n[Info] Processing record: {record_name}")
        
        for cam_name in cam_names:
            print(f"\n  Processing camera: {cam_name}")
            
            for rgb_folder_name in rgb_folder_names:
                self.process_single_folder(
                    record_name, cam_name, rgb_folder_name, output_suffix
                )
    
    def process_all_records(
        self,
        cam_names: List[str],
        rgb_folder_names: List[str],
        output_suffix: str = '_labels',
        record_indices: Optional[List[int]] = None
    ):
        """
        Process all records.
        
        Args:
            cam_names: List of camera names
            rgb_folder_names: List of RGB mask folder names
            output_suffix: Output suffix
            record_indices: List of record indices, None means all
        
        Example:
            processor.process_all_records(
                cam_names=['cam_0', 'cam_1'],
                rgb_folder_names=['masks_clean'],
                output_suffix='_labels'
            )
        """
        # Get all records
        all_records = sorted([
            d for d in os.listdir(self.GS_records_dir)
            if d.startswith('record_') and os.path.isdir(os.path.join(self.GS_records_dir, d))
        ])
        
        # Filter records to process
        if record_indices is not None:
            records_to_process = [all_records[i] for i in record_indices]
        else:
            records_to_process = all_records
        
        print(f"[Info] Starting batch conversion, {len(records_to_process)} records")
        
        for record_name in tqdm(records_to_process, desc="Processing records", unit="record"):
            self.process_record(record_name, cam_names, rgb_folder_names, output_suffix)
        
        print("\n✅ All record masks converted to label maps")


def create_label_colormap_image(
    label_map: np.ndarray,
    color_mapping: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Convert label map back to color image (for visualization verification).
    
    Args:
        label_map: Single-channel label map
        color_mapping: Label to color mapping, e.g., {0: (0,0,0), 1: (255,0,0), 2: (0,0,255)}
    
    Returns:
        RGB image
    """
    h, w = label_map.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label, color in color_mapping.items():
        mask = label_map == label
        rgb_image[mask] = color
    
    return rgb_image


def auto_detect_colors_from_mask(
    rgb_mask_path: str,
    exclude_background: bool = True
) -> List[Tuple[int, int, int]]:
    """
    Automatically detect all distinct colors from RGB mask.
    
    Args:
        rgb_mask_path: RGB mask image path
        exclude_background: Whether to exclude black background
    
    Returns:
        List of detected colors in BGR format
    """
    img = cv2.imread(rgb_mask_path)
    unique_colors = np.unique(img.reshape(-1, 3), axis=0)
    
    colors = [tuple(c) for c in unique_colors]
    
    if exclude_background:
        # Exclude black
        colors = [c for c in colors if c != (0, 0, 0)]
    
    return colors


def verify_label_map(
    label_map: np.ndarray,
    expected_labels: List[int]
) -> bool:
    """
    Verify that label map only contains expected label values.
    
    Args:
        label_map: Label map
        expected_labels: List of expected labels, e.g., [0, 1, 2, 3]
    
    Returns:
        Whether verification passed
    """
    unique_labels = np.unique(label_map).tolist()
    
    unexpected = [l for l in unique_labels if l not in expected_labels]
    
    if unexpected:
        print(f"[Warning] Unexpected labels found: {unexpected}")
        return False
    
    print(f"[Info] Label verification passed, contains: {unique_labels}")
    return True









# ========== Single channel to 3-channel conversion ==========

def convert_single_channel_to_3channel(
    input_image: np.ndarray
) -> np.ndarray:
    """
    Convert single-channel label map to 3-channel image (all channels have same values).
    
    Args:
        input_image: Single-channel image (H, W) or (H, W, 1)
    
    Returns:
        3-channel image (H, W, 3), all channels have same values
    
    Example:
        Input: shape=(480, 640), pixel values=[0, 1, 2, 3]
        Output: shape=(480, 640, 3), RGB three channels all have same values
    """
    # Ensure 2D array
    if len(input_image.shape) == 3:
        input_image = input_image[:, :, 0]
    
    # Copy to 3 channels
    output_image = np.stack([input_image, input_image, input_image], axis=2)
    
    return output_image


def convert_label_folder_to_3channel(
    input_folder: str,
    output_folder: str,
    file_pattern: str = '*.png'
):
    """
    Batch convert single-channel label map folder to 3-channel.
    
    Args:
        input_folder: Input folder path (single-channel label maps)
        output_folder: Output folder path (3-channel label maps)
        file_pattern: File matching pattern
    
    Example:
        convert_label_folder_to_3channel(
            input_folder="/path/to/masks_clean_labels",
            output_folder="/path/to/masks_clean_labels_3"
        )
    """
    import glob
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    pattern = os.path.join(input_folder, file_pattern)
    image_files = sorted(glob.glob(pattern))
    
    if len(image_files) == 0:
        print(f"[Warning] No image files found: {input_folder}")
        return
    
    print(f"[Info] Starting single-channel → 3-channel conversion, {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc="Converting to 3-channel", unit="img"):
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        
        # Read single-channel image
        label_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            print(f"[Warning] Cannot read: {img_path}")
            continue
        
        # Convert to 3-channel
        label_3ch = convert_single_channel_to_3channel(label_img)
        
        # Save
        cv2.imwrite(output_path, label_3ch)
    
    print(f"[Info] Conversion complete, output to: {output_folder}")
    
    # Verify first image
    if len(image_files) > 0:
        first_file = os.path.basename(image_files[0])
        first_output = cv2.imread(os.path.join(output_folder, first_file))
        print(f"[Info] Output verification:")
        print(f"  - shape: {first_output.shape}")
        print(f"  - dtype: {first_output.dtype}")
        print(f"  - Channel consistency: {np.all(first_output[:,:,0] == first_output[:,:,1]) and np.all(first_output[:,:,1] == first_output[:,:,2])}")


class LabelTo3ChannelBatchProcessor:
    """
    Batch processor for converting single-channel label maps to 3-channel.
    Automatically processes label folders in all records.
    """
    
    def __init__(self, GS_dir: str, records_name: str):
        """
        Initialize batch processor.
        
        Args:
            GS_dir: GS data directory
            records_name: Records name
        """
        self.GS_records_dir = os.path.join(GS_dir, records_name)
    
    def process_single_folder(
        self,
        record_name: str,
        cam_name: str,
        label_folder_name: str,
        output_suffix: str = '_3'
    ):
        """
        Process single label folder.
        
        Args:
            record_name: Record name, e.g., 'record_0'
            cam_name: Camera name, e.g., 'cam_0'
            label_folder_name: Label folder name, e.g., 'masks_clean_labels'
            output_suffix: Output folder suffix, e.g., '_3'
        
        Example:
            processor.process_single_folder(
                'record_0', 'cam_0', 'masks_clean_labels'
            )
            # Output to: cam_0/masks_clean_labels_3/
        """
        input_folder = os.path.join(
            self.GS_records_dir, record_name, cam_name, label_folder_name
        )
        
        output_folder_name = label_folder_name + output_suffix
        output_folder = os.path.join(
            self.GS_records_dir, record_name, cam_name, output_folder_name
        )
        
        if not os.path.exists(input_folder):
            print(f"[Warning] Input folder not found: {input_folder}")
            return
        
        convert_label_folder_to_3channel(input_folder, output_folder)
        print(f"  ✅ Output to: {output_folder_name}")
    
    def process_record(
        self,
        record_name: str,
        cam_names: List[str],
        label_folder_names: List[str],
        output_suffix: str = '_3'
    ):
        """
        Process multiple label folders for multiple cameras in a record.
        
        Args:
            record_name: Record name
            cam_names: List of camera names, e.g., ['cam_0', 'cam_1']
            label_folder_names: List of label folder names, e.g., ['masks_clean_labels']
            output_suffix: Output suffix
        
        Example:
            processor.process_record(
                'record_0',
                cam_names=['cam_0'],
                label_folder_names=['masks_clean_labels']
            )
        """
        print(f"\n[Info] Processing record: {record_name}")
        
        for cam_name in cam_names:
            print(f"\n  Processing camera: {cam_name}")
            
            for label_folder_name in label_folder_names:
                self.process_single_folder(
                    record_name, cam_name, label_folder_name, output_suffix
                )
    
    def process_all_records(
        self,
        cam_names: List[str],
        label_folder_names: List[str],
        output_suffix: str = '_3',
        record_indices: Optional[List[int]] = None
    ):
        """
        Process all records.
        
        Args:
            cam_names: List of camera names
            label_folder_names: List of label folder names
            output_suffix: Output suffix
            record_indices: List of record indices, None means all
        
        Example:
            processor.process_all_records(
                cam_names=['cam_0'],
                label_folder_names=['masks_clean_labels'],
                output_suffix='_3'
            )
        """
        # Get all records
        all_records = sorted([
            d for d in os.listdir(self.GS_records_dir)
            if d.startswith('record_') and os.path.isdir(os.path.join(self.GS_records_dir, d))
        ])
        
        # Filter records to process
        if record_indices is not None:
            records_to_process = [all_records[i] for i in record_indices]
        else:
            records_to_process = all_records
        
        print(f"[Info] Starting batch single-channel → 3-channel conversion, {len(records_to_process)} records")
        
        for record_name in tqdm(records_to_process, desc="Processing records", unit="record"):
            self.process_record(record_name, cam_names, label_folder_names, output_suffix)
        
        print("\n✅ All record single-channel labels converted to 3-channel")





if __name__ == "__main__":
    """Test example"""
    print("="*70)
    print("Mask to Label Conversion Tool Test")
    print("="*70)
    
    # Create test image (RGB mask)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background: black (label 0)
    # Already black
    
    # Robot: blue (label 1)
    cv2.circle(test_img, (320, 100), 80, (255, 0, 0), -1)
    
    # object_0: red (label 2)
    cv2.rectangle(test_img, (100, 200), (300, 400), (0, 0, 255), -1)
    
    # object_1: green (label 3)
    cv2.rectangle(test_img, (350, 250), (550, 450), (0, 255, 0), -1)
    
    # Initialize converter
    converter = MaskToLabelConverter(
        color_label_config={
            (0, 0, 0): 0,        # Black background → label 0
            (255, 0, 0): 1,      # Blue robot → label 1
            (0, 0, 255): 2,      # Red object_0 → label 2
            (0, 255, 0): 3,      # Green object_1 → label 3
        }
    )
    
    # Convert
    label_map = converter.convert_single_image_fast(test_img)
    
    # Statistics
    stats = converter.get_label_statistics(label_map)
    print(f"\nLabel distribution:")
    for label, count in stats.items():
        print(f"  Label {label}: {count} pixels")
    
    # Verify
    is_valid = verify_label_map(label_map, [0, 1, 2, 3])
    
    # Save test results
    output_dir = '/tmp/mask_label_test'
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f'{output_dir}/test_rgb_mask.png', test_img)
    cv2.imwrite(f'{output_dir}/test_label_map.png', label_map)
    
    # Visualize
    vis = converter.visualize_label_map(label_map)
    cv2.imwrite(f'{output_dir}/test_label_vis.png', vis)
    
    print(f"\n✅ Test images saved to: {output_dir}/")
    print(f"  - test_rgb_mask.png    (RGB mask)")
    print(f"  - test_label_map.png   (Label map, single channel)")
    print(f"  - test_label_vis.png   (Label visualization)")
    
    # Verify label values
    print(f"\nLabel map unique values: {np.unique(label_map)}")
    print(f"Label map dtype: {label_map.dtype}")
    print(f"Label map shape: {label_map.shape}")
    
    print("\n" + "="*70)
    print("✅ Test complete")
    print("="*70)

