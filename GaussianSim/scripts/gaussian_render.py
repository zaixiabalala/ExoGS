"""
Gaussian Splatting rendering script.
Generates RGB images, masks, and color augmentation variants.
"""

import os
import sys
import shutil
import torch
import logging
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussiansim.utils.logging_config import setup_logging
from gaussiansim.utils.color_config import ColorConfig
from gaussiansim.camera.generator import CamerasJSONGenerator
from gaussiansim.gaussiansim.multicolor_renderer import GaussianSceneMulticolorRenderer
from gaussiansim.gaussiansim.config import RenderConfigBuilder
from gaussiansim.camera.constants import CAM_EXTRINSIC_MATRIX_L
from gaussiansim.utils.mask_post_processor import MaskPostProcessor
from gaussiansim.masks.mask_to_label import MaskLabelBatchProcessor, LabelTo3ChannelBatchProcessor


def get_config():
    """
    Get rendering configuration.
    
    Key parameters:
    - GS_dir: Output directory for Gaussian Splatting data
    - Origin_dir: Source data directory containing original records
    - robot_type: Robot type to use, either "flexiv" or "franka" (default: "flexiv")
    - records_name: Name of the records folder to process
    - scene_name: Name of the scene to render
    - objects: List of object names to include in rendering
    - batch_size: Number of images to process in each batch (affects GPU memory usage)
    - camera_name_mapping: Mapping from camera index to camera name (e.g., {0: 'cam_0'})
    - scene_noise_range: Noise range for scene position [x_range, y_range, z_range]
    - color_index_list: List of color variant indices to generate
    - background_folder: Path to background images folder
    - bg_color_range_default: Background color range for original images (min, max) in 0-1 RGB
    - bg_color_range_aug: Background color range for augmented images (min, max) in 0-1 RGB
    - bg_image_index_list: List of background image indices to use
        If you choose to render Gaussian scene model instead of using background images, 
        you can use pure black images as background
    - color_ranges: Color augmentation ranges for scene, objects, robot, and background
    - force_overwrite: If True, delete existing records folder before rendering (default: False)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config = {
        'GS_dir': r"/home/ubuntu/data/GS_Data",  # Output directory
        'Origin_dir': r"/media/ubuntu/B0A8C06FA8C0361E/Data/Origin_Data",  # Source data directory
        'records_name': "records_dumbbell_1017",  # Records folder name
        'robot_type': "franka",
        'force_overwrite': True,  # If True, delete existing records folder before rendering
        'scene_name': "scene_steel",  # Scene name
        'objects': ['square_white_orange_box', "banana"],  # Object list
        'batch_size': 20,  # Batch size for processing
        'camera_name_mapping': {0: 'cam_0'},  # Camera index to name mapping
        'scene_noise_range': [[0, 0], [0, 0], [0, 0]],  # [x_range, y_range, z_range]
        'color_index_list': [0],  # Color variant indices
        'background_folder': os.path.join(project_root, "assets", "bg_images"),  # Background images path
        'bg_color_range_default': ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),  # Default background color range
        'bg_color_range_aug': ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)),  # Augmented background color range
        'bg_image_index_list': [0],  # Background image indices
        'color_ranges': {
            "scene_range": ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)),  # Scene color augmentation range
            "objects_range": ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)),  # Objects color augmentation range
            "robot_range": ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)),  # Robot color augmentation range
            "bg_range": ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))  # Background color augmentation range
        }
    }
    return config


def setup_mask_color_config(objects: List[str], object_color_rgb_255=(255, 0, 0)):
    """
    Setup mask color configuration.
    
    Key parameters:
    - objects: List of object names (automatically generates colors for each object)
    - object_color_rgb_255: RGB color (0-255) for all objects, default is red (255, 0, 0)
    
    Color configuration:
    - scene_color_rgb_255: Scene color in RGB (0-255), default is black (0, 0, 0)
    - bg_color_rgb_255: Background color in RGB (0-255), default is black (0, 0, 0)
    - robot_color_rgb_255: Robot color in RGB (0-255), default is blue (0, 0, 255)
    - objects_colors_rgb_255: Auto-generated list of object colors based on object count
    
    Mask cleaning configuration:
    - rgb_folder: Input folder name for masks (default: 'masks')
    - method: Cleaning method (default: 'instance' for instance segmentation)
    - output_suffix: Output folder suffix (default: '_clean')
    - min_object_size: Minimum object size in pixels (default: 50)
    - kernel_size: Morphological kernel size for noise removal (default: 3)
    
    Args:
        objects: List of object names
        object_color_rgb_255: RGB color (0-255) for all objects, default is red
    """
    # Auto-generate object colors list based on number of objects
    objects_colors_rgb_255 = [object_color_rgb_255] * len(objects)
    
    color_config = ColorConfig(
        scene_color_rgb_255=(0, 0, 0),  # Scene color: black
        bg_color_rgb_255=(0, 0, 0),  # Background color: black
        robot_color_rgb_255=(0, 0, 255),  # Robot color: blue
        objects_colors_rgb_255=objects_colors_rgb_255  # Object colors: auto-generated
    )
    
    mask_clean_configs = [color_config.get_mask_clean_config(
        rgb_folder='masks',  # Input mask folder
        method='instance',  # Instance segmentation method
        output_suffix='_clean',  # Output folder suffix
        min_object_size=50,  # Minimum object size (pixels)
        kernel_size=3  # Morphological kernel size
    )]
    
    return color_config, mask_clean_configs


def setup_cameras(GS_dir: str, records_name: str, logger: logging.Logger):
    """Setup camera poses."""
    logger.info("Generating camera pose JSON...")
    camera_json_generator = CamerasJSONGenerator(
        json_output_path=os.path.join(GS_dir, records_name, "transforms.json")
    )
    camera_json_generator.add_camera(CAM_EXTRINSIC_MATRIX_L)
    camera_json_generator.add_spherical_cameras(
        camera_extrinsic_matrix=CAM_EXTRINSIC_MATRIX_L, 
        phi_values=[35], 
        camera_counts=[2]
    )
    camera_json_generator.save_json()


def render_original_images(
    renderer: GaussianSceneMulticolorRenderer,
    background_folder: str,
    bg_image_index_list: List[int],
    bg_color_range: Tuple,
    scene_noise_range: List[List[int]],
    logger: logging.Logger
):
    """Render original color images with background variants."""
    logger.info("Mode 1: Generate original color images")
    for bg_idx in bg_image_index_list:
        logger.info(f"  Rendering rgbs with background {bg_idx}")
        config = (RenderConfigBuilder()
            .outputs(rgb=True, depth=True, alpha=True, ply=False)
            .background(
                folder=background_folder,
                color_range=bg_color_range,
                image_index=bg_idx,
                consistent_color=True
            )
            .scene_noise(
                x_range=(scene_noise_range[0][0], scene_noise_range[0][1]),
                y_range=(scene_noise_range[1][0], scene_noise_range[1][1]),
                z_range=(scene_noise_range[2][0], scene_noise_range[2][1])
            )
            .output_folders([f'rgbs_bg{bg_idx}'])
            .build())
        renderer.render(config, camera_type="fixed")


def render_masks(
    renderer: GaussianSceneMulticolorRenderer,
    scene_color,
    objects_colors,
    robot_color,
    bg_color,
    GS_dir: str,
    records_name: str,
    color_config: ColorConfig,
    mask_clean_configs: List[Dict],
    camera_name_mapping: Dict,
    logger: logging.Logger
):
    """
    Render classification masks and post-process them.
    
    Post-processing includes: cleaning masks, converting to labels, and generating 3-channel labels.
    """
    logger.info("Mode 2: Generate classification mask")
    config = (RenderConfigBuilder()
        .outputs(rgb=True, depth=False)
        .mask_mode(
            scene_color=scene_color,
            objects_colors=objects_colors,
            robot_color=robot_color,
            bg_color=bg_color
        )
        .output_folders(['masks'])
        .build())
    renderer.render(config, camera_type="fixed")
    
    # Post-process masks: clean, convert to labels, and generate 3-channel labels
    logger.info("Post-processing: Clean mask")
    try:
        processor = MaskPostProcessor(GS_dir, records_name)
        processor.clean_all_records(mask_clean_configs, cameras=list(camera_name_mapping.values()))
        
        # Build color_label_config: BGR color -> label value
        bg_color_bgr = color_config.target_colors_bgr[0]
        color_label_config = {bg_color_bgr: 0}
        color_label_config[color_config.robot_color_bgr] = 1
        for obj_color_bgr in color_config.objects_colors_bgr:
            color_label_config[obj_color_bgr] = 2
        
        # Convert masks to labels
        label_processor = MaskLabelBatchProcessor(GS_dir, records_name, color_label_config)
        label_processor.process_all_records(
            cam_names=list(camera_name_mapping.values()),
            rgb_folder_names=['masks_clean'],
            output_suffix='_labels'
        )
        
        # Convert labels to 3-channel
        label_3channel_processor = LabelTo3ChannelBatchProcessor(GS_dir, records_name)
        label_3channel_processor.process_all_records(
            cam_names=list(camera_name_mapping.values()),
            label_folder_names=['masks_clean_labels'],
            output_suffix='_3'
        )
    except Exception as e:
        logger.error(f"Mask post-processing failed: {e}", exc_info=True)


def render_color_augmentation(
    renderer: GaussianSceneMulticolorRenderer,
    background_folder: str,
    color_index_list: List[int],
    bg_image_index_list: List[int],
    bg_color_range: Tuple,
    color_ranges: Dict,
    scene_noise_range: List[List[int]],
    logger: logging.Logger
):
    """Render color augmentation variants."""
    logger.info("Mode 3: Generate color augmentation variants")
    try:
        for color_idx in color_index_list:
            for bg_idx in bg_image_index_list:
                logger.info(f"  Rendering rgb_{color_idx} with background {bg_idx}")
                config = (RenderConfigBuilder()
                    .outputs(rgb=True, depth=False, ply=False, alpha=False)
                    .background(
                        folder=background_folder,
                        color_range=bg_color_range,
                        image_index=bg_idx,
                        consistent_color=True
                    )
                    .color_augmentation(
                        scene_range=color_ranges.get("scene_range"),
                        objects_range=color_ranges.get("objects_range"),
                        robot_range=color_ranges.get("robot_range"),
                        bg_range=color_ranges.get("bg_range"),
                        seed=42 + color_idx * 1000
                    )
                    .scene_noise(
                        x_range=(scene_noise_range[0][0], scene_noise_range[0][1]),
                        y_range=(scene_noise_range[1][0], scene_noise_range[1][1]),
                        z_range=(scene_noise_range[2][0], scene_noise_range[2][1])
                    )
                    .output_folders([f'rgb_{color_idx}_bg{bg_idx}'])
                    .build())
                renderer.render(config, camera_type="fixed")
    except Exception as e:
        logger.error(f"Color augmentation failed: {e}", exc_info=True)


# Setup logging
setup_logging(log_level=logging.INFO, log_file="logs/demo_gaussian_render.log")
logger = logging.getLogger(__name__)

def main():
    """
    Main execution function.
    
    Note: Configuration should be modified in:
    - get_config(): Rendering parameters (paths, scene, objects, batch size, etc.)
    - setup_mask_color_config(): Mask color configuration (colors, cleaning parameters)
    """
    torch.cuda.set_device(0)
    
    # Load configuration
    # Modify parameters in get_config() function above
    config = get_config()
    # Modify mask color configuration in setup_mask_color_config() function above
    color_config, mask_clean_configs = setup_mask_color_config(config['objects'])
    
    # Check and delete existing records folder if force_overwrite is True
    if config.get('force_overwrite', False):
        records_dir = os.path.join(config['GS_dir'], config['records_name'])
        if os.path.exists(records_dir):
            logger.info(f"force_overwrite is True, deleting existing records folder: {records_dir}")
            try:
                shutil.rmtree(records_dir)
                logger.info(f"Successfully deleted records folder: {records_dir}")
            except Exception as e:
                logger.error(f"Failed to delete records folder {records_dir}: {e}")
                raise
        else:
            logger.info(f"Records folder does not exist, skipping deletion: {records_dir}")
    
    # Setup cameras
    setup_cameras(config['GS_dir'], config['records_name'], logger)
    
    # Initialize renderer
    logger.info(f"Initializing renderer with robot type: {config['robot_type']}...")
    renderer = GaussianSceneMulticolorRenderer(
        Origin_dir=config['Origin_dir'],
        GS_dir=config['GS_dir'],
        records_name=config['records_name'],
        scene_name=config['scene_name'],
        objects=config['objects'],
        enable_cache=True,
        batch_size=config['batch_size'],
        robot_type=config['robot_type']
    )
    renderer.fixed_cameras_init(
        json_name="transforms.json",
        camera_name_mapping=config['camera_name_mapping']
    )
    
    # Collect angles and tcps data
    renderer.collect_angles()
    renderer.collect_tcps()
    
    # Render modes
    render_original_images(
        renderer,
        config['background_folder'],
        config['bg_image_index_list'],
        config['bg_color_range_default'],
        config['scene_noise_range'],
        logger
    )
    
    render_masks(
        renderer,
        color_config.scene_color,
        color_config.objects_colors,
        color_config.robot_color,
        color_config.bg_color,
        config['GS_dir'],
        config['records_name'],
        color_config,
        mask_clean_configs,
        config['camera_name_mapping'],
        logger
    )
    
    # render_color_augmentation(
    #     renderer,
    #     config['background_folder'],
    #     config['color_index_list'],
    #     config['bg_image_index_list'],
    #     config['bg_color_range_aug'],
    #     config['color_ranges'],
    #     config['scene_noise_range'],
    #     logger
    # )
    
    # Statistics
    stats = renderer.get_statistics()
    logger.info(f"Scene: {stats['scene_name']}, Objects: {stats['num_objects']}, Cameras: {stats['num_cameras']}")
    logger.info(f"GPU memory: {stats['gpu_memory']['allocated_gb']:.2f} GB")
    logger.info("Complete!")


if __name__ == "__main__":
    main()
