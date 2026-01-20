from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class RenderConfig:
    """
    Unified render configuration for Gaussian Splatting scene rendering.
    """
    # Output settings
    output_rgb: bool = True
    output_depth: bool = False
    output_alpha: bool = False
    output_ply: bool = False
    
    # Camera settings
    camera_indices: Optional[List[int]] = None
    record_indices: Optional[List[int]] = None
    
    # Scene settings
    scene_noise_range: Optional[List[List[float]]] = None
    
    # Color settings
    scene_color: Optional[Tuple[float, float, float]] = None
    objects_colors: Optional[List[Tuple[float, float, float]]] = None
    robot_color: Optional[Tuple[float, float, float]] = None
    bg_color: Optional[Tuple[float, float, float]] = None
    
    # Color augmentation
    color_ranges: Optional[Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = None
    base_seed: Optional[int] = None
    
    # Background settings
    background_folder: Optional[str] = None
    bg_color_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    bg_image_index: Optional[int] = None
    bg_image_filename: Optional[str] = None
    consistent_bg_color: bool = False
    
    # Mask mode
    mask_mode: bool = False
    
    # Output folder names
    rgb_folder_names: Optional[List[str]] = None
    
    # Performance
    use_async_save: bool = True
    batch_size: int = 10
    
    def copy(self) -> 'RenderConfig':
        """Create a deep copy of this configuration."""
        return RenderConfig(
            output_rgb=self.output_rgb,
            output_depth=self.output_depth,
            output_alpha=self.output_alpha,
            output_ply=self.output_ply,
            camera_indices=self.camera_indices.copy() if self.camera_indices else None,
            record_indices=self.record_indices.copy() if self.record_indices else None,
            scene_noise_range=[r.copy() for r in self.scene_noise_range] if self.scene_noise_range else None,
            scene_color=self.scene_color,
            objects_colors=self.objects_colors.copy() if self.objects_colors else None,
            robot_color=self.robot_color,
            bg_color=self.bg_color,
            color_ranges=self.color_ranges.copy() if self.color_ranges else None,
            base_seed=self.base_seed,
            background_folder=self.background_folder,
            bg_color_range=self.bg_color_range,
            bg_image_index=self.bg_image_index,
            bg_image_filename=self.bg_image_filename,
            consistent_bg_color=self.consistent_bg_color,
            mask_mode=self.mask_mode,
            rgb_folder_names=self.rgb_folder_names.copy() if self.rgb_folder_names else None,
            use_async_save=self.use_async_save,
            batch_size=self.batch_size
        )


class RenderConfigBuilder:
    """
    Builder pattern for creating render configurations fluently.
    """
    
    def __init__(self):
        self.config = RenderConfig()
    
    def outputs(self, rgb: bool = True, depth: bool = False, alpha: bool = False, ply: bool = False) -> 'RenderConfigBuilder':
        """Set output types to generate."""
        self.config.output_rgb = rgb
        self.config.output_depth = depth
        self.config.output_alpha = alpha
        self.config.output_ply = ply
        return self
    
    def cameras(self, indices: Optional[List[int]] = None) -> 'RenderConfigBuilder':
        """Set camera indices to render."""
        self.config.camera_indices = indices
        return self
    
    def records(self, indices: Optional[List[int]] = None) -> 'RenderConfigBuilder':
        """Set record indices to render."""
        self.config.record_indices = indices
        return self
    
    def scene_noise(self, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]) -> 'RenderConfigBuilder':
        """Set scene position noise range for data augmentation."""
        self.config.scene_noise_range = [list(x_range), list(y_range), list(z_range)]
        return self
    
    def colors(self, scene: Optional[Tuple[float, float, float]] = None, 
               objects: Optional[List[Tuple[float, float, float]]] = None,
               robot: Optional[Tuple[float, float, float]] = None,
               bg: Optional[Tuple[float, float, float]] = None) -> 'RenderConfigBuilder':
        """Set fixed colors for scene components."""
        self.config.scene_color = scene
        self.config.objects_colors = objects
        self.config.robot_color = robot
        self.config.bg_color = bg
        return self
    
    def color_augmentation(self, scene_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                          objects_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                          robot_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                          bg_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                          seed: Optional[int] = None) -> 'RenderConfigBuilder':
        """Set color augmentation ranges for randomized color variations."""
        self.config.color_ranges = {}
        if scene_range:
            self.config.color_ranges['scene_range'] = scene_range
        if objects_range:
            self.config.color_ranges['objects_range'] = objects_range
        if robot_range:
            self.config.color_ranges['robot_range'] = robot_range
        if bg_range:
            self.config.color_ranges['bg_range'] = bg_range
        self.config.base_seed = seed
        return self
    
    def background(self, folder: Optional[str] = None, 
                   color_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                   image_index: Optional[int] = None,
                   image_filename: Optional[str] = None,
                   consistent_color: bool = False) -> 'RenderConfigBuilder':
        """Set background replacement settings."""
        self.config.background_folder = folder
        self.config.bg_color_range = color_range
        self.config.bg_image_index = image_index
        self.config.bg_image_filename = image_filename
        self.config.consistent_bg_color = consistent_color
        return self
    
    def mask_mode(self, scene_color: Optional[Tuple[float, float, float]] = None,
                  objects_colors: Optional[List[Tuple[float, float, float]]] = None,
                  robot_color: Optional[Tuple[float, float, float]] = None,
                  bg_color: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)) -> 'RenderConfigBuilder':
        """Enable mask mode with solid colors for segmentation."""
        self.config.mask_mode = True
        self.config.scene_color = scene_color
        self.config.objects_colors = objects_colors
        self.config.robot_color = robot_color
        self.config.bg_color = bg_color
        return self
    
    def output_folders(self, rgb_folders: List[str]) -> 'RenderConfigBuilder':
        """Set RGB output folder names for organizing outputs."""
        self.config.rgb_folder_names = rgb_folders
        return self
    
    def performance(self, async_save: bool = True, batch_size: int = 10) -> 'RenderConfigBuilder':
        """Set performance optimization settings."""
        self.config.use_async_save = async_save
        self.config.batch_size = batch_size
        return self
    
    def build(self) -> RenderConfig:
        """Build and return the final configuration."""
        return self.config.copy()
