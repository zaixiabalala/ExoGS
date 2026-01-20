"""
Color Configuration Utility
Provides unified color configuration with automatic format conversion.
All colors are defined in 0-255 RGB format and automatically converted to required formats.
"""
from typing import Tuple, List, Optional


class ColorConfig:
    """
    Unified color configuration class that handles automatic format conversion.
    
    Usage:
        # Define colors in 0-255 RGB format
        color_config = ColorConfig(
            scene_color_rgb_255=(0, 0, 0),
            bg_color_rgb_255=(0, 0, 0),
            robot_color_rgb_255=(0, 0, 255),
            objects_colors_rgb_255=[(255, 0, 0), (0, 255, 0)]
        )
        
        # Access converted colors
        scene_color = color_config.scene_color  # 0-1 RGB for mask_mode
        robot_color_bgr = color_config.robot_color_bgr  # 0-255 BGR for mask_clean
        target_colors = color_config.target_colors_bgr  # Auto-generated for mask_clean
    """
    
    def __init__(self, 
                 scene_color_rgb_255: Tuple[int, int, int] = (0, 0, 0),
                 bg_color_rgb_255: Tuple[int, int, int] = (0, 0, 0),
                 robot_color_rgb_255: Tuple[int, int, int] = (0, 0, 255),
                 objects_colors_rgb_255: List[Tuple[int, int, int]] = None):
        """
        Initialize color configuration with 0-255 RGB colors.
        
        Args:
            scene_color_rgb_255: Scene color in RGB format (0-255)
            bg_color_rgb_255: Background color in RGB format (0-255)
            robot_color_rgb_255: Robot color in RGB format (0-255)
            objects_colors_rgb_255: List of object colors in RGB format (0-255)
        """
        self.scene_color_rgb_255 = scene_color_rgb_255
        self.bg_color_rgb_255 = bg_color_rgb_255
        self.robot_color_rgb_255 = robot_color_rgb_255
        self.objects_colors_rgb_255 = objects_colors_rgb_255 if objects_colors_rgb_255 else []
        
        # Auto-convert all colors
        self._convert_all()
    
    @staticmethod
    def _rgb_255_to_rgb_01(rgb_255: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB from 0-255 range to 0-1 range"""
        return tuple(c / 255.0 for c in rgb_255)
    
    @staticmethod
    def _rgb_255_to_bgr_255(rgb_255: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB to BGR (reverse channel order)"""
        return tuple(reversed(rgb_255))
    
    def _convert_all(self):
        """Convert all colors to required formats"""
        # Convert to 0-1 RGB for mask_mode
        self.scene_color = self._rgb_255_to_rgb_01(self.scene_color_rgb_255)
        self.bg_color = self._rgb_255_to_rgb_01(self.bg_color_rgb_255)
        self.robot_color = self._rgb_255_to_rgb_01(self.robot_color_rgb_255)
        self.objects_colors = [self._rgb_255_to_rgb_01(c) for c in self.objects_colors_rgb_255]
        
        # Convert to 0-255 BGR for mask_clean
        self.robot_color_bgr = self._rgb_255_to_bgr_255(self.robot_color_rgb_255)
        self.objects_colors_bgr = [self._rgb_255_to_bgr_255(c) for c in self.objects_colors_rgb_255]
        
        # Auto-generate target_colors for mask_clean (0-255 BGR)
        # Order: [bg, object_0, object_1, ..., robot]
        self.target_colors_bgr = [
            self._rgb_255_to_bgr_255(self.bg_color_rgb_255),
        ]
        # Add all object colors
        for obj_color_bgr in self.objects_colors_bgr:
            self.target_colors_bgr.append(obj_color_bgr)
        # Add robot color at the end
        self.target_colors_bgr.append(self.robot_color_bgr)
    
    def get_mask_clean_config(self, rgb_folder: str = 'masks', 
                             method: str = 'instance',
                             output_suffix: str = '_clean',
                             min_object_size: int = 50,
                             kernel_size: int = 3) -> dict:
        """
        Generate mask_clean configuration dictionary.
        
        Args:
            rgb_folder: Folder name containing mask images
            method: Cleaning method ('instance', 'nearest', 'cluster', etc.)
            output_suffix: Suffix for output folder
            min_object_size: Minimum object size for cleaning
            kernel_size: Kernel size for morphological operations
            
        Returns:
            Dictionary with mask_clean configuration
        """
        return {
            'rgb_folder': rgb_folder,
            'target_colors': self.target_colors_bgr,
            'method': method,
            'output_suffix': output_suffix,
            'min_object_size': min_object_size,
            'kernel_size': kernel_size
        }
