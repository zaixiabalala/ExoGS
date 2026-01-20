import os
import logging
import numpy as np
import torch
import cv2
import glob
import random
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List, Union, Any

from .renderer import GaussianSceneRenderer
from .model import GaussianModel
from .config import RenderConfig
from ..camera.constants import EYEINHAND_IDX

logger = logging.getLogger(__name__)


class GaussianSceneMulticolorRenderer(GaussianSceneRenderer):
    """
    Extended renderer with color augmentation and background replacement capabilities.
    """
    def __init__(self, Origin_dir: Optional[str] = None, GS_dir: Optional[str] = None, records_name: Optional[str] = None,
                 scene_name: Optional[str] = None, objects: Optional[List[str]] = None, enable_cache: bool = True, batch_size: int = 10,
                 robot_type: str = "flexiv"):
        super().__init__(Origin_dir=Origin_dir, GS_dir=GS_dir, records_name=records_name, scene_name=scene_name,
                        objects=objects, enable_cache=enable_cache, batch_size=batch_size, robot_type=robot_type)
        self.scene_color_scale_gpu: Optional[torch.Tensor] = None
        self.objects_color_scales_gpu: List[Optional[torch.Tensor]] = [None] * self.num_objects
        self.robot_color_scale_gpu: Optional[torch.Tensor] = None
        self.mask_mode, self.scene_solid_color = False, None
        self.objects_solid_colors: List[Optional[Tuple[float, float, float]]] = [None] * self.num_objects
        self.robot_solid_color: Optional[Tuple[float, float, float]] = None
        self.background_images: List[np.ndarray] = []
        self.background_image_paths: List[str] = []
        self.background_folder: Optional[str] = None
        self.bg_color_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
        self.consistent_bg_color_per_sequence, self.current_bg_color_scale = False, None
        self.selected_bg_image_index: Optional[int] = None
        logger.info("GaussianSceneMulticolorRenderer initialized (GPU optimized)")

    def set_background_images(self, background_folder: str, bg_color_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                            consistent_color_per_sequence: bool = False, bg_image_index: Optional[int] = None, bg_image_filename: Optional[str] = None):
        self.background_folder, self.bg_color_range = background_folder, bg_color_range
        self.consistent_bg_color_per_sequence = consistent_color_per_sequence
        if not os.path.exists(background_folder):
            raise FileNotFoundError(f"Background image folder not found: {background_folder}")
        
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(background_folder, ext)))
            image_files.extend(glob.glob(os.path.join(background_folder, ext.upper())))
        
        if len(image_files) == 0:
            raise ValueError(f"Background image folder is empty: {background_folder}")
        
        image_files = sorted(image_files)
        self.background_images = []
        self.background_image_paths = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                self.background_images.append(img)
                self.background_image_paths.append(img_path)
        
        if len(self.background_images) == 0:
            raise ValueError(f"Cannot load any background images: {background_folder}")
        self.selected_bg_image_index = None
        if bg_image_filename is not None:
            for idx, img_path in enumerate(self.background_image_paths):
                if os.path.basename(img_path) == bg_image_filename:
                    self.selected_bg_image_index = idx
                    logger.info(f"Using specified background image: {bg_image_filename} (index: {idx})")
                    break
            if self.selected_bg_image_index is None:
                raise ValueError(f"Specified background image file not found: {bg_image_filename}")
        elif bg_image_index is not None:
            if 0 <= bg_image_index < len(self.background_images):
                self.selected_bg_image_index = bg_image_index
                logger.info(f"Using specified background image index: {bg_image_index} ({os.path.basename(self.background_image_paths[bg_image_index])})")
            else:
                raise ValueError(f"Background image index out of range: {bg_image_index} (valid range: 0-{len(self.background_images)-1})")
        
        logger.info(f"Loaded {len(self.background_images)} background images")
        if self.selected_bg_image_index is not None:
            logger.info(f"Current background image: {os.path.basename(self.background_image_paths[self.selected_bg_image_index])}")
        else:
            logger.info("Background image selection mode: random")
        
        if bg_color_range is not None:
            logger.info(f"Background color variation range: {bg_color_range}")
            if consistent_color_per_sequence:
                logger.info("Entire sequence will use consistent background color variation")

    def _get_background_image_tensor(self, height: int, width: int, random_select: bool = True, seed: Optional[int] = None) -> torch.Tensor:
        """Get background image tensor with optional color variation."""
        if len(self.background_images) == 0:
            raise ValueError("Background images not set, please call set_background_images() first")
        
        # Use seed for both image selection and color variation
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if self.selected_bg_image_index is not None:
            bg_img = self.background_images[self.selected_bg_image_index]
        else:
            bg_img = random.choice(self.background_images) if random_select and len(self.background_images) > 1 else self.background_images[0]
        bg_resized = cv2.resize(bg_img, (width, height), interpolation=cv2.INTER_LINEAR)
        if self.bg_color_range is not None:
            min_rgb, max_rgb = self.bg_color_range
            if self.consistent_bg_color_per_sequence and self.current_bg_color_scale is not None:
                scale_r, scale_g, scale_b = self.current_bg_color_scale
            else:
                # Use seed for color variation to ensure consistency with color augmentation
                if seed is not None:
                    np.random.seed(seed + 1000)  # Offset to avoid collision with image selection
                scale_r = np.random.uniform(min_rgb[0], max_rgb[0])
                scale_g = np.random.uniform(min_rgb[1], max_rgb[1])
                scale_b = np.random.uniform(min_rgb[2], max_rgb[2])
                if self.consistent_bg_color_per_sequence:
                    self.current_bg_color_scale = (scale_r, scale_g, scale_b)
            bg_resized = bg_resized.astype(np.float32)
            bg_resized[:, :, 0] *= scale_b
            bg_resized[:, :, 1] *= scale_g
            bg_resized[:, :, 2] *= scale_r
            bg_resized = np.clip(bg_resized, 0, 255).astype(np.uint8)
        bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
        bg_tensor = torch.from_numpy(bg_rgb).float() / 255.0
        bg_tensor = bg_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return bg_tensor

    def reset_background_color_scale(self):
        """Reset background color scale for new sequence."""
        self.current_bg_color_scale = None

    def _tuple_to_gpu_tensor(self, rgb_tuple: Tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor([[[rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]]]], dtype=torch.float32, device=self.device)

    def set_color_scales(self, scene: Optional[Tuple[float, float, float]] = None, objects: Optional[List[Tuple[float, float, float]]] = None,
                        robot: Optional[Tuple[float, float, float]] = None, bg: Optional[Tuple[float, float, float]] = None):
        if scene is not None:
            assert len(scene) == 3, "[Error] scene color scale must be 3-tuple (r, g, b)"
            self.scene_color_scale_gpu = self._tuple_to_gpu_tensor(scene)
            logger.info(f"Scene color scale set: {scene} (GPU)")
        if objects is not None:
            assert len(objects) == self.num_objects, f"[Error] objects color list length({len(objects)}) != object count({self.num_objects})"
            self.objects_color_scales_gpu = []
            for idx, obj_color in enumerate(objects):
                if obj_color is not None:
                    assert len(obj_color) == 3, f"[Error] object_{idx} color scale must be 3-tuple"
                    self.objects_color_scales_gpu.append(self._tuple_to_gpu_tensor(obj_color))
                    logger.info(f"object_{idx}({self.objects[idx]}) color scale set: {obj_color} (GPU)")
                else:
                    self.objects_color_scales_gpu.append(None)
        if robot is not None:
            assert len(robot) == 3, "[Error] robot color scale must be 3-tuple (r, g, b)"
            self.robot_color_scale_gpu = self._tuple_to_gpu_tensor(robot)
            logger.info(f"Robot color scale set: {robot} (GPU)")
        if bg is not None:
            assert len(bg) == 3, "[Error] bg color must be 3-tuple (r, g, b)"
            assert all(0.0 <= c <= 1.0 for c in bg), "[Error] color values must be in [0.0, 1.0] range"
            self.bg_color = torch.tensor(bg, dtype=torch.float32, device=self.device)
            logger.info(f"Background color set: {bg} (GPU)")

    def randomize_colors(self, scene_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                        objects_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                        robot_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                        bg_range: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None, seed: Optional[int] = None):
        if seed is not None: np.random.seed(seed)
        def sample_color_gpu(range_tuple):
            min_rgb, max_rgb = range_tuple
            rgb_values = [float(np.random.uniform(min_rgb[i], max_rgb[i])) for i in range(3)]
            return torch.tensor([[[rgb_values[0], rgb_values[1], rgb_values[2]]]], dtype=torch.float32, device=self.device)
        if scene_range is not None:
            self.scene_color_scale_gpu = sample_color_gpu(scene_range)
        if objects_range is not None:
            self.objects_color_scales_gpu = []
            for obj_idx in range(self.num_objects):
                self.objects_color_scales_gpu.append(sample_color_gpu(objects_range))
        if robot_range is not None:
            self.robot_color_scale_gpu = sample_color_gpu(robot_range)
        if bg_range is not None:
            min_rgb, max_rgb = bg_range
            bg_values = [float(np.random.uniform(min_rgb[i], max_rgb[i])) for i in range(3)]
            self.bg_color = torch.tensor(bg_values, dtype=torch.float32, device=self.device)

    def reset_colors(self):
        self.scene_color_scale_gpu = None
        self.objects_color_scales_gpu = [None] * self.num_objects
        self.robot_color_scale_gpu = None
        self.bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.mask_mode, self.scene_solid_color = False, None
        self.objects_solid_colors = [None] * self.num_objects
        self.robot_solid_color = None
        logger.info("All color settings reset to default")

    def set_mask_mode(self, scene_color: Optional[Tuple[float, float, float]] = None, 
                    objects_colors: Optional[List[Tuple[float, float, float]]] = None,
                    robot_color: Optional[Tuple[float, float, float]] = None, 
                    bg_color: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)):
        self.mask_mode, self.scene_solid_color, self.robot_solid_color = True, scene_color, robot_color
        if objects_colors is not None:
            assert len(objects_colors) == self.num_objects, f"[Error] objects_colors length({len(objects_colors)}) != object count({self.num_objects})"
            self.objects_solid_colors = objects_colors
        else:
            self.objects_solid_colors = [(0.0, 0.0, 0.0)] * self.num_objects
        if bg_color:
            self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        logger.info("Mask mode enabled")
        if scene_color: logger.info(f"  - Scene: RGB{scene_color}")
        for obj_idx, obj_color in enumerate(self.objects_solid_colors):
            if obj_color: logger.info(f"  - object_{obj_idx}({self.objects[obj_idx]}): RGB{obj_color}")
        if robot_color: logger.info(f"  - Robot: RGB{robot_color}")
        logger.info(f"  - Background: RGB{bg_color}")

    def disable_mask_mode(self):
        self.mask_mode, self.scene_solid_color = False, None
        self.objects_solid_colors = [None] * self.num_objects
        self.robot_solid_color = None
        logger.info("Mask mode disabled, normal rendering restored")

    def get_current_color_config(self) -> Dict:
        def tensor_to_tuple(tensor): return None if tensor is None else tuple(tensor.squeeze().cpu().numpy().tolist())
        config = {'scene_color_scale': tensor_to_tuple(self.scene_color_scale_gpu),
                 'objects_color_scales': [tensor_to_tuple(t) for t in self.objects_color_scales_gpu],
                 'robot_color_scale': tensor_to_tuple(self.robot_color_scale_gpu),
                 'background_color': tuple(self.bg_color.cpu().numpy().tolist())}
        if self.mask_mode:
            config['mask_mode'] = True
            config['scene_solid_color'] = self.scene_solid_color
            config['objects_solid_colors'] = self.objects_solid_colors
            config['robot_solid_color'] = self.robot_solid_color
        return config

    def apply_color_config(self, config: Dict):
        if 'scene_color_scale' in config and config['scene_color_scale'] is not None:
            self.scene_color_scale_gpu = self._tuple_to_gpu_tensor(config['scene_color_scale'])
        if 'objects_color_scales' in config and config['objects_color_scales'] is not None:
            self.objects_color_scales_gpu = []
            for obj_color in config['objects_color_scales']:
                self.objects_color_scales_gpu.append(self._tuple_to_gpu_tensor(obj_color) if obj_color is not None else None)
        if 'robot_color_scale' in config and config['robot_color_scale'] is not None:
            self.robot_color_scale_gpu = self._tuple_to_gpu_tensor(config['robot_color_scale'])
        if 'background_color' in config:
            self.bg_color = torch.tensor(config['background_color'], dtype=torch.float32, device=self.device)
        logger.info("Color config applied (GPU optimized)")

    def _get_scene_color_scale(self) -> Optional[Union[Tuple[float, float, float], torch.Tensor, Tuple[str, Tuple]]]:  # type: ignore
        if self.mask_mode and self.scene_solid_color:
            return ('solid', self.scene_solid_color)
        return self.scene_color_scale_gpu

    def _get_object_color_scale(self, object_idx: int) -> Optional[Union[Tuple[float, float, float], torch.Tensor, Tuple[str, Tuple]]]:  # type: ignore
        if self.mask_mode:
            if object_idx < len(self.objects_solid_colors) and self.objects_solid_colors[object_idx]:
                return ('solid', self.objects_solid_colors[object_idx])
            else:
                return ('solid', (0.0, 0.0, 0.0))
        if object_idx < len(self.objects_color_scales_gpu):
            return self.objects_color_scales_gpu[object_idx]
        return None

    def _get_robot_color_scale(self) -> Optional[Union[Tuple[float, float, float], torch.Tensor, Tuple[str, Tuple]]]:  # type: ignore
        if self.mask_mode and self.robot_solid_color:
            return ('solid', self.robot_solid_color)
        return self.robot_color_scale_gpu

    def render_single_frame(self, camera, objects_poses: List[torch.Tensor], robot_links_pose: Dict[str, torch.Tensor],
                           scene_noise: Optional[List[float]], output_rgb: bool = True, output_depth: bool = True,
                           output_ply: bool = False, output_alpha: bool = False, return_on_gpu: bool = False, bg_seed: Optional[int] = None) -> Dict[str, Any]:
        result = super().render_single_frame(camera, objects_poses, robot_links_pose, scene_noise, output_rgb, output_depth, output_ply, output_alpha, return_on_gpu)
        # In mask mode, use pure black background (bg_color) instead of background images
        if output_rgb and len(self.background_images) > 0 and not self.mask_mode:
            rgb_result, height, width = result['rgb'], result['rgb'].shape[1], result['rgb'].shape[2]
            if 'alpha' in result: alpha = result['alpha']
            else:
                temp_result = super().render_single_frame(camera, objects_poses, robot_links_pose, scene_noise, output_rgb=False, output_depth=False, output_ply=False, output_alpha=True, return_on_gpu=True)
                alpha = temp_result['alpha']
            if alpha.dim() == 2: alpha = alpha.unsqueeze(0)
            elif alpha.dim() == 3 and alpha.shape[0] > 1: alpha = alpha[0:1]
            bg_tensor = self._get_background_image_tensor(height, width, random_select=True, seed=bg_seed)
            rgb_result = rgb_result.unsqueeze(0)
            alpha_expanded = alpha.expand_as(rgb_result)
            rgb_result = rgb_result * alpha_expanded + bg_tensor * (1 - alpha_expanded)
            result['rgb'] = rgb_result.squeeze(0)
        return result

    def _transform_model_on_gpu(self, model: GaussianModel, transform, rgb_scale: Optional[Union[Tuple[float, float, float], torch.Tensor, Tuple[str, Tuple]]] = None) -> GaussianModel:
        with torch.no_grad():
            if rgb_scale is not None:
                from ..utils.sh_utils import C0
                if isinstance(rgb_scale, tuple) and len(rgb_scale) == 2 and rgb_scale[0] == 'solid':
                    solid_color = rgb_scale[1]
                    rgb_solid = torch.tensor([[list(solid_color)]], dtype=torch.float32, device=self.device).expand(model._features_dc.shape[0], 1, 3)
                    model._features_dc = ((rgb_solid - 0.5) / C0).contiguous()
                else:
                    sh_dc, rgb = model._features_dc.clone(), model._features_dc.clone() * C0 + 0.5
                    rgb_scale_tensor = rgb_scale if isinstance(rgb_scale, torch.Tensor) else torch.tensor(rgb_scale, dtype=torch.float32, device=self.device).view(1, 1, 3)
                    rgb_adjusted = torch.clamp(rgb * rgb_scale_tensor, 0.0, 1.0)
                    model._features_dc = ((rgb_adjusted - 0.5) / C0).contiguous()
            return super()._transform_model_on_gpu(model, transform, None)

    def _transform_model_object_on_gpu(self, model: GaussianModel, obj_pose, rgb_scale: Optional[Union[Tuple[float, float, float], torch.Tensor, Tuple[str, Tuple]]] = None) -> GaussianModel:
        with torch.no_grad():
            if rgb_scale is not None:
                from ..utils.sh_utils import C0
                if isinstance(rgb_scale, tuple) and len(rgb_scale) == 2 and rgb_scale[0] == 'solid':
                    solid_color = rgb_scale[1]
                    rgb_solid = torch.tensor([[list(solid_color)]], dtype=torch.float32, device=self.device).expand(model._features_dc.shape[0], 1, 3)
                    model._features_dc = ((rgb_solid - 0.5) / C0).contiguous()
                else:
                    sh_dc, rgb = model._features_dc.clone(), model._features_dc.clone() * C0 + 0.5
                    rgb_scale_tensor = rgb_scale if isinstance(rgb_scale, torch.Tensor) else torch.tensor(rgb_scale, dtype=torch.float32, device=self.device).view(1, 1, 3)
                    rgb_adjusted = torch.clamp(rgb * rgb_scale_tensor, 0.0, 1.0)
                    model._features_dc = ((rgb_adjusted - 0.5) / C0).contiguous()
            return super()._transform_model_object_on_gpu(model, obj_pose, None)

    def render_batch_fixed_cameras(self, rgb: bool = True, depth: bool = False, ply: bool = False, alpha: bool = False,
                                  scene_noise_range: Optional[List[List[float]]] = None, record_indices: Optional[List[int]] = None,
                                  use_async_save: bool = True, rgb_folder_names: Optional[List[str]] = None,
                                  color_ranges: Optional[Dict] = None, base_seed: Optional[int] = None):
        assert self.cameras is not None, "[Error] Camera poses not initialized"
        if rgb_folder_names is None:
            super().render_batch_fixed_cameras(rgb=rgb, depth=depth, ply=ply, alpha=alpha, scene_noise_range=scene_noise_range, record_indices=record_indices, use_async_save=use_async_save)
            return
        num_variants = len(rgb_folder_names)
        if not self.mask_mode and num_variants > 1 and color_ranges is None:
            logger.warning("Multiple RGB variants but color_ranges not provided, all variants will use same color")
        records_to_process = list(enumerate(self.record_names)) if record_indices is None else [(idx, self.record_names[idx]) for idx in record_indices]
        for record_idx, record_name in records_to_process:
            logger.info(f"{'='*70}")
            logger.info(f"[Record {record_idx+1}/{len(self.record_names)}] {record_name}")
            logger.info(f"  Generating {num_variants} RGB variants: {rgb_folder_names}")
            if self.mask_mode: logger.info("  Mode: Mask rendering (solid color)")
            logger.info(f"{'='*70}")
            objects_poses_frames, robot_links_poses = self.load_objects_poses(record_name), self.load_robot_links_poses(record_name)
            scene_noise = [np.random.uniform(scene_noise_range[i][0], scene_noise_range[i][1]) for i in range(3)] if scene_noise_range is not None else None
            for variant_idx, rgb_folder_name in enumerate(rgb_folder_names):
                if not self.mask_mode and color_ranges is not None:
                    seed = base_seed + record_idx * 1000 + variant_idx if base_seed is not None else None
                    self.randomize_colors(**color_ranges, seed=seed)
                logger.info(f"[Record {record_idx+1}][Variant {variant_idx + 1}/{num_variants}] -> {rgb_folder_name}")
                if self.mask_mode:
                    config = self.get_current_color_config()
                    logger.info(f"  Mask config: scene{config.get('scene_solid_color')}, objects{config.get('objects_solid_colors')}")
                for cam_idx in range(len(self.cameras)):
                    if cam_idx != 0 and not rgb and not depth and ply: continue
                    cam_folder_name = self.get_camera_folder_name(cam_idx)
                    rgb_output_dir = os.path.join(self.GS_records_dir, record_name, cam_folder_name, rgb_folder_name)
                    if os.path.exists(rgb_output_dir) and len(os.listdir(rgb_output_dir)) > 0:
                        logger.info(f"[Skip] {cam_folder_name}/{rgb_folder_name} already exists")
                        continue
                    variant_seed = base_seed + record_idx * 1000 + variant_idx if base_seed is not None else None
                    self._render_fixed_single_camera_custom(record_name=record_name, cam_idx=cam_idx, rgb_folder_name=rgb_folder_name,
                                                           scene_noise=scene_noise, objects_poses_frames=objects_poses_frames,
                                                           robot_links_poses=robot_links_poses, rgb=rgb, depth=(depth and variant_idx == 0),
                                                           ply=(ply and variant_idx == 0 and cam_idx == 0), alpha=alpha, use_async_save=use_async_save,
                                                           bg_seed=variant_seed)
            logger.info(f"Record {record_idx+1} complete")

    def _render_fixed_single_camera_custom(self, record_name: str, cam_idx: int, rgb_folder_name: str, objects_poses_frames: List[List[torch.Tensor]],
                                          robot_links_poses: List[Dict[str, torch.Tensor]], scene_noise: Optional[List[float]], rgb: bool = True,
                                          depth: bool = True, ply: bool = False, alpha: bool = False, use_async_save: bool = True, bg_seed: Optional[int] = None):
        record_output_dir, cam_folder_name = os.path.join(self.GS_records_dir, record_name), self.get_camera_folder_name(cam_idx)
        cam_dir = os.path.join(record_output_dir, cam_folder_name)
        if rgb: os.makedirs(os.path.join(cam_dir, rgb_folder_name), exist_ok=True)
        if alpha: os.makedirs(os.path.join(cam_dir, 'alphas'), exist_ok=True)
        if depth: os.makedirs(os.path.join(cam_dir, 'depths'), exist_ok=True)
        if ply: os.makedirs(os.path.join(record_output_dir, 'plys'), exist_ok=True)
        assert self.cameras is not None, "[Error] Camera poses not initialized"
        camera, num_frames = self.cameras[cam_idx], len(robot_links_poses)
        if self.consistent_bg_color_per_sequence: self.reset_background_color_scale()
        with self._async_saver(enable=use_async_save, max_workers=2) as saver:
            pbar = tqdm(total=num_frames, desc=f"{cam_folder_name}/{rgb_folder_name}", unit="f", leave=False)
            for batch_start in range(0, num_frames, self.batch_size):
                batch_end, batch_results = min(batch_start + self.batch_size, num_frames), []
                for i in range(batch_start, batch_end):
                    frame_objects_poses = [obj_poses[i] for obj_poses in objects_poses_frames]
                    # Use provided bg_seed if available, otherwise use fixed seed for consistency
                    frame_bg_seed = bg_seed if bg_seed is not None else (42 if self.consistent_bg_color_per_sequence else None)
                    result = self.render_single_frame(camera=camera, objects_poses=frame_objects_poses, robot_links_pose=robot_links_poses[i],
                                                     scene_noise=scene_noise, output_rgb=rgb, output_depth=depth, output_ply=ply,
                                                     output_alpha=alpha, return_on_gpu=True, bg_seed=frame_bg_seed)
                    batch_results.append((i, result))
                torch.cuda.synchronize(self.device)
                save_tasks = []
                for i, result in batch_results:
                    if rgb and 'rgb' in result: save_tasks.append({'type': 'rgb', 'data': result['rgb'].cpu(), 'path': os.path.join(cam_dir, rgb_folder_name, f'{i:05d}.png')})
                    if alpha and 'alpha' in result: save_tasks.append({'type': 'alpha', 'data': result['alpha'].detach().cpu(), 'path': os.path.join(cam_dir, 'alphas', f'{i:05d}.png')})
                    if depth and 'depth' in result: save_tasks.append({'type': 'depth', 'data': result['depth'].detach().cpu(), 'path': os.path.join(cam_dir, 'depths', f'{i:05d}.png')})
                    if ply and 'point_cloud' in result: save_tasks.append({'type': 'ply', 'data': result['point_cloud'], 'path': os.path.join(record_output_dir, 'plys', f'{i:05d}.ply'), 'target_points': 10000})
                del batch_results
                if self.merged_gaussians is not None: del self.merged_gaussians; self.merged_gaussians = None
                if (batch_start // self.batch_size) % 5 == 0: torch.cuda.empty_cache()
                if save_tasks: saver.submit(self._save_batch, save_tasks); saver.wait_batch(max_pending=2)
                pbar.update(batch_end - batch_start)
            pbar.close()

    def render_batch_eyeinhand_cameras(self, cam_idx: int = EYEINHAND_IDX, rgb: bool = True, depth: bool = False, alpha: bool = False,
                                      scene_noise_range: Optional[List[List[float]]] = None, record_indices: Optional[List[int]] = None,
                                      use_async_save: bool = True, rgb_folder_names: Optional[List[str]] = None,
                                      color_ranges: Optional[Dict] = None, base_seed: Optional[int] = None):
        if rgb_folder_names is None:
            super().render_batch_eyeinhand_cameras(cam_idx=cam_idx, rgb=rgb, depth=depth, alpha=alpha, scene_noise_range=scene_noise_range,
                                                  record_indices=record_indices, use_async_save=use_async_save)
            return
        num_variants = len(rgb_folder_names)
        if not self.mask_mode and num_variants > 1 and color_ranges is None:
            logger.warning("Multiple RGB variants but color_ranges not provided, all variants will use same color")
        records_to_process = list(enumerate(self.record_names)) if record_indices is None else [(idx, self.record_names[idx]) for idx in record_indices]
        for record_idx, record_name in records_to_process:
            print(f"\n{'='*70}")
            print(f"[Record {record_idx+1}] Eye-in-Hand: {record_name}")
            print(f"  Generating {num_variants} RGB variants: {rgb_folder_names}")
            if self.mask_mode: print(f"  Mode: Mask rendering (solid color)")
            print(f"{'='*70}")
            objects_poses_frames, robot_links_poses = self.load_objects_poses(record_name), self.load_robot_links_poses(record_name)
            eyeinhand_cameras = self.load_eyeinhand_cameras(record_name)
            scene_noise = [np.random.uniform(scene_noise_range[i][0], scene_noise_range[i][1]) for i in range(3)] if scene_noise_range is not None else None
            for variant_idx, rgb_folder_name in enumerate(rgb_folder_names):
                if not self.mask_mode and color_ranges is not None:
                    seed = base_seed + record_idx * 1000 + variant_idx if base_seed is not None else None
                    self.randomize_colors(**color_ranges, seed=seed)
                logger.info(f"[Record {record_idx+1}][Variant {variant_idx + 1}/{num_variants}] -> {rgb_folder_name}")
                self._render_eyeinhand_camera_custom(record_name=record_name, cam_idx=cam_idx, rgb_folder_name=rgb_folder_name,
                                                    objects_poses_frames=objects_poses_frames, robot_links_poses=robot_links_poses,
                                                    eyeinhand_cameras=eyeinhand_cameras, scene_noise=scene_noise, rgb=rgb,
                                                    depth=(depth and variant_idx == 0), alpha=alpha, use_async_save=use_async_save)
            print(f"[Info] Record {record_idx+1} Eye-in-Hand complete")

    def _render_eyeinhand_camera_custom(self, record_name: str, cam_idx: int, rgb_folder_name: str, objects_poses_frames: List[List[torch.Tensor]],
                                       robot_links_poses: List[Dict[str, torch.Tensor]], eyeinhand_cameras: List, scene_noise: Optional[List[float]],
                                       rgb: bool = True, depth: bool = True, alpha: bool = False, use_async_save: bool = True):
        record_output_dir, cam_folder_name = os.path.join(self.GS_records_dir, record_name), self.get_camera_folder_name(cam_idx)
        cam_dir = os.path.join(record_output_dir, cam_folder_name)
        if rgb: os.makedirs(os.path.join(cam_dir, rgb_folder_name), exist_ok=True)
        if alpha: os.makedirs(os.path.join(cam_dir, 'alphas'), exist_ok=True)
        if depth: os.makedirs(os.path.join(cam_dir, 'depths'), exist_ok=True)
        num_frames = len(robot_links_poses)
        with self._async_saver(enable=use_async_save, max_workers=2) as saver:
            pbar = tqdm(total=num_frames, desc=f"{cam_folder_name}/{rgb_folder_name}", unit="f", leave=False)
            for batch_start in range(0, num_frames, self.batch_size):
                batch_end, batch_results = min(batch_start + self.batch_size, num_frames), []
                for i in range(batch_start, batch_end):
                    frame_objects_poses = [obj_poses[i] for obj_poses in objects_poses_frames]
                    result = self.render_single_frame(camera=eyeinhand_cameras[i], objects_poses=frame_objects_poses, robot_links_pose=robot_links_poses[i],
                                                     scene_noise=scene_noise, output_rgb=rgb, output_depth=depth, output_ply=False,
                                                     output_alpha=alpha, return_on_gpu=True)
                    batch_results.append((i, result))
                torch.cuda.synchronize(self.device)
                save_tasks = []
                for i, result in batch_results:
                    if rgb and 'rgb' in result: save_tasks.append({'type': 'rgb', 'data': result['rgb'].cpu(), 'path': os.path.join(cam_dir, rgb_folder_name, f'{i:05d}.png')})
                    if alpha and 'alpha' in result: save_tasks.append({'type': 'alpha', 'data': result['alpha'].detach().cpu(), 'path': os.path.join(cam_dir, 'alphas', f'{i:05d}.png')})
                    if depth and 'depth' in result: save_tasks.append({'type': 'depth', 'data': result['depth'].detach().cpu(), 'path': os.path.join(cam_dir, 'depths', f'{i:05d}.png')})
                if save_tasks: saver.submit(self._save_batch, save_tasks); saver.wait_batch(max_pending=2)
                pbar.update(batch_end - batch_start)
            pbar.close()

    def render(self, config: RenderConfig, camera_type: str = "fixed"):
        """
        Unified render interface with configuration
        
        Args:
            config: RenderConfig object containing all render settings
            camera_type: "fixed" or "eyeinhand"
        """
        # Apply background settings
        if config.background_folder:
            self.set_background_images(
                background_folder=config.background_folder,
                bg_color_range=config.bg_color_range,
                consistent_color_per_sequence=config.consistent_bg_color,
                bg_image_index=config.bg_image_index,
                bg_image_filename=config.bg_image_filename
            )
        
        # Apply color settings
        if config.mask_mode:
            self.set_mask_mode(
                scene_color=config.scene_color,
                objects_colors=config.objects_colors,
                robot_color=config.robot_color,
                bg_color=config.bg_color
            )
        elif config.scene_color or config.objects_colors or config.robot_color or config.bg_color:
            self.set_color_scales(
                scene=config.scene_color,
                objects=config.objects_colors,
                robot=config.robot_color,
                bg=config.bg_color
            )
        
        # Render based on camera type
        if camera_type == "fixed":
            self.render_batch_fixed_cameras(
                rgb=config.output_rgb,
                depth=config.output_depth,
                ply=config.output_ply,
                alpha=config.output_alpha,
                record_indices=config.record_indices,
                use_async_save=config.use_async_save,
                scene_noise_range=config.scene_noise_range,
                rgb_folder_names=config.rgb_folder_names,
                color_ranges=config.color_ranges,
                base_seed=config.base_seed
            )
        elif camera_type == "eyeinhand":
            self.render_batch_eyeinhand_cameras(
                rgb=config.output_rgb,
                depth=config.output_depth,
                alpha=config.output_alpha,
                record_indices=config.record_indices,
                use_async_save=config.use_async_save,
                scene_noise_range=config.scene_noise_range,
                rgb_folder_names=config.rgb_folder_names,
                color_ranges=config.color_ranges,
                base_seed=config.base_seed
            )
        else:
            raise ValueError(f"Unknown camera_type: {camera_type}, must be 'fixed' or 'eyeinhand'")
        
        # Disable mask mode if it was enabled
        if config.mask_mode:
            self.disable_mask_mode()
