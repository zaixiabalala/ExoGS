import os
import logging
import cv2
import shutil
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from ..utils.graphics_utils import focal2fov
from ..utils.camera_utils import cameraList_from_camInfos
from ..utils.ply_utils import extract_and_save_point_cloud
from ..render.renderer import render
from ..camera.camera import Camera
from ..camera.reader import readCamerasFromTransforms
from ..camera.constants import FX, FY, WIDTH, HEIGHT, EYEINHAND_IDX
from .model import GaussianModel
from .kinematics import URDFKinematics, merge_3dgs_models, transform_gs_ply_full, transform_gs_ply_full_object

logger = logging.getLogger(__name__)


class GaussianSceneBase:
    """
    Base class for Gaussian Splatting scene management with robot kinematics and object tracking.
    """
    
    def __init__(self, Origin_dir: Optional[str] = None, GS_dir: Optional[str] = None, records_name: Optional[str] = None,
                 scene_name: Optional[str] = None, objects: Optional[List[str]] = None, enable_cache: bool = True, batch_size: int = 10,
                 robot_type: str = "flexiv"):
        logger.info("Initializing GaussianSceneBase")
        assert Origin_dir is not None, "[Error] Origin_dir cannot be empty"
        assert GS_dir is not None, "[Error] GS_dir cannot be empty"
        assert records_name is not None, "[Error] records_name cannot be empty"
        assert scene_name is not None, "[Error] scene_name cannot be empty"
        assert objects is not None and len(objects) > 0, "[Error] objects list cannot be empty"
        assert robot_type in ["flexiv", "franka"], f"[Error] robot_type must be 'flexiv' or 'franka', got '{robot_type}'"
        
        self.device = torch.device("cuda")
        self.Origin_records_dir = os.path.join(Origin_dir, records_name)
        self.GS_records_dir = os.path.join(GS_dir, records_name)
        os.makedirs(self.GS_records_dir, exist_ok=True)
        self.records_name = records_name
        self.record_names = sorted([f for f in os.listdir(self.Origin_records_dir) if f.startswith('record_')])
        self.record_dirs = [os.path.join(self.Origin_records_dir, f) for f in self.record_names]
        self.scene_name = scene_name
        self.objects = objects
        self.num_objects = len(objects)
        self.robot_type = robot_type
        
        # Setup asset paths based on robot type
        gaussian_sim_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assets_dir = os.path.join(gaussian_sim_dir, "assets")
        if robot_type == "flexiv":
            self.robot_urdf_path = os.path.join(self.assets_dir, "robot", "robot_urdf", "flexiv", "flexiv_Rizon4s_kinematics.urdf")
            self.robot_gs_ply_dir = os.path.join(self.assets_dir, "robot", "robot_ply", "flexiv")
            joint_states = {
                'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0, 'joint4': 0.0,
                'joint5': 0.0, 'joint6': 0.0, 'joint7': 0.0,
                'xense_finger_joint1': 0.0, 'xense_finger_joint2': 0.0
            }
        elif robot_type == "franka":
            self.robot_urdf_path = os.path.join(self.assets_dir, "robot", "robot_urdf", "panda", "panda_v2.urdf")
            self.robot_gs_ply_dir = os.path.join(self.assets_dir, "robot", "robot_ply", "panda")
            joint_states = {
                'panda_joint1': 0.0, 'panda_joint2': 0.0, 'panda_joint3': 0.0, 'panda_joint4': 0.0,
                'panda_joint5': 0.0, 'panda_joint6': 0.0, 'panda_joint7': 0.0,
                'panda_finger_joint1': 0.0, 'panda_finger_joint2': 0.0
            }
        logger.info(f"Robot type: {robot_type}, URDF: {self.robot_urdf_path}, PLY dir: {self.robot_gs_ply_dir}")
        
        self.cam_infos = None
        self.cameras = None
        self.eyeinhand_cameras = None
        self.bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.camera_name_mapping = None
        
        # Initialize forward kinematics

        self.fk = URDFKinematics(self.robot_urdf_path, joint_states, np.eye(4))
        self.link_trans_poses = self.fk.get_new_link_tran_poses_gpu(joint_states, device=self.device, keep_on_gpu=True)
        
        self.merged_gaussians = None
        self.enable_cache = enable_cache
        self.batch_size = batch_size
        self.model_cache: Dict[str, Any] = {}
        self.pose_cache: Dict[str, Any] = {}
        
        if self.enable_cache:
            self._cache_models()
        logger.info(f"GaussianSceneBase initialized (batch_size={self.batch_size})")

    def _cache_models(self):
        """Cache scene, object, and robot link models to GPU memory."""
        logger.info("Caching models to GPU...")
        if self.scene_name is not None:
            scene_ply_path = os.path.join(self.assets_dir, "scenes", self.scene_name, "point_cloud.ply")
            if os.path.exists(scene_ply_path):
                scene_model = GaussianModel(0)
                scene_model.load_ply(scene_ply_path)
                self.model_cache['scene'] = scene_model
                logger.info(f"Scene model cached: {self.scene_name}")
            else:
                raise FileNotFoundError(f"[Error] Scene model not found: {self.scene_name}")
        
        self.model_cache['objects'] = {}
        for idx, object_name in enumerate(self.objects):
            object_ply_path = os.path.join(self.assets_dir, "objects", f"{object_name}.ply")
            if os.path.exists(object_ply_path):
                object_model = GaussianModel(0)
                object_model.load_ply(object_ply_path)
                self.model_cache['objects'][idx] = object_model
                logger.info(f"Object model cached [object_{idx}]: {object_name}")
            else:
                raise FileNotFoundError(f"[Error] Object model not found [object_{idx}]: {object_name}")
        
        if os.path.exists(self.robot_gs_ply_dir):
            self.model_cache['robot_links'] = {}
            link_files = [f for f in os.listdir(self.robot_gs_ply_dir) if f.endswith('.ply')]
            for link_file in link_files:
                link_name = link_file[:-4]
                link_ply_path = os.path.join(self.robot_gs_ply_dir, link_file)
                link_model = GaussianModel(0)
                link_model.load_ply(link_ply_path)
                self.model_cache['robot_links'][link_name] = link_model
            logger.info(f"Robot {len(link_files)} link models cached")
        else:
            raise FileNotFoundError(f"[Error] Robot link models not found: {self.robot_gs_ply_dir}")
        logger.info("Model caching complete")

    def _get_cached_or_load_model(self, cache_key: str, ply_path: str) -> GaussianModel:
        if self.enable_cache and cache_key in self.model_cache:
            cached_model = self.model_cache[cache_key]
            model = GaussianModel(0)
            with torch.no_grad():
                model._xyz = cached_model._xyz.clone()
                model._features_dc = cached_model._features_dc.clone()
                model._features_rest = cached_model._features_rest.clone()
                model._scaling = cached_model._scaling.clone()
                model._rotation = cached_model._rotation.clone()
                model._opacity = cached_model._opacity.clone()
                model.max_radii2D = cached_model.max_radii2D.clone()
                model.active_sh_degree = cached_model.active_sh_degree
                model.max_sh_degree = cached_model.max_sh_degree
            return model
        else:
            logger.info(f"cache_key not found, loading model directly: {ply_path}")
            model = GaussianModel(0)
            model.load_ply(ply_path)
            return model

    def fixed_cameras_init(self, json_name: str = "transforms.json", width: int = 640, height: int = 480,
                          camera_name_mapping: Optional[Dict[int, str]] = None):
        cam_json_path = os.path.join(self.GS_records_dir, json_name)
        assert os.path.exists(cam_json_path), f"[Error] cam_json_path not found: {cam_json_path}"
        self.cam_infos = readCamerasFromTransforms(cam_json_path, width, height)
        self.cameras = cameraList_from_camInfos(self.cam_infos)
        self.camera_name_mapping = camera_name_mapping
        if camera_name_mapping:
            logger.info(f"Fixed cameras initialized, {len(self.cameras)} cameras, using custom naming")
            for idx, name in camera_name_mapping.items():
                if idx < len(self.cameras):
                    logger.info(f"  - Index {idx} -> {name}")
        else:
            logger.info(f"Fixed cameras initialized, {len(self.cameras)} cameras, using default naming cam_0, cam_1, ...")

    def init_scene(self, rgb_scale: Optional[Tuple[float, float, float]] = None, scene_noise: Optional[List[float]] = None):
        assert self.scene_name is not None, "[Error] scene_name cannot be empty"
        scene_gs_ply_path = os.path.join(self.assets_dir, "scenes", self.scene_name, "point_cloud.ply")
        if scene_noise is not None and len(scene_noise) == 3:
            scene_position = np.array(scene_noise)
        else:
            scene_position = np.zeros(3)
        scene_pose = np.eye(4)
        scene_pose[:3, 3] = scene_position
        if self.enable_cache and 'scene' in self.model_cache:
            scene_gaussians = self._get_cached_or_load_model('scene', scene_gs_ply_path)
            scene_gaussians = self._transform_model_on_gpu(scene_gaussians, scene_pose, rgb_scale)
        else:
            logger.info(f"cache_key not found, loading model directly: {scene_gs_ply_path}")
            scene_gaussians = transform_gs_ply_full(scene_gs_ply_path, scene_pose, rgb_scale)
        self.merged_gaussians = scene_gaussians

    def update_object(self, object_idx: int, object_pose, rgb_scale: Optional[Tuple[float, float, float]] = None):
        assert object_idx is not None, "[Error] object_idx cannot be empty"
        assert 0 <= object_idx < self.num_objects, f"[Error] object_idx {object_idx} out of range [0, {self.num_objects-1}]"
        assert object_pose is not None, "[Error] object_pose cannot be empty"
        assert len(object_pose.shape) == 2 and object_pose.shape[0] == 4 and object_pose.shape[1] == 4, "[Error] object_pose must be 4x4 matrix"
        object_name = self.objects[object_idx]
        object_gs_ply_path = os.path.join(self.assets_dir, "objects", f"{object_name}.ply")
        if self.enable_cache and 'objects' in self.model_cache and object_idx in self.model_cache['objects']:
            cached_model = self.model_cache['objects'][object_idx]
            object_gaussians = GaussianModel(0)
            with torch.no_grad():
                object_gaussians._xyz = cached_model._xyz.clone()
                object_gaussians._features_dc = cached_model._features_dc.clone()
                object_gaussians._features_rest = cached_model._features_rest.clone()
                object_gaussians._scaling = cached_model._scaling.clone()
                object_gaussians._rotation = cached_model._rotation.clone()
                object_gaussians._opacity = cached_model._opacity.clone()
                object_gaussians.max_radii2D = cached_model.max_radii2D.clone()
                object_gaussians.active_sh_degree = cached_model.active_sh_degree
                object_gaussians.max_sh_degree = cached_model.max_sh_degree
            object_gaussians = self._transform_model_object_on_gpu(object_gaussians, object_pose, rgb_scale)
        else:
            object_gaussians = transform_gs_ply_full_object(object_gs_ply_path, object_pose, rgb_scale)
        self.merged_gaussians = merge_3dgs_models(self.merged_gaussians, object_gaussians)

    def update_objects(self, objects_poses: List[torch.Tensor], objects_rgb_scales: Optional[List[Optional[Tuple[float, float, float]]]] = None):
        assert len(objects_poses) == self.num_objects, f"[Error] objects_poses length({len(objects_poses)}) != object count({self.num_objects})"
        if objects_rgb_scales is None:
            objects_rgb_scales = [None] * self.num_objects
        for obj_idx in range(self.num_objects):
            self.update_object(obj_idx, objects_poses[obj_idx], objects_rgb_scales[obj_idx])

    def update_robot(self, robot_links_poses, rgb_scale: Optional[Tuple[float, float, float]] = None):
        assert robot_links_poses is not None, "[Error] robot_links_poses cannot be empty"
        for link_name, T_link in robot_links_poses.items():
            input_ply = os.path.join(self.robot_gs_ply_dir, f"{link_name}.ply")
            if not os.path.exists(input_ply):
                continue
            if self.enable_cache and 'robot_links' in self.model_cache and link_name in self.model_cache['robot_links']:
                cached_model = self.model_cache['robot_links'][link_name]
                robot_gaussians = GaussianModel(0)
                with torch.no_grad():
                    robot_gaussians._xyz = cached_model._xyz.clone()
                    robot_gaussians._features_dc = cached_model._features_dc.clone()
                    robot_gaussians._features_rest = cached_model._features_rest.clone()
                    robot_gaussians._scaling = cached_model._scaling.clone()
                    robot_gaussians._rotation = cached_model._rotation.clone()
                    robot_gaussians._opacity = cached_model._opacity.clone()
                    robot_gaussians.max_radii2D = cached_model.max_radii2D.clone()
                    robot_gaussians.active_sh_degree = cached_model.active_sh_degree
                    robot_gaussians.max_sh_degree = cached_model.max_sh_degree
                robot_gaussians = self._transform_model_on_gpu(robot_gaussians, T_link, rgb_scale)
            else:
                robot_gaussians = transform_gs_ply_full(input_ply, T_link, rgb_scale)
            self.merged_gaussians = merge_3dgs_models(self.merged_gaussians, robot_gaussians)

    def _transform_model_on_gpu(self, model: GaussianModel, transform, rgb_scale: Optional[Tuple[float, float, float]] = None) -> GaussianModel:
        """Transform Gaussian model on GPU with optional RGB scaling."""
        with torch.no_grad():
            if rgb_scale is not None:
                from ..utils.sh_utils import C0
                sh_dc, rgb = model._features_dc.clone(), model._features_dc.clone() * C0 + 0.5
                rgb_scale_tensor = torch.tensor(rgb_scale, dtype=torch.float32, device=self.device).view(1, 1, 3)
                rgb_adjusted = torch.clamp(rgb * rgb_scale_tensor, 0.0, 1.0)
                model._features_dc = ((rgb_adjusted - 0.5) / C0).contiguous()
            if isinstance(transform, torch.Tensor):
                mdh_tensor = transform.to(self.device) if transform.device != self.device else transform
            else:
                mdh_tensor = torch.from_numpy(transform).float().to(self.device)
            R, t = mdh_tensor[:3, :3].contiguous(), mdh_tensor[:3, 3].contiguous()
            xyz_original = model._xyz.clone()
            transformed_xyz = (xyz_original @ R.T + t).contiguous()
            from .kinematics import build_rotation_tensor, rotmat_to_quaternion_tensor
            if model._rotation.size(0) > 0:
                rot_original, rot_matrices = model._rotation.clone(), build_rotation_tensor(model._rotation.clone())
                R_expanded = R.unsqueeze(0).expand(rot_matrices.size(0), -1, -1).contiguous()
                transformed_rot_mat = torch.bmm(R_expanded, rot_matrices).contiguous()
                transformed_quat = rotmat_to_quaternion_tensor(transformed_rot_mat).contiguous()
            else:
                transformed_quat = model._rotation
            model._xyz, model._rotation = transformed_xyz, transformed_quat
        return model

    def _transform_model_object_on_gpu(self, model: GaussianModel, obj_pose, rgb_scale: Optional[Tuple[float, float, float]] = None) -> GaussianModel:
        """Transform object Gaussian model on GPU with optional RGB scaling."""
        with torch.no_grad():
            if rgb_scale is not None:
                from ..utils.sh_utils import C0
                sh_dc, rgb = model._features_dc.clone(), model._features_dc.clone() * C0 + 0.5
                rgb_scale_tensor = torch.tensor(rgb_scale, dtype=torch.float32, device=self.device).view(1, 1, 3)
                rgb_adjusted = torch.clamp(rgb * rgb_scale_tensor, 0.0, 1.0)
                model._features_dc = ((rgb_adjusted - 0.5) / C0).contiguous()
            xyz_original = model._xyz.clone()
            # min_xyz, max_xyz = xyz_original.min(dim=0)[0], xyz_original.max(dim=0)[0]
            # center_xyz, centered_xyz = (min_xyz + max_xyz) / 2, xyz_original - (min_xyz + max_xyz) / 2
            if isinstance(obj_pose, torch.Tensor):
                obj2base_tensor = obj_pose.to(self.device) if obj_pose.device != self.device else obj_pose
            else:
                obj2base_tensor = torch.from_numpy(obj_pose).float().to(self.device)
            R_obj2base, t_obj2base = obj2base_tensor[:3, :3].contiguous(), obj2base_tensor[:3, 3].contiguous()
            # transformed_xyz = (centered_xyz @ R_obj2base.T + t_obj2base).contiguous()
            transformed_xyz = (xyz_original @ R_obj2base.T + t_obj2base).contiguous()
            from .kinematics import build_rotation_tensor, rotmat_to_quaternion_tensor
            rot_original, rot_matrices = model._rotation.clone(), build_rotation_tensor(model._rotation.clone())
            R_obj2base_expanded = R_obj2base.unsqueeze(0).expand(rot_matrices.size(0), -1, -1).contiguous()
            transformed_rot_mat_base = torch.bmm(R_obj2base_expanded, rot_matrices).contiguous()
            transformed_quat = rotmat_to_quaternion_tensor(transformed_rot_mat_base).contiguous()
            model._xyz, model._rotation = transformed_xyz, transformed_quat
        return model

    def load_object_pose(self, record_name: str, object_idx: int) -> List[torch.Tensor]:
        cache_key = f"{record_name}_object_{object_idx}_pose"
        if cache_key in self.pose_cache:
            return self.pose_cache[cache_key]
        object_poses = []
        object_pose_dir = os.path.join(self.Origin_records_dir, record_name, "poses", f"object_{object_idx + 1}")
        if not os.path.exists(object_pose_dir):
            raise FileNotFoundError(f"[Error] Object {object_idx} pose directory not found: {object_pose_dir}")
        files = sorted([f for f in os.listdir(object_pose_dir) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))
        for file in files:
            file_path = os.path.join(object_pose_dir, file)
            with open(file_path, "r") as f:
                matrix = [list(map(float, line.strip().split())) for line in f if line.strip()]
                pose_tensor = torch.from_numpy(np.array(matrix)).float().to(self.device)
                object_poses.append(pose_tensor)
        if self.enable_cache:
            self.pose_cache[cache_key] = object_poses
        return object_poses

    def load_objects_poses(self, record_name: str) -> List[List[torch.Tensor]]:
        all_objects_poses = []
        for obj_idx in range(self.num_objects):
            all_objects_poses.append(self.load_object_pose(record_name, obj_idx))
        return all_objects_poses

    def load_robot_links_poses(self, record_name: str) -> List[Dict[str, torch.Tensor]]:
        cache_key = f"{record_name}_robot_poses"
        if cache_key in self.pose_cache:
            return self.pose_cache[cache_key]
        robot_links_poses = []
        angles_dir = os.path.join(self.GS_records_dir, record_name, "angles")
        if not os.path.exists(angles_dir):
            angles_dir = os.path.join(self.Origin_records_dir, record_name, "organized", "angles")
        if not os.path.exists(angles_dir):
            angles_dir = os.path.join(self.Origin_records_dir, record_name, "angles")
        assert os.path.exists(angles_dir), f"[Error] angles_dir not found: {angles_dir}"
        angles_files = sorted([f for f in os.listdir(angles_dir) if f.endswith('.npy')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for file in tqdm(angles_files, desc="Computing FK", unit="frame"):
            angles = np.load(os.path.join(angles_dir, file))
            if self.robot_type == "flexiv":
                joint_states = {
                    'joint1': angles[0], 'joint2': angles[1], 'joint3': angles[2], 'joint4': angles[3],
                    'joint5': angles[4], 'joint6': angles[5], 'joint7': angles[6],
                    'xense_finger_joint1': min(max(float(angles[7] / 2.0), 0.0), 0.5),
                    'xense_finger_joint2': min(max(float(angles[7]) / 2.0, 0.0), 0.5)
                }
            elif self.robot_type == "franka":
                joint_states = {
                    'panda_joint1': angles[0], 'panda_joint2': angles[1], 'panda_joint3': angles[2], 'panda_joint4': angles[3],
                    'panda_joint5': angles[4], 'panda_joint6': angles[5], 'panda_joint7': angles[6],
                    'panda_finger_joint1': min(max(float(angles[7] / 2.0), 0.0), 0.04),
                    'panda_finger_joint2': min(max(float(angles[7] / 2.0), 0.0), 0.04)
                }
            else:
                raise ValueError(f"[Error] Invalid robot type: {self.robot_type}")
            robot_links_pose = self.fk.get_new_link_tran_poses_gpu(joint_states, device='cuda', keep_on_gpu=True)
            robot_links_poses.append(robot_links_pose)
        if self.enable_cache:
            self.pose_cache[cache_key] = robot_links_poses
        return robot_links_poses

    def load_eyeinhand_cameras(self, record_name: str) -> List[Camera]:
        cache_key = f"{record_name}_eyeinhand_cameras"
        if cache_key in self.pose_cache:
            return self.pose_cache[cache_key]
        eyeinhand_npy_files_dir = os.path.join(self.Origin_records_dir, record_name, "ees")
        eyeinhand_npy_files = sorted([f for f in os.listdir(eyeinhand_npy_files_dir) if f.endswith('.npy')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
        eyeinhand_cameras = []
        FovX, FovY = focal2fov(FX, WIDTH), focal2fov(FY, HEIGHT)
        for file in eyeinhand_npy_files:
            eyeinhand_pose = np.load(os.path.join(eyeinhand_npy_files_dir, file))
            c2w, w2c = eyeinhand_pose.copy(), np.linalg.inv(eyeinhand_pose.copy())
            R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]
            eyeinhand_cameras.append(Camera(EYEINHAND_IDX, R, T, FovX, FovY, WIDTH, HEIGHT))
        if self.enable_cache:
            self.pose_cache[cache_key] = eyeinhand_cameras
        self.eyeinhand_cameras = eyeinhand_cameras
        return eyeinhand_cameras

    def clear_cache(self):
        self.model_cache.clear()
        self.pose_cache.clear()
        if self.merged_gaussians is not None:
            del self.merged_gaussians
            self.merged_gaussians = None
        torch.cuda.empty_cache()
        logger.info("Cache cleared")

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
        return {'allocated_gb': allocated, 'reserved_gb': reserved, 'max_allocated_gb': max_allocated}

    @contextmanager
    def _async_saver(self, enable: bool = True, max_workers: int = 2):
        executor = None
        try:
            if enable:
                executor = ThreadPoolExecutor(max_workers=max_workers)
            class Saver:
                def __init__(self, executor):
                    self.executor = executor
                    self.futures = []
                def submit(self, func, *args, **kwargs):
                    if self.executor:
                        future = self.executor.submit(func, *args, **kwargs)
                        self.futures.append(future)
                        return future
                    else:
                        func(*args, **kwargs)
                        return None
                def wait_batch(self, max_pending: int = 2):
                    while len(self.futures) >= max_pending:
                        self.futures[0].result()
                        self.futures.pop(0)
            saver = Saver(executor)
            yield saver
            for future in saver.futures:
                if future:
                    future.result()
        finally:
            if executor:
                executor.shutdown(wait=True)

    def _save_batch(self, save_tasks: List[Dict]):
        for task in save_tasks:
            if task['type'] == 'rgb':
                torchvision.utils.save_image(task['data'], task['path'])
            elif task['type'] == 'depth':
                depth_np = task['data'].squeeze().numpy()
                cv2.imwrite(task['path'], (depth_np * 1000).astype(np.uint16))
            elif task['type'] == 'ply':
                extract_and_save_point_cloud(task['data'], task['path'], target_points=task.get('target_points', 10000))
            elif task['type'] == 'alpha':
                cv2.imwrite(task['path'], (task['data'].squeeze().cpu().numpy() * 255).astype(np.uint8))

    def _get_scene_color_scale(self) -> Optional[Tuple[float, float, float]]:
        return None
    def _get_object_color_scale(self, object_idx: int) -> Optional[Tuple[float, float, float]]:
        return None
    def _get_robot_color_scale(self) -> Optional[Tuple[float, float, float]]:
        return None

    def render_single_frame(self, camera, objects_poses: List[torch.Tensor], robot_links_pose: Dict[str, torch.Tensor],
                           scene_noise: Optional[List[float]], output_rgb: bool = True, output_depth: bool = True,
                           output_ply: bool = False, output_alpha: bool = False, return_on_gpu: bool = False) -> Dict[str, Any]:
        assert len(objects_poses) == self.num_objects, f"[Error] objects_poses length({len(objects_poses)}) != object count({self.num_objects})"
        if self.merged_gaussians is not None:
            del self.merged_gaussians
            self.merged_gaussians = None
        self.merged_gaussians = GaussianModel(0)
        self.init_scene(scene_noise=scene_noise, rgb_scale=self._get_scene_color_scale())
        for obj_idx in range(self.num_objects):
            self.update_object(obj_idx, objects_poses[obj_idx], rgb_scale=self._get_object_color_scale(obj_idx))
        self.update_robot(robot_links_pose, rgb_scale=self._get_robot_color_scale())
        render_result = render(camera, self.merged_gaussians, bg_color=self.bg_color)
        result = {}
        if output_rgb:
            result['rgb'] = render_result["render"]
        if output_depth:
            result['depth'] = render_result["depth"]
        if output_alpha:
            result['alpha'] = render_result["alpha"]
        if output_ply:
            result['point_cloud'] = self.merged_gaussians
        return result

    def collect_angles(self):
        logger.info("Collecting angle data...")
        for record_name in tqdm(self.record_names, desc="Collecting angles", unit="record"):
            origin_angles_dir = os.path.join(self.Origin_records_dir, record_name, "organized", "angles")
            if not os.path.exists(origin_angles_dir):
                origin_angles_dir = os.path.join(self.Origin_records_dir, record_name, "angles")
            if not os.path.exists(origin_angles_dir):
                logger.warning(f"Skipping {record_name}: angles directory not found")
                continue
            target_angles_dir = os.path.join(self.GS_records_dir, record_name, "angles")
            os.makedirs(target_angles_dir, exist_ok=True)
            for file in os.listdir(origin_angles_dir):
                if not file.endswith('.npy'):
                    continue
                frame_num = file.split('_')[-1].split('.')[0]
                shutil.copyfile(os.path.join(origin_angles_dir, file), os.path.join(target_angles_dir, f"{frame_num}.npy"))
        logger.info("Angle data collection complete")

    def collect_tcps(self):
        for record_name in tqdm(self.record_names, desc="Collecting tcps", unit="record"):
            origin_tcps_dir = os.path.join(self.Origin_records_dir, record_name, "tcps")
            shutil.copytree(origin_tcps_dir, os.path.join(self.GS_records_dir, record_name, "tcps"), dirs_exist_ok=True)

    def get_camera_folder_name(self, cam_idx: int) -> str:
        if self.camera_name_mapping and cam_idx in self.camera_name_mapping:
            return self.camera_name_mapping[cam_idx]
        else:
            return f'cam_{cam_idx}'

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'num_records': len(self.record_names),
            'num_cameras': len(self.cameras) if self.cameras else 0,
            'scene_name': self.scene_name,
            'objects': self.objects,
            'num_objects': self.num_objects,
            'batch_size': self.batch_size,
            'gpu_memory': self.get_gpu_memory_usage(),
            'enable_cache': self.enable_cache
        }
        rendered_records = []
        for record_name in self.record_names:
            record_dir = os.path.join(self.GS_records_dir, record_name)
            if os.path.exists(record_dir):
                cam_dirs = [d for d in os.listdir(record_dir) if d.startswith('cam_')]
                if cam_dirs:
                    cam_stats = {}
                    for cam_dir in cam_dirs:
                        cam_path = os.path.join(record_dir, cam_dir)
                        rgb_variants = [d for d in os.listdir(cam_path) if d.startswith('rgb_')]
                        if rgb_variants:
                            cam_stats[cam_dir] = {'num_variants': len(rgb_variants), 'variants': {}}
                            for variant in rgb_variants:
                                variant_path = os.path.join(cam_path, variant)
                                num_frames = len([f for f in os.listdir(variant_path) if f.endswith('.png')])
                                cam_stats[cam_dir]['variants'][variant] = num_frames
                        else:
                            rgbs_dir = os.path.join(cam_path, 'rgbs')
                            if os.path.exists(rgbs_dir):
                                cam_stats[cam_dir] = len([f for f in os.listdir(rgbs_dir) if f.endswith('.png')])
                    if cam_stats:
                        rendered_records.append({'record_name': record_name, 'cameras': cam_stats})
        stats['rendered_records'] = rendered_records
        return stats

    def __del__(self):
        try:
            if hasattr(self, 'clear_cache'):
                self.clear_cache()
        except Exception:
            pass
