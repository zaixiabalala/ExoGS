import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

from .base import GaussianSceneBase
from .model import GaussianModel
from ..camera.constants import EYEINHAND_IDX

logger = logging.getLogger(__name__)


class GaussianSceneRenderer(GaussianSceneBase):
    """
    Renderer for Gaussian Splatting scenes with fixed and eye-in-hand cameras.
    """
    
    def __init__(self, Origin_dir: Optional[str] = None, GS_dir: Optional[str] = None, records_name: Optional[str] = None,
                 scene_name: Optional[str] = None, objects: Optional[List[str]] = None, enable_cache: bool = True, batch_size: int = 10,
                 robot_type: str = "flexiv"):
        super().__init__(Origin_dir=Origin_dir, GS_dir=GS_dir, records_name=records_name, scene_name=scene_name,
                        objects=objects, enable_cache=enable_cache, batch_size=batch_size, robot_type=robot_type)
        print(f"[Info] GaussianSceneRenderer initialized")

    def render_fixed_single_camera(self, record_name: str, cam_idx: int, scene_noise: Optional[List[float]],
                                   objects_poses_frames: List[List[torch.Tensor]], robot_links_poses: List[Dict[str, torch.Tensor]],
                                   rgb: bool = True, depth: bool = True, ply: bool = False, alpha: bool = False, use_async_save: bool = True):
        """Render a single fixed camera for all frames in a record."""
        assert self.cameras is not None, "[Error] Camera poses not initialized"
        num_frames = len(robot_links_poses)
        for obj_idx, obj_poses in enumerate(objects_poses_frames):
            assert len(obj_poses) == num_frames, f"[Error] object_{obj_idx} pose count({len(obj_poses)}) != robot pose count({num_frames})"
        
        record_output_dir = os.path.join(self.GS_records_dir, record_name)
        cam_folder_name = self.get_camera_folder_name(cam_idx)
        cam_dir = os.path.join(record_output_dir, cam_folder_name)
        
        if rgb:
            os.makedirs(os.path.join(cam_dir, 'rgbs'), exist_ok=True)
        if depth:
            os.makedirs(os.path.join(cam_dir, 'depths'), exist_ok=True)
        if alpha:
            os.makedirs(os.path.join(cam_dir, 'alphas'), exist_ok=True)
        if ply:
            os.makedirs(os.path.join(record_output_dir, 'plys'), exist_ok=True)
        
        camera = self.cameras[cam_idx]
        with self._async_saver(enable=use_async_save, max_workers=2) as saver:
            pbar = tqdm(total=num_frames, desc=f"Cam_{cam_idx}", unit="frame")
            for batch_start in range(0, num_frames, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_frames)
                batch_results = []
                for i in range(batch_start, batch_end):
                    frame_objects_poses = [obj_poses[i] for obj_poses in objects_poses_frames]
                    result = self.render_single_frame(
                        camera=camera,
                        objects_poses=frame_objects_poses,
                        robot_links_pose=robot_links_poses[i],
                        scene_noise=scene_noise,
                        output_rgb=rgb,
                        output_depth=depth,
                        output_ply=ply,
                        output_alpha=alpha,
                        return_on_gpu=True
                    )
                    batch_results.append((i, result))
                
                torch.cuda.synchronize(self.device)
                save_tasks = []
                for i, result in batch_results:
                    if rgb and 'rgb' in result:
                        save_tasks.append({
                            'type': 'rgb',
                            'data': result['rgb'].cpu(),
                            'path': os.path.join(cam_dir, 'rgbs', f'{i:05d}.png')
                        })
                    if depth and 'depth' in result:
                        save_tasks.append({
                            'type': 'depth',
                            'data': result['depth'].detach().cpu(),
                            'path': os.path.join(cam_dir, 'depths', f'{i:05d}.png')
                        })
                    if alpha and 'alpha' in result:
                        save_tasks.append({
                            'type': 'alpha',
                            'data': result['alpha'].detach().cpu(),
                            'path': os.path.join(cam_dir, 'alphas', f'{i:05d}.png')
                        })
                    if ply and 'point_cloud' in result:
                        save_tasks.append({
                            'type': 'ply',
                            'data': result['point_cloud'],
                            'path': os.path.join(record_output_dir, 'plys', f'{i:05d}.ply'),
                            'target_points': 10000
                        })
                
                del batch_results
                if self.merged_gaussians is not None:
                    del self.merged_gaussians
                    self.merged_gaussians = None
                if (batch_start // self.batch_size) % 5 == 0:
                    torch.cuda.empty_cache()
                if save_tasks:
                    saver.submit(self._save_batch, save_tasks)
                    saver.wait_batch(max_pending=2)
                pbar.update(batch_end - batch_start)
            pbar.close()

    def render_batch_fixed_cameras(self, rgb: bool = True, depth: bool = True, ply: bool = False, alpha: bool = False,
                                  scene_noise_range: Optional[List[List[float]]] = None, record_indices: Optional[List[int]] = None, use_async_save: bool = True):
        """Render all fixed cameras for specified records."""
        assert self.cameras is not None, "[Error] Camera poses not initialized"
        records_to_process = list(enumerate(self.record_names)) if record_indices is None else [(idx, self.record_names[idx]) for idx in record_indices]
        
        for i, record_name in records_to_process:
            print(f"\n[Info][Record {i+1}/{len(self.record_names)}] Processing: {record_name}")
            print(f"[Info] Loading pose data...")
            objects_poses_frames = self.load_objects_poses(record_name)
            robot_links_poses = self.load_robot_links_poses(record_name)
            scene_noise = [np.random.uniform(scene_noise_range[j][0], scene_noise_range[j][1]) for j in range(3)] if scene_noise_range is not None else None
            num_frames = len(robot_links_poses)
            
            for obj_idx, obj_poses in enumerate(objects_poses_frames):
                assert len(obj_poses) == num_frames, f"[Error] object_{obj_idx} pose count({len(obj_poses)}) != robot pose count({num_frames})"
            
            for cam_idx in range(len(self.cameras)):
                if cam_idx != 0 and not rgb and not depth and ply:
                    continue
                print(f"[Info][Cam {cam_idx+1}/{len(self.cameras)}] Starting render")
                self.render_fixed_single_camera(
                    record_name=record_name,
                    cam_idx=cam_idx,
                    scene_noise=scene_noise,
                    objects_poses_frames=objects_poses_frames,
                    robot_links_poses=robot_links_poses,
                    rgb=rgb,
                    depth=depth,
                    ply=ply and (cam_idx == 0),
                    alpha=alpha,
                    use_async_save=use_async_save
                )

    def render_eyeinhand_camera(self, record_name: str, scene_noise: Optional[List[float]], objects_poses_frames: List[List[torch.Tensor]],
                               robot_links_poses: List[Dict[str, torch.Tensor]], eyeinhand_cameras: List, cam_idx: int = EYEINHAND_IDX,
                               rgb: bool = True, depth: bool = True, alpha: bool = False, use_async_save: bool = True):
        """Render eye-in-hand camera for all frames in a record."""
        num_frames = len(robot_links_poses)
        assert len(eyeinhand_cameras) == num_frames, "[Error] Camera count != robot pose count"
        for obj_idx, obj_poses in enumerate(objects_poses_frames):
            assert len(obj_poses) == num_frames, f"[Error] object_{obj_idx} pose count != robot pose count"
        
        record_output_dir = os.path.join(self.GS_records_dir, record_name)
        cam_folder_name = self.get_camera_folder_name(cam_idx)
        cam_dir = os.path.join(record_output_dir, cam_folder_name)
        
        if rgb:
            os.makedirs(os.path.join(cam_dir, 'rgbs'), exist_ok=True)
        if depth:
            os.makedirs(os.path.join(cam_dir, 'depths'), exist_ok=True)
        if alpha:
            os.makedirs(os.path.join(cam_dir, 'alphas'), exist_ok=True)
        
        with self._async_saver(enable=use_async_save, max_workers=2) as saver:
            pbar = tqdm(total=num_frames, desc="EyeInHand", unit="frame")
            for batch_start in range(0, num_frames, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_frames)
                batch_results = []
                for i in range(batch_start, batch_end):
                    frame_objects_poses = [obj_poses[i] for obj_poses in objects_poses_frames]
                    result = self.render_single_frame(
                        camera=eyeinhand_cameras[i],
                        objects_poses=frame_objects_poses,
                        robot_links_pose=robot_links_poses[i],
                        scene_noise=scene_noise,
                        output_rgb=rgb,
                        output_depth=depth,
                        output_ply=False,
                        output_alpha=alpha,
                        return_on_gpu=True
                    )
                    batch_results.append((i, result))
                
                torch.cuda.synchronize(self.device)
                save_tasks = []
                for i, result in batch_results:
                    if rgb and 'rgb' in result:
                        save_tasks.append({
                            'type': 'rgb',
                            'data': result['rgb'].cpu(),
                            'path': os.path.join(cam_dir, 'rgbs', f'{i:05d}.png')
                        })
                    if depth and 'depth' in result:
                        save_tasks.append({
                            'type': 'depth',
                            'data': result['depth'].detach().cpu(),
                            'path': os.path.join(cam_dir, 'depths', f'{i:05d}.png')
                        })
                    if alpha and 'alpha' in result:
                        save_tasks.append({
                            'type': 'alpha',
                            'data': result['alpha'].detach().cpu(),
                            'path': os.path.join(cam_dir, 'alphas', f'{i:05d}.png')
                        })
                
                if save_tasks:
                    saver.submit(self._save_batch, save_tasks)
                    saver.wait_batch(max_pending=2)
                pbar.update(batch_end - batch_start)
            pbar.close()

    def render_batch_eyeinhand_cameras(self, cam_idx: int = EYEINHAND_IDX, rgb: bool = True, depth: bool = True, alpha: bool = False,
                                      scene_noise_range: Optional[List[List[float]]] = None, record_indices: Optional[List[int]] = None, use_async_save: bool = True):
        """Render eye-in-hand cameras for specified records."""
        records_to_process = list(enumerate(self.record_names)) if record_indices is None else [(idx, self.record_names[idx]) for idx in record_indices]
        
        for i, record_name in records_to_process:
            print(f"\n[Info][Record {i+1}/{len(self.record_names)}] Processing: {record_name}")
            print(f"[Info] Loading pose data...")
            objects_poses_frames = self.load_objects_poses(record_name)
            robot_links_poses = self.load_robot_links_poses(record_name)
            eyeinhand_cameras = self.load_eyeinhand_cameras(record_name)
            scene_noise = [np.random.uniform(scene_noise_range[j][0], scene_noise_range[j][1]) for j in range(3)] if scene_noise_range is not None else None
            num_frames = len(robot_links_poses)
            
            assert len(eyeinhand_cameras) == num_frames, "[Error] Camera count != robot pose count"
            for obj_idx, obj_poses in enumerate(objects_poses_frames):
                assert len(obj_poses) == num_frames, f"[Error] object_{obj_idx} pose count != robot pose count"
            
            print(f"[Info][Cam: Eye-in-Hand] Starting render")
            self.render_eyeinhand_camera(
                record_name=record_name,
                scene_noise=scene_noise,
                objects_poses_frames=objects_poses_frames,
                robot_links_poses=robot_links_poses,
                eyeinhand_cameras=eyeinhand_cameras,
                cam_idx=cam_idx,
                rgb=rgb,
                depth=depth,
                alpha=alpha,
                use_async_save=use_async_save
            )
            print(f"[Info][Record {i+1}/{len(self.record_names)}] Complete: {record_name}")
