import logging
import os
import numpy as np
from typing import Optional, Any, Dict, Union
from pathlib import Path
logger = logging.getLogger(__name__)

o3d: Any
gui: Any
rendering: Any
try:
    import open3d as o3d  # type: ignore
    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering
except ImportError:
    o3d = None
    gui = None
    rendering = None
    logger.error(
        "Open3D is not installed. Please install it using:\n"
        "  pip install open3d>=0.17\n"
        "or\n"
        "  conda install -c open3d-admin open3d"
    )


class PoseMeshViewer:
    """
    Visualize pose sequences with mesh models using Open3D GUI.
    
    - Load mesh files (obj, mtl) for objects and TCP end-effector
    - Interactive 3D window (orbit/pan/zoom)
    - Right panel slider to select current frame
    - Show trajectory, current pose frames, and mesh models at current frame
    - Automatically searches for mesh files in assets/meshes/{name}/ directory
    """

    def __init__(
        self,
        pose_dict: Dict[str, np.ndarray],
        mesh_base_dir: Optional[Union[str, Path]] = None,
        axes_size: float = 0.1,
        line_width: float = 2.0,
    ):
        """
        Args:
            pose_dict: Dictionary mapping object/eef names to pose arrays (N, 4, 4)
            mesh_base_dir: Base directory for mesh files (default: assets/meshes relative to project root)
            axes_size: Size of coordinate frame axes
            line_width: Width of trajectory lines
        """
        if o3d is None:
            raise ImportError(
                "Open3D is not installed. Please install it using:\n"
                "  pip install open3d>=0.17\n"
                "or\n"
                "  conda install -c open3d-admin open3d"
            )
        
        # Find mesh base directory
        if mesh_base_dir is None:
            # Try to find assets/meshes relative to common project locations
            current_file = Path(__file__).resolve()
            # Go up from poseprocess/vis/pose_mesh_vis.py to project root
            project_root = current_file.parent.parent.parent
            mesh_base_dir = project_root / "assets" / "meshes"
            if not mesh_base_dir.exists():
                # Try alternative: look for assets in parent directories
                for parent in current_file.parents:
                    potential_mesh_dir = parent / "assets" / "meshes"
                    if potential_mesh_dir.exists():
                        mesh_base_dir = potential_mesh_dir
                        break
        self.mesh_base_dir = Path(mesh_base_dir) if mesh_base_dir else None
        
        self.pose_dict = {}
        for name, poses in pose_dict.items():
            poses = np.asarray(poses, dtype=np.float64)
            assert poses.ndim == 3 and poses.shape[1:] == (4, 4), f"Poses for {name} must be (N,4,4)"
            self.pose_dict[name] = poses
        
        # Determine number of frames
        if not self.pose_dict:
            raise ValueError("No pose data provided")
        self.N = max(len(poses) for poses in self.pose_dict.values())
        
        # Verify all pose sequences have the same length
        for name, poses in self.pose_dict.items():
            if len(poses) != self.N:
                logger.warning(f"{name} has {len(poses)} poses, expected {self.N}. Padding with last pose.")
                # Pad with last pose
                last_pose = poses[-1]
                padded = np.array([last_pose] * (self.N - len(poses)))
                self.pose_dict[name] = np.vstack([poses, padded])
        
        self.axes_size = axes_size
        self.line_width = line_width
        
        # Load meshes
        self.meshes = {}
        self._load_meshes()

        try:
            self.app = gui.Application.instance
            self.app.initialize()
            self.window = self.app.create_window("Pose Mesh Viewer", 1280, 800)
            self._build_ui()
        except Exception as e:
            logger.warning(f"Failed to initialize GUI visualization: {e}")
            logger.info("Falling back to headless mode or skipping visualization")
            self.app = None
            self.window = None

    def _find_mesh_file(self, name: str) -> Optional[Path]:
        """Find mesh file for a given name in assets/meshes/{name}/ directory"""
        if self.mesh_base_dir is None or not self.mesh_base_dir.exists():
            return None
        
        mesh_dir = self.mesh_base_dir / name
        if not mesh_dir.exists() or not mesh_dir.is_dir():
            return None
        
        # Look for .obj files first (preferred)
        obj_files = list(mesh_dir.glob("*.obj"))
        if obj_files:
            # Prefer files named exactly as the directory or "model.obj"
            preferred_names = [name + ".obj", "model.obj", "mesh.obj"]
            for preferred in preferred_names:
                preferred_path = mesh_dir / preferred
                if preferred_path.exists():
                    return preferred_path
            # Otherwise return the first .obj file found
            return obj_files[0]
        
        # Fallback: look for other common mesh formats
        for ext in [".ply", ".stl", ".off"]:
            mesh_files = list(mesh_dir.glob(f"*{ext}"))
            if mesh_files:
                return mesh_files[0]
        
        return None

    def _load_meshes(self):
        """Load mesh files for all objects/eef from assets/meshes directory"""
        for name in self.pose_dict.keys():
            mesh_path = self._find_mesh_file(name)
            if mesh_path is None:
                logger.debug(f"No mesh file found for {name} in {self.mesh_base_dir / name if self.mesh_base_dir else 'assets/meshes'}")
                continue
            
            try:
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                if len(mesh.vertices) == 0:
                    logger.warning(f"Mesh {mesh_path} is empty. Skipping.")
                    continue
                mesh.compute_vertex_normals()
                self.meshes[name] = mesh
                logger.info(f"Loaded mesh for {name}: {mesh_path}")
            except Exception as e:
                logger.warning(f"Failed to load mesh {mesh_path} for {name}: {e}")

    def _build_ui(self):
        """Build the UI components"""
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])
        self.window.add_child(self.scene_widget)

        self.scene_widget.scene.scene.set_sun_light(
            [-1, -1, -1], [1.0, 1.0, 1.0], 50000
        )
        self.scene_widget.scene.scene.enable_sun_light(True)
        self.scene_widget.scene.show_axes(False)

        # Base frame
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axes_size)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene_widget.scene.add_geometry("base_frame", base_frame, mat)

        # Add meshes and pose frames for all objects/eef
        # Use different colors for eef vs objects
        eef_names = ["eef", "tcp", "end_effector", "gripper", "hand"]
        finger_names = ["leftfinger", "rightfinger", "left_finger", "right_finger"]
        is_eef = lambda name: any(eef_name in name.lower() for eef_name in eef_names)
        is_finger = lambda name: any(finger_name in name.lower() for finger_name in finger_names)
        
        for name, poses in self.pose_dict.items():
            # Skip pose frame and trajectory for fingers
            if not is_finger(name):
                # Create pose frame
                pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axes_size * 1.2)
                self.scene_widget.scene.add_geometry(f"pose_frame_{name}", pose_frame, mat)
            
            # Add mesh if available
            if name in self.meshes:
                mesh = self.meshes[name]
                mesh_mat = rendering.MaterialRecord()
                mesh_mat.shader = "defaultLit"
                self.scene_widget.scene.add_geometry(f"mesh_{name}", mesh, mesh_mat)
            else:
                # Create a sphere as placeholder
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.axes_size * 0.2)
                sphere.compute_vertex_normals()
                # Use red for eef, blue for objects
                if is_eef(name):
                    sphere.paint_uniform_color([1.0, 0.2, 0.2])
                else:
                    sphere.paint_uniform_color([0.7, 0.7, 0.9])
                mesh_mat = rendering.MaterialRecord()
                mesh_mat.shader = "defaultLit"
                self.scene_widget.scene.add_geometry(f"mesh_{name}", sphere, mesh_mat)
            
            # Skip trajectory for fingers
            if is_finger(name):
                continue
            
            # Add trajectory
            pts = poses[:, :3, 3]
            if len(pts) >= 2:
                # Check if all points are effectively the same (within tolerance)
                # This happens when pose_overwrite makes all poses identical
                pts_diff = np.max(pts, axis=0) - np.min(pts, axis=0)
                if np.max(pts_diff) > 1e-6:  # Only create trajectory if points span some distance
                    lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
                    traj = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(pts),
                        lines=o3d.utility.Vector2iVector(lines),
                    )
                    # Use red for eef trajectory, gray for objects
                    if is_eef(name):
                        traj_color = np.tile(np.array([[0.8, 0.2, 0.2]]), (len(lines), 1))
                    else:
                        traj_color = np.tile(np.array([[0.6, 0.6, 0.6]]), (len(lines), 1))
                    traj.colors = o3d.utility.Vector3dVector(traj_color)
                    line_mat = rendering.MaterialRecord()
                    line_mat.shader = "unlitLine"
                    line_mat.line_width = self.line_width
                    self.scene_widget.scene.add_geometry(f"trajectory_{name}", traj, line_mat)
                else:
                    logger.debug(f"Skipping trajectory for {name}: all poses are effectively identical (max diff: {np.max(pts_diff):.6f})")
            else:
                logger.debug(f"Skipping trajectory for {name}: insufficient poses ({len(pts)})")

        # Setup camera
        aabb = self.scene_widget.scene.bounding_box
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        center = 0.5 * (min_bound + max_bound)
        extent = float(np.linalg.norm(max_bound - min_bound))
        if not np.isfinite(extent) or extent < 1e-6:
            extent = 1.0
        self.scene_widget.setup_camera(60.0, aabb, center)
        self.scene_widget.look_at(center, center + np.array([1, -1, 1]) * extent, np.array([0, 0, 1]))

        # Add control panel
        em = self.window.theme.font_size
        panel = gui.Vert(0.5 * em, gui.Margins(0.75 * em, 0.75 * em, 0.75 * em, 0.75 * em))
        label = gui.Label(f"Frame: 0 / {self.N - 1}")
        self.frame_label = label
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(0, max(0, self.N - 1))
        slider.set_on_value_changed(self._on_slider)
        self.slider = slider
        panel.add_child(label)
        panel.add_child(slider)
        self.window.add_child(panel)
        self._panel = panel
        self.window.set_on_layout(self._on_layout)

        self._apply_frame(0)

    def _on_layout(self, layout_context):
        """Handle window layout"""
        r = self.window.content_rect
        panel_width = int(300 * self.window.scaling)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self._panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    def _on_slider(self, v):
        """Handle slider value change"""
        self._apply_frame(int(round(v)))

    def _apply_frame(self, idx: int):
        """Apply pose transformations for the given frame index"""
        if self.N == 0:
            return
        idx = int(np.clip(idx, 0, self.N - 1))
        
        finger_names = ["leftfinger", "rightfinger", "left_finger", "right_finger"]
        is_finger = lambda name: any(finger_name in name.lower() for finger_name in finger_names)
        
        # Update all poses
        for name, poses in self.pose_dict.items():
            if idx < len(poses):
                T = poses[idx]
                if not is_finger(name):
                    self.scene_widget.scene.set_geometry_transform(f"pose_frame_{name}", T)
                self.scene_widget.scene.set_geometry_transform(f"mesh_{name}", T)
        
        self.frame_label.text = f"Frame: {idx} / {self.N - 1}"

    def run(self):
        """Run the visualization"""
        self.app.run()

