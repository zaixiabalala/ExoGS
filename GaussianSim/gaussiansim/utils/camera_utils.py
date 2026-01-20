import numpy as np
from typing import List, Tuple, Optional

from ..camera.camera import Camera
from .graphics_utils import fov2focal


def camera_to_JSON(id, camera: Camera):
    """Convert Camera object to JSON-serializable dictionary."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    
    camera_entry = {
        'id': id,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def loadCam(id, cam_info):
    """Load Camera object from CameraInfo."""
    return Camera(
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        FovX=cam_info.FovX,
        FovY=cam_info.FovY,
        width=cam_info.width,
        height=cam_info.height
    )


def cameraList_from_camInfos(cam_infos):
    """Convert list of CameraInfo to list of Camera objects."""
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(id, c))
    return camera_list


def calculate_target_distance(origin_camera_pose: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate the intersection distance of camera ray with xy plane.
    
    Args:
        origin_camera_pose: 4x4 camera extrinsic matrix
    
    Returns:
        float: Distance from camera to intersection point with xy plane
        np.ndarray: Target point position
    """
    # Camera position
    camera_pos = origin_camera_pose[:3, 3]

    # Camera forward direction
    camera_forward = origin_camera_pose[:3, :3] @ np.array([0, 0, -1])

    # Calculate intersection with xy plane (z=0)
    # Ray equation: P = camera_pos + t * camera_forward
    # xy plane equation: z = 0
    # Solve: camera_pos[2] + t * camera_forward[2] = 0
    # Result: t = -camera_pos[2] / camera_forward[2]
    if abs(camera_forward[2]) < 1e-6:
        raise ValueError("[Error] Camera direction parallel to xy plane, please check camera extrinsic parameters")

    t = -camera_pos[2] / camera_forward[2]

    # Calculate intersection point
    intersection_point = camera_pos + t * camera_forward

    # Calculate distance
    distance = np.linalg.norm(intersection_point - camera_pos)

    return float(distance), intersection_point


def generate_spherical_camera_array(
    origin_camera_pose: np.ndarray,
    phi_values: List[float],
    camera_counts: List[int]
) -> List[np.ndarray]:
    """
    Generate multi-layer spherical camera array, all cameras look at the same target point.
    
    Args:
        origin_camera_pose: 4x4 center camera extrinsic matrix
        phi_values: List of polar angles (degrees) for each layer, starting from horizontal plane
        camera_counts: List of camera counts for each layer
    
    Returns:
        List of generated camera extrinsic matrices
    """
    # Calculate target distance (intersection of camera ray with xy plane)
    target_distance, target_pos = calculate_target_distance(origin_camera_pose)

    # Calculate camera view ray direction vector
    camera_forward = origin_camera_pose[:3, :3] @ np.array([0, 0, -1])
    camera_forward = camera_forward / np.linalg.norm(camera_forward)

    # Build coordinate system with camera view ray as z-axis
    camera_z_axis = camera_forward

    # Camera x-axis: perpendicular to camera z-axis
    if abs(np.dot(camera_z_axis, np.array([1, 0, 0]))) < 0.9:
        camera_x_axis = np.cross(camera_z_axis, np.array([1, 0, 0]))
    else:
        camera_x_axis = np.cross(camera_z_axis, np.array([0, 1, 0]))
    camera_x_axis = camera_x_axis / np.linalg.norm(camera_x_axis)

    # Camera y-axis: cross product of z-axis and x-axis
    camera_y_axis = np.cross(camera_z_axis, camera_x_axis)
    camera_y_axis = camera_y_axis / np.linalg.norm(camera_y_axis)

    # Build rotation matrix: from camera coordinate system to world coordinate system
    camera_to_world = np.column_stack([camera_x_axis, camera_y_axis, camera_z_axis])

    camera_poses = []

    for phi_deg, num_cameras in zip(phi_values, camera_counts):
        phi = np.radians(phi_deg)
        # Calculate layer radius (distance from target point)
        layer_radius = target_distance * np.sin(phi)
        layer_height = target_distance * np.cos(phi)
        
        # Generate azimuth angles for cameras in this layer
        azimuth_angles = np.linspace(0, 2*np.pi, num_cameras, endpoint=False)
        
        for azimuth in azimuth_angles:
            # Calculate position in camera coordinate system
            camera_x_local = layer_radius * np.cos(azimuth)
            camera_y_local = layer_radius * np.sin(azimuth)
            camera_z_local = layer_height
            
            # Transform to world coordinate system
            camera_pos_local = np.array([camera_x_local, camera_y_local, camera_z_local])
            camera_pos_world = target_pos + camera_to_world @ camera_pos_local
            
            # Calculate camera orientation (pointing to target)
            direction_to_target = target_pos - camera_pos_world
            direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
            
            # Build camera coordinate system
            # Camera x-axis should be perpendicular to world z-axis (for normal viewing angle)
            world_z = np.array([0, 0, 1])
            # If camera direction parallel to z-axis, use y-axis as reference
            if abs(np.dot(direction_to_target, world_z)) > 0.99:
                reference_axis = np.array([0, 1, 0])
            else:
                reference_axis = world_z
            
            # Calculate camera x-axis (right direction)
            camera_x_axis = np.cross(direction_to_target, reference_axis)
            camera_x_axis = camera_x_axis / np.linalg.norm(camera_x_axis)
            
            # Calculate camera y-axis (down direction)
            camera_y_axis = np.cross(direction_to_target, camera_x_axis)
            camera_y_axis = camera_y_axis / np.linalg.norm(camera_y_axis)
            
            # Build rotation matrix (camera coordinate system: X right, Y down, Z forward)
            rotation_matrix = np.column_stack([camera_x_axis, camera_y_axis, direction_to_target])
            
            # Build 4x4 extrinsic matrix
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = rotation_matrix
            camera_pose[:3, 3] = camera_pos_world
            camera_poses.append(camera_pose)
    
    return camera_poses


def visualize_camera_positions(
    camera_poses: List[np.ndarray],
    axis_len: float = 0.1,
    draw_frustum: bool = False,
    fov_y_deg: float = 60.0,
    aspect: float = 4.0/3.0,
    frustum_depth: float = 0.2,
    draw_view_rays: bool = True,
    ray_len: float = 0.4,
    target_point: Optional[np.ndarray] = None,
    title: Optional[str] = None
):
    """Visualize camera positions in 3D space with optional frustums and view rays."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[Error] Please install matplotlib: pip install matplotlib")
        return

    assert len(camera_poses) > 0, "camera_poses cannot be empty"
    for P in camera_poses:
        assert P.shape == (4, 4), "Each pose must be a 4x4 matrix"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = [], [], []

    # Precompute frustum corners (in camera coordinate system)
    frustum_pts_cam = None
    if draw_frustum:
        fov_y = np.deg2rad(fov_y_deg)
        h = 2.0 * frustum_depth * np.tan(fov_y * 0.5)
        w = h * aspect
        z = -frustum_depth
        frustum_pts_cam = np.array([
            [w/2,  h/2, z],
            [-w/2,  h/2, z],
            [-w/2, -h/2, z],
            [w/2, -h/2, z],
        ], dtype=np.float64)

    tp = None if target_point is None else np.asarray(target_point, dtype=np.float64).reshape(3,)

    for i, pose in enumerate(camera_poses):
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Camera local axes (right, down, forward) in world coordinates
        x_axis = R @ np.array([1, 0, 0], dtype=np.float64)
        y_axis = R @ np.array([0, 1, 0], dtype=np.float64)   # down
        z_axis = R @ np.array([0, 0, 1], dtype=np.float64)   # forward

        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])

        color = plt.cm.viridis(i / max(1, len(camera_poses)-1))
        ax.scatter(t[0], t[1], t[2], color=color, s=50, alpha=0.9)

        # Coordinate axes
        ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2],
                  length=axis_len, color='r', linewidth=1.5, alpha=0.9)
        ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2],
                  length=axis_len, color='g', linewidth=1.5, alpha=0.9)
        ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2],
                  length=axis_len, color='b', linewidth=1.5, alpha=0.9)

        # Frustum (optional)
        if frustum_pts_cam is not None:
            pts_w = (R @ frustum_pts_cam.T).T + t
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                ax.plot([pts_w[a, 0], pts_w[b, 0]],
                        [pts_w[a, 1], pts_w[b, 1]],
                        [pts_w[a, 2], pts_w[b, 2]],
                        color=color, alpha=0.7, linewidth=1.2)
            for k in range(4):
                ax.plot([t[0], pts_w[k, 0]],
                        [t[1], pts_w[k, 1]],
                        [t[2], pts_w[k, 2]],
                        color=color, alpha=0.4, linewidth=1.0)

        # View direction rays
        if draw_view_rays:
            if tp is None:
                ray_end = t + z_axis * ray_len  # Fixed length
                ax.plot([t[0], ray_end[0]], [t[1], ray_end[1]], [t[2], ray_end[2]],
                        color='k', alpha=0.6, linewidth=1.5)
            else:
                ax.plot([t[0], tp[0]], [t[1], tp[1]], [t[2], tp[2]],
                        color='k', alpha=0.4, linewidth=1.2, linestyle='--')

    # Ground xy plane
    try:
        xx, yy = np.meshgrid(
            np.linspace(min(xs+[-1]), max(xs+[1]), 10),
            np.linspace(min(ys+[-1]), max(ys+[1]), 10)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray', edgecolor='none')
    except Exception:
        pass

    # Draw target point if provided
    if tp is not None:
        ax.scatter(tp[0], tp[1], tp[2], c='orange', s=80, label='Target', alpha=0.9)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title or 'Cameras (poses)')

    if len(xs) > 0:
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)
        max_range = max(x_range, y_range, z_range, 1e-6)
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        cz = (max(zs) + min(zs)) / 2
        margin = max_range * 0.25
        ax.set_xlim(cx - max_range/2 - margin, cx + max_range/2 + margin)
        ax.set_ylim(cy - max_range/2 - margin, cy + max_range/2 + margin)
        ax.set_zlim(cz - max_range/2 - margin, cz + max_range/2 + margin)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)
