# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import logging
import math
import numpy as np
from typing import List, Optional, Tuple
from assets.camera_constants import CAM_TO_BASE_MATRIX_319522062799, CAM_TO_BASE_MATRIX_327322062498
from scipy.spatial.transform import Rotation
logger = logging.getLogger(__name__)

def transform_poses_cam_to_base(pose_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    transform poses from camera coordinate system to base coordinate system

    Args:
        pose_list: a list of 4x4 numpy arrays
    
    Returns:
        a list of 4x4 numpy arrays
    """
    if len(pose_list) == 0:
        logger.error("pose list is empty");raise ValueError("pose list is empty") from None
    
    return [CAM_TO_BASE_MATRIX @ pose for pose in pose_list]


def transform_poses_base_to_cam(pose_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    transform poses from base coordinate system to camera coordinate system
    
    Args:
        pose_list: a list of 4x4 numpy arrays
    
    Returns:
        a list of 4x4 numpy arrays
    """
    if len(pose_list) == 0:
        logger.error("pose list is empty");raise ValueError("pose list is empty") from None
    
    logger.debug(f"Transform {len(pose_list)} poses from base coordinate system to camera coordinate system")
    base_to_cam = np.linalg.inv(CAM_TO_BASE_MATRIX)
    return [base_to_cam @ pose for pose in pose_list]

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    归一化四元数，接受形如 [qx, qy, qz, qw]。
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def xyzquat_to_matrix(xyzquat: np.ndarray) -> np.ndarray:
    """
    convert (7,) format pose [x, y, z, qx, qy, qz, qw] to 4x4 homogeneous transformation matrix
    """
    xyzquat = np.asarray(xyzquat, dtype=np.float64)
    if xyzquat.shape != (7,):
        e = f"xyzquat must be shape (7,), got {xyzquat.shape}";logger.error(e);raise ValueError(e) from None
    t = xyzquat[:3]
    q = xyzquat[3:7]
    rotation = Rotation.from_quat(q)
    R = rotation.as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """
    将 (7,) 形式位姿 [x, y, z, qx, qy, qz, qw] 转为 4x4 齐次矩阵。
    """
    pose7 = np.asarray(pose7, dtype=np.float64)
    assert pose7.shape == (7,), "pose7 必须为形状 (7,)"
    t = pose7[:3]
    q = pose7[3:7]
    R = quaternion_to_rotation_matrix(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def poses_to_matrices(arr: np.ndarray) -> np.ndarray:
    """
    支持形状：
      - (N, 7): [x, y, z, qx, qy, qz, qw]
      - (N, 4, 4): 齐次矩阵
    返回 (N, 4, 4)
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        Ts = arr.astype(np.float64).copy()
        # 规范化最后一行
        Ts[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return Ts
    if arr.ndim == 2 and arr.shape[1] == 7:
        N = arr.shape[0]
        Ts = np.repeat(np.eye(4, dtype=np.float64)[None, ...], N, axis=0)
        for i in range(N):
            Ts[i] = pose7_to_matrix(arr[i])
        return Ts
    raise ValueError("输入必须是 (N,7) 或 (N,4,4) 的数组")


def invert_se3(T: np.ndarray) -> np.ndarray:
    """
    求 4x4 齐次矩阵的逆（SE(3)）。
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def compose_se3(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    SE(3) 复合 A ∘ B，对应矩阵乘法 A @ B。
    """
    return (np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64))


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    用 4x4 齐次矩阵 T 变换 (N,3) 点集。
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(points, dtype=np.float64)
    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    Ph = np.concatenate([P, ones], axis=1)
    Qh = (T @ Ph.T).T
    return Qh[:, :3]


def rpy_to_rot(rpy: List[float]) -> np.ndarray:
    """
    将 RPY 欧拉角转换为 3x3 旋转矩阵。
    旋转顺序为: Rz(yaw) @ Ry(pitch) @ Rx(roll)
    
    参数:
        rpy: [roll, pitch, yaw]
    返回:
        3x3 旋转矩阵
    """
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ], dtype=np.float64)
    return R


def axis_angle_to_rot(axis: List[float], angle: float) -> np.ndarray:
    """
    由旋转轴与角度生成 3x3 旋转矩阵（Rodrigues 公式）。
    
    参数:
        axis: 旋转轴向量 [ax, ay, az]
        angle: 旋转角（弧度）
    返回:
        3x3 旋转矩阵
    """
    axis = np.array(axis, dtype=float)
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3, dtype=np.float64)

    u = axis / np.linalg.norm(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1 - c

    ux, uy, uz = u
    K = np.array([
        [0,   -uz,  uy],
        [uz,   0,  -ux],
        [-uy, ux,   0]
    ], dtype=np.float64)

    # Rodrigues: R = I + s*K + (1-c)*K^2
    R = np.eye(3, dtype=np.float64) + s * K + one_c * (K @ K)
    return R


def align_pose_z_axis_to_reference(
        pose: np.ndarray, 
        reference_up_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
        ) -> Tuple[np.ndarray, str, np.ndarray]:
    """
    Reorder axes so that the axis closest to the reference upward direction becomes z-axis.
    This only permutes the axes, preserving the object's orientation relative to the base coordinate system.
    The object's pose in base frame remains unchanged, only the axis labels are swapped.
    
    Args:
        pose: 4x4 homogeneous transformation matrix
        reference_up_axis: Reference upward direction (default: [0, 0, 1] for base coordinate system z-axis)
    
    Returns:
        Tuple of (aligned_pose, detected_axis_name, R_align):
        - aligned_pose: 4x4 homogeneous transformation matrix with reordered axes
        - detected_axis_name: Name of the detected axis ('x', 'y', or 'z') that was initially closest to reference upward axis
        - R_align: 3x3 rotation matrix that can be applied to other poses in the sequence
                   Usage: new_pose[:3, :3] = R_align @ original_pose[:3, :3]
    """
    pose = np.asarray(pose, dtype=np.float64)
    reference_up_axis = np.asarray(reference_up_axis, dtype=np.float64)
    
    if pose.shape != (4, 4):
        e = f"Pose must be 4x4 matrix, got shape {pose.shape}";logger.error(e);raise ValueError(e) from None
    
    if len(reference_up_axis) != 3:
        e = f"reference_up_axis must be a 3D vector, got shape {reference_up_axis.shape}";logger.error(e);raise ValueError(e) from None
    
    ref_z = reference_up_axis / np.linalg.norm(reference_up_axis)
    
    R = pose[:3, :3]
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    # Detect which axis is currently closest to reference z-axis
    x_alignment = abs(np.dot(x_axis, ref_z))
    y_alignment = abs(np.dot(y_axis, ref_z))
    z_alignment = abs(np.dot(z_axis, ref_z))
    
    detected_axis_name = max({'x': x_alignment, 'y': y_alignment, 'z': z_alignment}.items(), key=lambda k: k[1])[0]
    
    # Reorder axes by permutation: make the detected axis become z-axis
    # This is a pure axis swap, not a rotation, so object's pose in base frame is preserved
    # The key: we only swap axis labels, keeping the object's actual orientation in base unchanged
    # The object should remain tilted/slanted in base frame, just with relabeled axes
    
    # Determine the permutation based on detected axis
    if detected_axis_name == 'x':
        # Original x-axis becomes new z-axis
        # We need to swap: (x, y, z) -> (y, z, x)
        # This means: new_z = old_x, new_x = old_y, new_y = old_z
        detected_axis = x_axis.copy()
        if np.dot(detected_axis, ref_z) < 0:
            detected_axis = -detected_axis
        new_z = detected_axis
        new_x = y_axis.copy()
        new_y = z_axis.copy()
        # Ensure right-handed coordinate system: cross(new_x, new_y) should point in new_z direction
        if np.dot(np.cross(new_x, new_y), new_z) < 0:
            new_y = -new_y
    elif detected_axis_name == 'y':
        # Original y-axis becomes new z-axis
        # We need to swap: (x, y, z) -> (z, x, y)
        # This means: new_z = old_y, new_x = old_z, new_y = old_x
        detected_axis = y_axis.copy()
        if np.dot(detected_axis, ref_z) < 0:
            detected_axis = -detected_axis
        new_z = detected_axis
        new_x = z_axis.copy()
        new_y = x_axis.copy()
        # Ensure right-handed coordinate system
        if np.dot(np.cross(new_x, new_y), new_z) < 0:
            new_y = -new_y
    else:  # detected_axis_name == 'z'
        # z-axis is already closest, just check direction
        detected_axis = z_axis.copy()
        if np.dot(detected_axis, ref_z) < 0:
            # Flip z-axis to match reference direction
            new_z = -detected_axis
            # Flip one other axis to maintain right-handed system
            new_x = -x_axis.copy()
            new_y = y_axis.copy()
        else:
            # No change needed, axes stay the same
            new_z = detected_axis
            new_x = x_axis.copy()
            new_y = y_axis.copy()
    
    # Construct new rotation matrix with reordered axes
    # This represents the same orientation in base frame, just with different axis labels
    # The object remains tilted/slanted in base, not aligned with base axes
    R_new = np.column_stack([new_x, new_y, new_z])
    
    # Calculate transformation matrix: R_align transforms from original to reordered axes
    # Since R_new = R_align @ R, we have R_align = R_new @ R.T
    R_align = R_new @ R.T
    
    # Apply reordering to the pose (rotation part only, keep translation unchanged)
    # aligned_pose[:3, :3] = R_align @ R = R_new, which is the reordered axes
    aligned_pose = pose.copy()
    aligned_pose[:3, :3] = R_new
    
    return aligned_pose, detected_axis_name, R_align


def visualize_pose_axes(
        pose: np.ndarray,
        axes_size: float = 0.1,
        window_name: str = "Pose Axes Visualization"
    ) -> None:
    """
    Visualize pose coordinate axes in 3D world coordinate system using matplotlib.
    
    Args:
        pose: 4x4 homogeneous transformation matrix
        axes_size: Size of the coordinate axes (default: 0.1)
        window_name: Name of the visualization window (used as figure title)
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is not installed. Please install with: pip install matplotlib")
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        e = f"Pose must be 4x4 matrix, got shape {pose.shape}";logger.error(e);raise ValueError(e)
    
    # Extract position and rotation
    position = pose[:3, 3]
    rotation = pose[:3, :3]
    
    # Extract axes from pose rotation matrix
    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    z_axis = rotation[:, 2]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw world coordinate frame at origin (gray, larger)
    world_axes_size = axes_size * 1.2
    ax.quiver(0, 0, 0, world_axes_size, 0, 0, color='gray', arrow_length_ratio=0.2, linewidth=2, alpha=0.5, label='World X')
    ax.quiver(0, 0, 0, 0, world_axes_size, 0, color='gray', arrow_length_ratio=0.2, linewidth=2, alpha=0.5, label='World Y')
    ax.quiver(0, 0, 0, 0, 0, world_axes_size, color='gray', arrow_length_ratio=0.2, linewidth=2, alpha=0.5, label='World Z')
    
    # Draw pose coordinate frame (colored arrows)
    ax.quiver(position[0], position[1], position[2], 
              x_axis[0] * axes_size, x_axis[1] * axes_size, x_axis[2] * axes_size,
              color='red', arrow_length_ratio=0.2, linewidth=2.5, label='X-axis')
    ax.quiver(position[0], position[1], position[2],
              y_axis[0] * axes_size, y_axis[1] * axes_size, y_axis[2] * axes_size,
              color='green', arrow_length_ratio=0.2, linewidth=2.5, label='Y-axis')
    ax.quiver(position[0], position[1], position[2],
              z_axis[0] * axes_size, z_axis[1] * axes_size, z_axis[2] * axes_size,
              color='blue', arrow_length_ratio=0.2, linewidth=2.5, label='Z-axis')
    
    # Draw position point
    ax.scatter([position[0]], [position[1]], [position[2]], color='black', s=50, marker='o', label='Position')
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'{window_name}\nPosition: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]', fontsize=14)
    ax.legend(loc='upper left')
    
    # Set equal aspect ratio
    max_range = max(abs(position)) + axes_size * 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Set grid
    ax.grid(True, alpha=0.3)
    
    # Log information
    logger.info(f"Visualizing pose at position: {position}")
    logger.info(f"X-axis direction: {x_axis}")
    logger.info(f"Y-axis direction: {y_axis}")
    logger.info(f"Z-axis direction: {z_axis}")
    
    # Show plot
    plt.tight_layout()
    plt.show()


def main():
    pose_in_cam = np.array([[7.792358994483947754e-01, -3.512832336127758026e-03, 6.267208456993103027e-01, 1.401628106832504272e-01],
                            [-4.048894047737121582e-01, -7.661183476448059082e-01, 4.991266429424285889e-01, 9.418264031410217285e-02],
                            [4.783890247344970703e-01, -6.426898837089538574e-01, -5.984093546867370605e-01, 9.672685265541076660e-01],
                            [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    from assets.camera_constants import CAM_TO_BASE_MATRIX
    pose_in_base = CAM_TO_BASE_MATRIX @ pose_in_cam
    aligned_pose, detected_axis_name, R_align = align_pose_z_axis_to_reference(pose_in_base, reference_up_axis=np.array([0.0, 0.0, 1.0]))
    
    print("Original pose in base:")
    print(pose_in_base)
    print(f"\nDetected axis name: {detected_axis_name}")
    print("\nRotation matrix (3x3):")
    print(R_align)
    print("\nAligned pose in base:")
    print(aligned_pose)
    
    # Visualize original pose
    print("\nVisualizing original pose...")
    visualize_pose_axes(pose_in_base, window_name="Original Pose")
    
    # Visualize aligned pose
    print("\nVisualizing aligned pose...")
    visualize_pose_axes(aligned_pose, window_name="Aligned Pose")

if __name__ == "__main__":
    main()