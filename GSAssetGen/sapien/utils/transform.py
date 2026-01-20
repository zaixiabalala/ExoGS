import numpy as np
import sapien
from scipy.spatial.transform import Rotation


def matrix_to_pose(transformation_matrix):
    """
    Convert 4x4 transformation matrix to SAPIEN Pose object.

    Args:
        transformation_matrix: 4x4 transformation matrix, numpy array

    Returns:
        sapien.Pose: Corresponding Pose object containing position and quaternion
                     (quaternion order is wxyz)

    Raises:
        ValueError: If matrix is not 4x4 or format is incorrect
    """
    if transformation_matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")

    # Extract position (first three elements of fourth column)
    position = transformation_matrix[:3, 3]

    # Extract rotation matrix (top-left 3x3 submatrix)
    rotation_matrix = transformation_matrix[:3, :3]

    # Convert to quaternion
    quaternion = matrix_to_quaternion(rotation_matrix)

    # Create Pose object
    return sapien.Pose(position, quaternion)


def matrix_to_quaternion(rotation_matrix):
    """
    Convert 3x3 rotation matrix to quaternion.

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        numpy.ndarray: Quaternion [w, x, y, z]
    """
    # Ensure matrix is orthogonal
    rotation_matrix = rotation_matrix / np.linalg.norm(rotation_matrix, axis=0)

    # Use scipy method (recommended)
    r = Rotation.from_matrix(rotation_matrix)
    quat = r.as_quat()  # [x, y, z, w] format
    # Convert to [w, x, y, z] format
    return np.array([quat[3], quat[0], quat[1], quat[2]])
