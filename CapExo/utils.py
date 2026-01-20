import numpy as np
import pybullet as p

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mapping(angles: np.ndarray, middles: np.ndarray = np.array([77.08, 318.34, 285.21, 207.33, 238.27, 348.22, 312.89, 79.5])) -> np.ndarray:
    """Map exoskeleton encoder angles (degrees) to robot joint angles (radians) and gripper distance (meters)."""
    assert len(angles) == 8

    angles[5] += 90.0
    angles[:7] = (angles[:7] - middles[:7] + 180.0) % 360.0 - 180.0
    angles[5] -= 90.0
    angles[:7] *= -1.0 * np.pi / 180.

    angles[7] = (angles[7] - middles[7]) * np.pi / 180. * 18. / 1000.
    if angles[7] < 0:
        angles[7] = 0

    return angles


class TCPPoseCalculator:
    """Compute TCP pose from joint angles using PyBullet FK."""

    def __init__(
        self,
        urdf_path: str = "/home/ubuntu/workspace/CapExo/assets/exo/urdf/RizonExoUrdf.urdf",
        end_effector_index: int = 7
    ):
        self.urdf_path = urdf_path
        self.end_effector_index = end_effector_index
        self.robot_id = None
        self._initialize_pybullet()

    def _initialize_pybullet(self):
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)

        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True
        )

    def compute_tcp_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        assert len(joint_angles) == 7, f"Expected 7 joint angles, got {len(joint_angles)}"

        for i in range(7):
            p.resetJointState(self.robot_id, i + 1, targetValue=joint_angles[i])

        p.stepSimulation()

        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
        end_effector_pos = np.array(end_effector_state[0])
        end_effector_orn = np.array(end_effector_state[1])

        tcp_pose = np.concatenate([end_effector_pos, end_effector_orn])

        return tcp_pose

    def __del__(self):
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
