"""
visualize the gripper width over time
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from poseprocess.vis.gripper_width_vis import plot_gripper_widths
from poseprocess.utils.angle_utils import read_angles_from_folder


def main():
    angles_npy_dir = "/home/ubuntu/data/Origin_Data/test/record_20251221_173256/angles"
    angles_list = read_angles_from_folder(angles_npy_dir)
    widths = np.array([float(angle[7]) for angle in angles_list])
    plot_gripper_widths(widths)


if __name__ == "__main__":
    main()