import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import numpy as np
from tap import Tap
from typing import List
from poseprocess.poseprocessor.processor import PoseProcessorProcessor
from poseprocess.utils.logging_config import setup_logging


class ArgumentParser(Tap):

    records_dir: str = "/media/ubuntu/B0A8C06FA8C0361E/Data/Origin_Data/records_dumbbell_1017"
    record_name: str = "record_20251017_210347"
    eef_type: str = "franka"
    object_name_list: List[str] = ["plate", "banana"]

    offset_vector: np.ndarray = np.array([0.0, 0.0, 0.006]) # offset vector in base coordinate system
    grasp_delay: int = -0
    release_delay: int = -0
    pose_file_type: str = "txt"
    pose_init_reference: str = "base"
    
    # 这些将在 configure 方法中设置
    tcp_pose_dir: str = ""
    angles_dir: str = ""
    gripper_width_list: List[float] = []
    object_pose_dir_list: List[str] = []
    object_pose_writeback_dir_list: List[str] = []
    
    def configure(self):
        # 在实例化后计算这些依赖值
        self.tcp_pose_dir = os.path.join(self.records_dir, self.record_name, "tcps")
        self.angles_dir = os.path.join(self.records_dir, self.record_name, "angles")
        
        # 加载 gripper_width_list
        if os.path.exists(self.angles_dir):
            for angles_file in sorted(os.listdir(self.angles_dir)):
                self.gripper_width_list.append(np.load(os.path.join(self.angles_dir, angles_file))[7])
        
        # 计算 object_pose_dir_list 和 object_pose_writeback_dir_list
        self.object_pose_dir_list = [
            os.path.join(self.records_dir, self.record_name, "poses", f"object_{i+1}_org")
            for i in range(len(self.object_name_list))
        ]
        self.object_pose_writeback_dir_list = [
            os.path.join(self.records_dir, self.record_name, "poses", f"object_{i+1}")
            for i in range(len(self.object_name_list))
        ]
    
def main(args:ArgumentParser):
    setup_logging(
        log_level=logging.INFO,
        log_file="logs/test.log"
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting pose processing")

    pose_processor = PoseProcessorProcessor(
        pose_init_reference=args.pose_init_reference,
        object_name_list=args.object_name_list,
        object_pose_dir_list=args.object_pose_dir_list,
        eef_type=args.eef_type,
        tcp_pose_dir=args.tcp_pose_dir,
        tcp_offset=np.array([0.0, 0.0, 0.06]),
        gripper_width_list=args.gripper_width_list,
    )

    pose_processor.pose_overwrite(
        object_name_list=[args.object_name_list[0]],
        overwrite_matrix=None,
    )

    # pose_processor.pose_offset(
    #     object_name_list=[args.object_name_list[1]],
    #     offset_vector=args.offset_vector,
    # )

    # pose_processor.pose_rotate(
    #     object_name_list=[args.object_name_list[1]],
    #     axis="x",
    #     angle_deg=90,
    # )

    # pose_processor.set_container_bbox(
    #     container_name="plate", 
    #     bbox_size_max=np.array([0.07, 0.07, 0.0225 - 0.005]), 
    #     bbox_size_min=np.array([-0.07, -0.07, -0.0225 + 0.005]), 
    #     reference_up_axis=np.array([0.0, 0.0, 1.0])
    # )

    # pose_processor.pose_freeze(
    #     object_name_list=[args.object_name_list[1]],
    # )

    # pose_processor.pose_fix(
    #     object_name_list=[args.object_name_list[1]],
    #     grasp_delay=args.grasp_delay,
    #     release_delay=args.release_delay,
    # )

    pose_processor.pose_visualize()

    pose_processor.pose_writeback(
        object_name_list=args.object_name_list,
        pose_output_dir_list=args.object_pose_writeback_dir_list,
        force_overwrite=True,
        file_name_format="{i:016d}.txt",
    )

    logger.info(f"{args.record_name} pose processing finished")

    # pose_processor.show_states()
    
if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)