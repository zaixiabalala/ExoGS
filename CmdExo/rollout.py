import os
import json
import time
from tap import Tap
from typing import List, Tuple
import numpy as np
import cv2

from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.devices.gripper.xense.xense import Xense
from r3kit.devices.camera.realsense.general import RealSenseCamera
from r3kit.utils.buffer import ObsBuffer, ActBuffer
from r3kit.utils.vis import SequenceKeyboardListener
from r3kit.utils.vis import Sequence2DVisualizer, Sequence1DVisualizer
from r3kit.utils.transformation import xyzrot6d2mat, mat2xyzrot6d


class ArgumentParser(Tap):
    robot_id: str = 'Rizon4s-063231'
    robot_name: str = 'Rizon4s'
    block: bool = False
    wait_time: float = 0.1

    gripper_id: str = '5e77ff097831'
    gripper_name: str = 'Xense'

    camera_id: str = '327322062498'
    camera_streams: List[Tuple[str, int, int, int, int]] = [('color', -1, 640, 480, 30)]
    camera_name: str = 'D415'

    num_obs: int = 1
    num_actions: int = 1
    num_cmd: int = 1
    num_steps: int = -1
    sleep_time: float = 0.01

    meta_path: str = '/home/ubuntu/workspace/CmdExo/.meta'
    vis: bool = False
    action_mode: str = 'tcp'


def main(args: ArgumentParser):
    obs_dict, act_dict = {}, {}

    robot = Rizon(id=args.robot_id, gripper=False, name=args.robot_name, tool_name='xense')
    if args.action_mode == 'joint':
        robot.motion_mode('joint')
        robot.block(args.block)
        obs_dict['joints'] = ((7,), np.float32.__name__)
        act_dict['joints'] = ((7,), np.float32.__name__)
    else:
        robot.motion_mode('tcp')
        robot.block(args.block)
        obs_dict['tcp_pose'] = ((9,), np.float32.__name__)
        act_dict['tcp_pose'] = ((9,), np.float32.__name__)

    gripper = Xense(id=args.gripper_id, name=args.gripper_name)
    gripper.block(args.block)
    obs_dict['width'] = ((1,), np.float32.__name__)
    act_dict['width'] = ((1,), np.float32.__name__)

    camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)
    obs_dict['images'] = (camera.color_image_shape, np.uint8.__name__)

    obs_buffer = ObsBuffer(num_obs=args.num_obs, obs_dict=obs_dict, create=True)
    act_buffer = ActBuffer(num_act=args.num_actions, act_dict=act_dict, create=True)
    os.makedirs(args.meta_path, exist_ok=True)
    with open(os.path.join(args.meta_path, 'obs_dict.json'), 'w') as f:
        json.dump(obs_dict, f, indent=4)
    with open(os.path.join(args.meta_path, 'act_dict.json'), 'w') as f:
        json.dump(act_dict, f, indent=4)
    print(f"==========> Initialized (action_mode: {args.action_mode})")

    if args.num_steps < 0:
        listener = SequenceKeyboardListener(verbose=False)

    step_idx = 0
    quit = False
    while not quit:
        if step_idx % args.num_cmd == 0:
            o = {}
            if args.action_mode == 'joint':
                joints = robot.joint_read()
                o['joints'] = joints
            else:
                tcp_pose_mat = robot.tcp_read()
                xyz, rot6d = mat2xyzrot6d(tcp_pose_mat)
                tcp_pose_vec = np.concatenate([xyz, rot6d])
                o['tcp_pose'] = tcp_pose_vec

            width = gripper.read()
            o['width'] = width

            camera_data = camera.get()
            color_image = camera_data['color']
            o['images'] = color_image

            act_buffer.setf(False)
            obs_buffer.add1(o)
            obs_buffer.setf(True)
            print("Add obs", o.keys(), step_idx)

        while not act_buffer.getf():
            time.sleep(args.sleep_time)
        a = act_buffer.get1()
        print("Get act", a.keys(), step_idx)

        # skip first 50 steps for warm-up
        if step_idx > 50:
            if args.action_mode == 'joint':
                joints = a['joints']
                robot.joint_move(joints)
            else:
                tcp_pose_vec = a['tcp_pose']
                tcp_pose_mat = xyzrot6d2mat(tcp_pose_vec[:3], tcp_pose_vec[3:9])
                robot.tcp_move(tcp_pose_mat)

            width = a['width']
            gripper.move(width)

        if not args.block:
            time.sleep(args.wait_time)

        if args.num_steps == -1:
            quit = listener.quit
        else:
            quit = step_idx >= args.num_steps
        step_idx += 1


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
