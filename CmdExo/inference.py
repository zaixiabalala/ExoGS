import os
import json
import time
import argparse
import torch
import numpy as np
import cv2

from r3kit.utils.buffer import ObsBuffer, ActBuffer
from maskdp.policies.mask_act.modeling_mask_act import MaskACTPolicy


def parse_args():
    parser = argparse.ArgumentParser(description='Policy inference node')
    parser.add_argument('--model_path', type=str,
                        default='/home/ubuntu/workspace/ruogu/maskdp_train/outputs/mask_act/junbo_nomask_nopose/stage2/checkpoints/150000/pretrained_model',
                        help='Path to the pretrained model')
    parser.add_argument('--sleep_time', type=float, default=0.01,
                        help='Sleep time between observations')
    parser.add_argument('--meta_path', type=str, default='/home/ubuntu/workspace/CmdExo/.meta/',
                        help='Path to meta files')
    parser.add_argument('--action_mode', type=str, default='tcp', choices=['tcp', 'joint'],
                        help='Action mode: tcp or joint')
    parser.add_argument('--policy', type=str, default='act')
    return parser.parse_args()


def preprocess_obs(obs, device, action_mode):
    image = obs[0]["images"]
    width = obs[0]["width"]

    image = image[:, 100:580, :]
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image.reshape(1, 1, 3, 224, 224)

    if action_mode == 'joint':
        joints = obs[0]["joints"]
        state = np.concatenate([joints, width], axis=-1)
    else:
        tcp_pose = obs[0]["tcp_pose"]
        state = np.concatenate([tcp_pose, width], axis=-1)

    state = state.reshape(1, 1, 10)
    return {
        "observation.images.cam_0": torch.from_numpy(image).to(device),
        "observation.states": torch.from_numpy(state).to(device),
        "observation.images.mask_cam_0": torch.zeros(1, 1, 3, 224, 224).to(device)
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CONFIG_PATH = os.path.join(args.model_path, "config.json")
    config = json.load(open(CONFIG_PATH))
    config["dino_model_dir"] = "/home/ubuntu/workspace/dinov3-vits/dinov3-vitb16-pretrain-lvd1689m"
    json.dump(config, open(CONFIG_PATH, "w"))

    policy = MaskACTPolicy.from_pretrained(args.model_path).to(device)

    with open(os.path.join(args.meta_path, 'obs_dict.json'), 'r') as f:
        obs_dict = json.load(f)
    with open(os.path.join(args.meta_path, 'act_dict.json'), 'r') as f:
        act_dict = json.load(f)
    obs_buffer = ObsBuffer(num_obs=1, obs_dict=obs_dict, create=False)
    act_buffer = ActBuffer(num_act=1, act_dict=act_dict, create=False)
    print(f"==========> Initialized (action_mode: {args.action_mode})")

    policy.eval()
    with torch.inference_mode():
        step_idx = 0
        while True:
            while not obs_buffer.getf():
                time.sleep(args.sleep_time)
            obs = obs_buffer.getn()
            obs_buffer.setf(False)
            print("==========> Obs received")

            batch = preprocess_obs(obs, device, args.action_mode)
            action = policy.select_action(batch).cpu().numpy()

            if args.action_mode == 'joint':
                action_dict = {"joints": action[0][:7], "width": action[0][7]}
            else:
                action_dict = {"tcp_pose": action[0][:9], "width": action[0][9]}

            act_buffer.addn([action_dict])
            act_buffer.setf(True)
            print("==========> Act set")
            step_idx += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
