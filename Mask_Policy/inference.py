import os
import json
import time
import torch
import numpy as np
import torch
import cv2

from r3kit.utils.buffer import ObsBuffer, ActBuffer

from maskpolicy.policies.mask_act.modeling_mask_act import MaskACTPolicy

model_path = ''
sleep_time = 0.01
meta_path = ''

def preprocess_obs(obs, device):
    image = obs[0]["images"]
    joints = obs[0]["joints"]
    width = obs[0]["width"]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image.reshape(1, 3, 224, 224) # (1, 3, 224, 224)

    state = np.concatenate([joints, width], axis=-1)
    state = state.reshape(1, 8) # (1, 8)

    mask = np.zeros((1, 3, 224, 224))

    return {
        "observation.images.cam_0": torch.from_numpy(image).to(device),
        #"observation.images.mask_cam_0": torch.from_numpy(mask).to(device),
        "observation.state": torch.from_numpy(state).to(device),
    }


def main():
    # initialize policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    CONFIG_PATH = os.path.join(model_path, "config.json")
    config = json.load(open(CONFIG_PATH))
    config["dino_model_dir"] = ""
    json.dump(config, open(CONFIG_PATH, "w"))

    policy = MaskACTPolicy.from_pretrained(model_path).to(device)

    # initialize buffers
    with open(os.path.join(meta_path, 'obs_dict.json'), 'r') as f:
        obs_dict = json.load(f)
    with open(os.path.join(meta_path, 'act_dict.json'), 'r') as f:
        act_dict = json.load(f)
    obs_buffer = ObsBuffer(num_obs=1, obs_dict=obs_dict, create=False)
    act_buffer = ActBuffer(num_act=1, act_dict=act_dict, create=False)
    print("==========> Initialized")

    # rollout
    policy.eval()
    with torch.inference_mode():
        while True:
            # get obs
            while not obs_buffer.getf():
                time.sleep(sleep_time)
            obs = obs_buffer.getn()
            obs_buffer.setf(False)
            print("==========> Obs received")

            # preprocess obs
            batch = preprocess_obs(obs, device)

            # get act
            action = policy.select_action(batch).cpu().numpy()
            action[0][7] -= 0.005
            print(action[0][7])
            action_dict = {"joints": action[0][:7], "width": action[0][7]}

            # set act
            act_buffer.addn([action_dict])
            act_buffer.setf(True)
            print("==========> Act set")


if __name__ == '__main__':
    main()
