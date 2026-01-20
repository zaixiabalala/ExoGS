# CmdExo

Deploy trained policies on Flexiv Rizon robot arm with Xense gripper.

## Requirements

- Flexiv Rizon robot arm
- Xense gripper
- Intel RealSense camera (D415)
- CUDA-enabled GPU

## Installation

```bash
conda create -n cmdexo python=3.10
conda activate cmdexo

git clone git@github.com:dadadadawjb/CmdExo.git
cd CmdExo
pip install -r requirements.txt
cd ..

git clone git@github.com:dadadadawjb/r3kit.git
cd r3kit
pip install -e .[flexiv,xense,rs] --config-settings editable_mode=compat
```

## Usage

### Homing

Move robot to home position before rollout:

```bash
python homing.py
```

### Rollout

Run rollout node (hardware control) and inference node (policy) in separate terminals:

```bash
# Terminal 1: rollout node
python rollout.py --action_mode tcp

# Terminal 2: inference node
python inference.py --model_path /path/to/model --action_mode tcp
```

### Arguments

**rollout.py**
- `--robot_id`: Robot serial number
- `--gripper_id`: Gripper serial number
- `--camera_id`: Camera serial number
- `--action_mode`: `tcp` or `joint`
- `--block`: Enable blocking mode
- `--num_steps`: Number of steps (-1 for infinite)
- `--meta_path`: Path to save observation/action metadata

**inference.py**
- `--model_path`: Path to pretrained model
- `--action_mode`: `tcp` or `joint`
- `--meta_path`: Path to observation/action metadata

## Acknowledgement

This project uses [r3kit](https://github.com/dadadadawjb/r3kit) for robot control.
