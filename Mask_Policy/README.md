# MaskPolicy

A lightweight Mask Adapter that mitigates domain shift by introducing semantic constraints, thereby guiding learned policies to focus on interaction-relevant features. The Mask adapter combines mask prediction with action prediction using DINOv3 vision encoder and transformer architecture.pipelineThis repository showcases the code for inserting a Mask Adapter into the ACT policy, along with a complete training and deployment pipeline.



## Installation

```bash
conda create -n maskpolicy python=3.10
conda activate maskpolicy
pip install -e .
```

## Quick Start

### 1. Prepare HDF5 Dataset

HDF5 file structure:
```
data/
  demo_000/
    images: (N, H, W, 3) uint8
    tcps: (N, 8) float64
    masks: (N, H, W, 3) uint8  # optional, for mask prediction
```

### 2. Training

```bash
# Basic training
python train.py --config train/train_config/mask_act/try_mask_act.json

# Resume from checkpoint
python train.py --config train/train_config/mask_act/try_mask_act.json --resume
```

### Two-Stage Training

MaskPolicy supports two-stage training:

1. **Stage 1**: Train mask prediction head only
2. **Stage 2**: Joint training of action and mask prediction

See `train/bash/mask_act/train_mask_act.sh` for an example.

## Configuration

### Policy Configuration

Key parameters:
- `type`: `"mask_act"` (required)
- `chunk_size`: Action sequence length to predict
- `n_action_steps`: Number of actions to execute
- `n_obs_steps`: Observation window size
- `vision_backbone`: Vision encoder (e.g., `"dinov3_vitb16"`)
- `use_lora`: Whether to use LoRA fine-tuning
- `use_mask`: Whether to enable mask prediction
- `training_stage`: `"stage1"` (mask only) or `"stage2"` (joint training)

See `train/mask_act_config.md` for complete configuration reference.

### Dataset Configuration

**hdf5_fields**: Define raw fields in HDF5
```json
"hdf5_fields": {
  "images": {
    "hdf5_key": "images",
    "shape": [480, 480, 3],
    "dtype": "uint8"
  }
}
```

**input_fields**: Define model inputs
```json
"input_fields": {
  "observation.images.cam_0": {
    "output_key": "observation.images.cam_0",
    "hdf5_fields": ["images"],
    "shape": [3, 480, 480],
    "dtype": "float32",
    "normalization": "imagenet"
  }
}
```

**output_fields**: Define model outputs
```json
"output_fields": {
  "action": {
    "output_key": "action",
    "hdf5_fields": ["joints"],
    "shape": [100, 8],
    "dtype": "float32",
    "normalization": "minmax"
  }
}
```

## API Usage

### Import Model

```python
from maskpolicy.policies.mask_act.modeling_mask_act import MaskACTPolicy
from maskpolicy.policies.factory import make_policy
```

### Load Pretrained Model

```python
# Direct loading
policy = MaskACTPolicy.from_pretrained("/path/to/pretrained_model")

# Using factory function
from maskpolicy.configs import parser
cfg = parser.parse_config("/path/to/config.json")
policy = make_policy(cfg.policy, ds_meta=dataset.meta, hdf5_config=hdf5_config)
```

### Model Methods

```python
# Predict single action (for deployment)
action = policy.select_action(batch)  # Returns (B, action_dim)

# Predict action sequence
action_chunk = policy.predict_action_chunk(batch)  # Returns (B, chunk_size, action_dim)

# Training forward pass
loss, output_dict = policy.forward(batch)

# Save model
policy.save_pretrained("/path/to/save_directory")

# Reset state
policy.reset()
```

### Input Data Format

```python
batch = {
    "observation.images.cam_0": torch.tensor,  # (B, C, H, W) raw image [0, 1], needed to be / 255
    "observation.state": torch.tensor,         # (B, state_dim) raw state values
    "observation.images.mask_cam_0": torch.tensor,  # (B, C, H, W) Mask (optional)
}
```

**Note**: Input data should be raw values (unnormalized), the model handles normalization automatically.

## Acknowledgments

This project is based on and adapts code from:
- **LeRobot** (https://github.com/huggingface/lerobot)
  - Licensed under Apache License 2.0
  - Copyright (c) HuggingFace Inc.

## License

MIT License

See [LICENSE](LICENSE) file for details.
