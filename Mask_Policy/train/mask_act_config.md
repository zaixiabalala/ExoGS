# MaskACT Configuration

Complete configuration reference for MaskACT (Mask Action Chunking with Transformers).

## Basic Parameters

### Input/Output Structure

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_obs_steps` | int | 1 | Observation window size (currently only supports 1) |
| `chunk_size` | int | 100 | Action sequence length to predict |
| `n_action_steps` | int | 100 | Number of actions to execute (must ≤ chunk_size) |

### Training Stages

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_stage` | str | "stage2" | Training stage: "stage1" (mask only) or "stage2" (joint training) |
| `mask_loss_weight` | float | 0.3 | Mask loss weight in stage2 |
| `action_loss_weight` | float | 1.0 | Action loss weight in stage2 |
| `stage2_dino_lr_ratio` | float | 0.1 | DINO encoder learning rate ratio in stage2 |
| `stage2_freeze_mask_head` | bool | False | Whether to freeze mask head in stage2 |

### Normalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalization_mapping` | dict | `{"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}` | Normalization mapping, supports `MEAN_STD`, `MIN_MAX`, `IDENTITY` |

## Architecture Parameters

### Vision Encoder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_backbone` | str | "dinov3_vitb16" | Vision encoder type, supports DINOv3 models: `"dinov3_vits16"`, `"dinov3_vitb16"`, `"dinov3_vitl16"` |
| `freeze_backbone` | bool | True | Whether to freeze vision encoder weights |
| `dino_model_dir` | str | "dinov3-vitb16" | Local DINOv3 model directory path |
| `use_lora` | bool | True | Whether to use LoRA fine-tuning |
| `lora_config` | dict | None | LoRA config dict: `r`, `lora_alpha`, `target_modules`, `lora_dropout`, `bias` |

### Mask Prediction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mask` | bool | True | Whether to enable mask prediction |
| `num_mask` | int | 4 | Number of mask classes |
| `mask_importance_weights` | list[float] | None | Mask class importance weights |
| `image_size` | int | 480 | Target image size for mask prediction |
| `use_aux_head` | bool | False | Whether to use auxiliary head for training |
| `label_attention_rules` | dict | None | Label-based attention rules dict |

### Transformer Layers

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pre_norm` | bool | False | Whether to use pre-norm architecture |
| `dim_model` | int | 512 | Transformer model dimension |
| `n_heads` | int | 8 | Number of attention heads |
| `dim_feedforward` | int | 3200 | Feedforward network dimension |
| `feedforward_activation` | str | "relu" | Feedforward activation function |
| `n_encoder_layers` | int | 4 | Number of encoder layers |
| `n_decoder_layers` | int | 1 | Number of decoder layers |

### VAE

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_vae` | bool | True | Whether to use VAE for action encoding |
| `latent_dim` | int | 32 | Latent space dimension |
| `n_vae_encoder_layers` | int | 4 | VAE encoder layers |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_lr` | float | 1e-4 | Main model learning rate |
| `optimizer_lr_backbone` | float | 1e-5 | Vision encoder learning rate |
| `optimizer_betas` | tuple[float, float] | [0.9, 0.999] | Adam optimizer beta parameters |
| `optimizer_weight_decay` | float | 1e-3 | Weight decay coefficient |
| `temporal_ensemble_coeff` | float | None | Temporal ensemble coefficient (None to disable) |

## Configuration Examples

### Stage1: Mask Prediction Only

```json
{
  "type": "mask_act",
  "training_stage": "stage1",
  "n_obs_steps": 1,
  "vision_backbone": "dinov3_vitb16",
  "freeze_backbone": false,
  "use_lora": true,
  "use_mask": true,
  "num_mask": 4,
  "image_size": 480,
  "dim_model": 512,
  "optimizer_lr": 0.0001,
  "optimizer_lr_backbone": 0.00001
}
```

### Stage2: Joint Training

```json
{
  "type": "mask_act",
  "training_stage": "stage2",
  "n_obs_steps": 1,
  "chunk_size": 100,
  "n_action_steps": 100,
  "vision_backbone": "dinov3_vitb16",
  "freeze_backbone": true,
  "use_lora": true,
  "use_mask": true,
  "num_mask": 4,
  "image_size": 480,
  "use_vae": true,
  "latent_dim": 32,
  "dim_model": 512,
  "n_heads": 8,
  "n_encoder_layers": 4,
  "mask_loss_weight": 0.3,
  "action_loss_weight": 1.0,
  "stage2_dino_lr_ratio": 0.1,
  "stage2_freeze_mask_head": false,
  "optimizer_lr": 0.0001,
  "optimizer_lr_backbone": 0.00001
}
```

### Action-Only Mode (Mask Disabled)

```json
{
  "type": "mask_act",
  "training_stage": "stage2",
  "n_obs_steps": 1,
  "chunk_size": 100,
  "n_action_steps": 100,
  "vision_backbone": "dinov3_vitb16",
  "freeze_backbone": true,
  "use_lora": true,
  "use_mask": false,
  "use_vae": true,
  "latent_dim": 32,
  "dim_model": 512,
  "n_heads": 8,
  "n_encoder_layers": 4,
  "optimizer_lr": 0.0001,
  "optimizer_lr_backbone": 0.00001
}
```

## Notes

1. **use_mask**: When set to `false`, the model will not perform mask prediction, only using ACT action prediction
2. **num_mask**: Typically set to 3 or 4, corresponding to: background, robot arm, single object, or background, robot arm, multiple object instances
3. **image_size**: Must match the image size of training data
4. **label_attention_rules**: Defines attention interaction rules between mask classes, e.g., `{"0": ["1"], "1": ["0", "2"]}` means class 0 only attends to class 1, class 1 attends to classes 0 and 2
