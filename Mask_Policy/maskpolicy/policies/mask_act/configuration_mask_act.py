#!/usr/bin/env python
"""
MaskACT Configuration.

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""
from dataclasses import dataclass, field

from maskpolicy.configs.policies import PreTrainedConfig
from maskpolicy.configs.types import NormalizationMode
from maskpolicy.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("mask_act")
@dataclass
class MaskACTConfig(PreTrainedConfig):
    """Configuration class for Mask Action Chunking with Transformers."""

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture.
    # Vision backbone.
    # Options: DINOv3 models like "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"
    vision_backbone: str = "dinov3_vitb16"
    freeze_backbone: bool = True  # Whether to freeze the backbone weights during training
    # Directory for local DINOv3 torch.hub repo
    dino_model_dir: str = "dinov3-vitb16"
    use_lora: bool = True
    lora_config: dict = None

    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1

    # VAE for action chunking.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training stages (similar to mask_dp)
    training_stage: str = "stage2"  # "stage1" or "stage2"
    mask_loss_weight: float = 0.3
    action_loss_weight: float = 1.0
    stage2_dino_lr_ratio: float = 0.1  # DINO LR = base_lr * ratio
    stage2_freeze_mask_head: bool = False

    # Mask-related parameters
    use_mask: bool = True  # Whether to use mask prediction components
    num_mask: int = 3  # Number of mask classes
    use_lowdim: bool = True  # Whether to use lowdim/state features
    # Importance weights for mask classes
    mask_importance_weights: list[float] | None = None
    image_size: int = 480  # Target image size for mask prediction
    use_aux_head: bool = False  # Whether to use auxiliary head for mask prediction
    # Cross attention rules: {query_label: [allowed_labels]}
    # Example: {1: [0, 2, 3]} means label=1 can only attend to labels 0,2,3
    label_attention_rules: dict[int, list[int]] | None = field(
        default_factory=lambda: {
            0: [0, 1, 2, -1],        
            1: [0, 1, 2, -1],           
            2: [0, 1, 2, -1],        
            -1: [0, 1, 2, -1],    
        }
    )
    lowdim_label: int = -1  # Label for low-dimensional features (latent/state)

    # Optimization.
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_weight_decay: float = 1e-3

    # Training.
    temporal_ensemble_coeff: float | None = None

    def __post_init__(self):
        """Post-initialization hook."""
        super().__post_init__()
        if self.lora_config is None:
            self.lora_config = {
                'r': 16,
                'lora_alpha': 16,
                'target_modules': ["q_proj", "k_proj", "v_proj"],
                'lora_dropout': 0.1,
                'bias': 'none',
            }
        if self.label_attention_rules is None:
            self.label_attention_rules = {
                0: [0, 1, 2, -1],        
                1: [0, 1, 2, -1],         
                2: [0, 1, 2, -1],      
                -1: [0, 1, 2, -1],   
            }
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        # Action should be future frames, not including current frame
        # [1, 2, 3, ..., chunk_size] for future chunk_size steps
        return list(range(1, self.chunk_size + 1))

    @property
    def reward_delta_indices(self) -> None:
        return None
