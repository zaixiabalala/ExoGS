#!/usr/bin/env python
"""
Mask Action Chunking Transformer Policy.

This implementation is based on code from LeRobot (https://github.com/huggingface/lerobot).
LeRobot is licensed under Apache License 2.0, Copyright (c) HuggingFace Inc.
"""

from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from maskpolicy.constants import ACTION, OBS_IMAGES
from maskpolicy.configs.types import NormalizationMode
from maskpolicy.policies.mask_act.configuration_mask_act import MaskACTConfig
from maskpolicy.policies.mask_adapter.base_policy import BaseMaskPolicy
from maskpolicy.policies.mask_adapter.vision_encoder import VisionEncoder
from maskpolicy.policies.mask_adapter.vision_encoder import Sine2DPositionalEncoding
from maskpolicy.policies.mask_adapter.mask_head import MaskHead
from maskpolicy.policies.mask_adapter.base_model import BaseMaskModel
from maskpolicy.policies.normalize import create_normalization_modules
from maskpolicy.policies.mask_adapter.transformer import (
    Transformer, 
    build_label_based_attention_mask,
    TransformerEncoder,
    TransformerEncoderLayer
)


class MaskACTPolicy(BaseMaskPolicy):
    """Mask Action Chunking with Transformers Policy."""

    config_class = MaskACTConfig
    name = "mask_act"

    def __init__(
        self,
        config: MaskACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        ds_meta=None,
        field_norm_map: dict[str, "NormalizationMode"] | None = None,
    ):
        """Initialize MaskACT policy.

        Args:
            config: Policy configuration.
            dataset_stats: Dataset statistics for normalization.
            ds_meta: Dataset metadata.
            field_norm_map: Optional field-level normalization mapping.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs, self.normalize_targets, self.unnormalize_outputs = create_normalization_modules(
            config.input_features,
            config.output_features,
            config.normalization_mapping,
            dataset_stats,
            ds_meta=ds_meta,
            field_norm_map=field_norm_map
        )

        self.model = MaskACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = MaskACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> list:
        """Get optimizer parameter groups with different learning rates."""
        if self.config.training_stage == "stage2" and self.config.vision_backbone.startswith("dinov3"):
            base_lr = self.config.optimizer_lr
            dino_lr = base_lr * self.config.stage2_dino_lr_ratio

            params = []

            # DINO backbone (vision encoder)
            dino_params = [
                p for n, p in self.named_parameters()
                if "model.vision_encoder" in n and p.requires_grad
            ]
            if dino_params:
                params.append({"params": dino_params, "lr": dino_lr})

            # Mask head (stage2 only)
            if self.config.training_stage == "stage2":
                mask_head_params = [
                    p for n, p in self.named_parameters()
                    if "model.mask_head" in n and p.requires_grad
                ]
                if mask_head_params:
                    mask_lr = 0 if self.config.stage2_freeze_mask_head else 0.1 * base_lr
                    params.append({"params": mask_head_params, "lr": mask_lr})

            # Other parameters
            other_params = [
                p for n, p in self.named_parameters()
                if not ("model.vision_encoder" in n or "model.mask_head" in n) and p.requires_grad
            ]
            if other_params:
                params.append({"params": other_params})

            return params if params else [{"params": [p for p in self.parameters() if p.requires_grad]}]
        else:
            return [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not n.startswith("model.vision_encoder") and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if n.startswith("model.vision_encoder") and p.requires_grad
                    ],
                    "lr": self.config.optimizer_lr_backbone,
                },
            ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._last_mask_dict = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Forward pass during training."""
        assert self.config.n_obs_steps == 1, "MaskACT only supports single observation step"
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, mask_sequence, _ = self._prepare_image_and_mask_sequences(
            batch)
        states = self._prepare_state_sequence(batch, images)
        gt_masks = self._prepare_gt_masks(mask_sequence)

        if self.config.training_stage == "stage1":
            if gt_masks is None:
                raise ValueError("Stage1 training requires GT masks")
            loss = self.model.forward_stage1(images, gt_masks)
            return loss, {'mask_loss': loss.item(), 'action_loss': 0.0}
        else:  # stage2
            if ACTION not in batch:
                raise ValueError(
                    f"Stage2 training requires actions, but '{ACTION}' not found")
            loss_dict = self.model.forward_stage2(
                images, states, batch[ACTION], gt_masks)
            action_loss = loss_dict['action_loss']
            mask_loss = loss_dict['mask_loss']
            total_loss = (self.config.action_loss_weight * action_loss +
                          self.config.mask_loss_weight * mask_loss)
            return total_loss, {'mask_loss': mask_loss.item(), 'action_loss': action_loss.item()}

    @torch.no_grad()
    def select_action_and_mask(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Select action and mask for inference.

        Args:
            batch: Input batch with images and states

        Returns:
            tuple of (action, mask_dict) where:
                - action: (action_dim,) single action tensor
                - mask_dict: dict containing 'patch_labels', 'pixel_preds', 'pixel_logits', 'action'
        """
        self.eval()
        # NOTE: Do not normalize here. `BaseMaskPolicy.predict_action_chunk_and_mask()`
        # already calls `self.normalize_inputs(batch)`. Normalizing twice will break inference.
        if len(self._action_queue) == 0:
            actions, mask_dict = self.predict_action_chunk_and_mask(
                batch, return_patch_labels=True)
            actions = actions[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
            self._last_mask_dict = mask_dict  # 缓存 mask_dict
        return self._action_queue.popleft(), self._last_mask_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action for inference."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk_and_mask(
                batch, return_patch_labels=False)
            actions = actions[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk for a given observation."""
        return self.predict_action_chunk_and_mask(batch, return_patch_labels=False)


class MaskACTTemporalEnsembler:
    """Temporal ensembling with exponential weights wᵢ = exp(-coeff * i)."""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(
            -temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(
            self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Reset online computation variables."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """Update temporal ensemble and return first action."""
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(
            device=actions.device)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions * self.ensemble_weights
            self.ensembled_actions_count = self.ensemble_weights.clone()
        else:
            self.ensembled_actions = (
                self.ensembled_actions + actions * self.ensemble_weights)
            self.ensembled_actions_count += self.ensemble_weights

        ensembled_actions = self.ensembled_actions / self.ensembled_actions_count
        return ensembled_actions[:, 0]


class MaskACT(BaseMaskModel):
    """Mask Action Chunking Transformer: Combines mask prediction with action prediction.

    Architecture:
    MaskEncoder + Sine2DPositionalEncoding + MaskHead + CrossAttentionTransformer
    """

    def __init__(self, config: MaskACTConfig):
        super().__init__()
        self.config = config
        self.num_mask = config.num_mask

        self.vision_encoder = VisionEncoder(
            hidden_dim=config.dim_model,
            num_obs=config.n_obs_steps,
            vision_backbone=config.vision_backbone,
            freeze_backbone=config.freeze_backbone,
            dino_model_dir=config.dino_model_dir,
            use_lora=config.use_lora,
            lora_config=config.lora_config,
        )
        patch_size = 16 if config.vision_backbone.startswith("dinov3") else 32

        if self.config.use_mask:
            self.mask_head = MaskHead(
                hidden_dim=config.dim_model,
                num_classes=config.num_mask,
                patch_size=patch_size,
                image_size=config.image_size,
                importance_weights=config.mask_importance_weights,
                use_aux_head=config.use_aux_head,
            )
        else:
            self.mask_head = None

        state_dim = sum(
            ft.shape[0] for key, ft in config.input_features.items()
            if "observation.state" in key or "observation.environment_state" in key
        )

        if ACTION in config.output_features:
            action_shape = config.output_features[ACTION].shape
            action_dim = action_shape[-1]
        else:
            action_dim = state_dim

        # Modality: 0: image, 1: lowdim/state (if use_lowdim), 2: latent
        self.num_modality = 2 + (1 if config.use_lowdim else 0)
        self.modality_embed = nn.Embedding(self.num_modality, config.dim_model)
        nn.init.normal_(self.modality_embed.weight, mean=0.0, std=0.02)

        if self.config.use_mask:
            self.label_embed = nn.Embedding(config.num_mask, config.dim_model)
            nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)
        else:
            self.label_embed = None

        # Lowdim label for attention masking
        self.lowdim_label = getattr(config, 'lowdim_label', -1)

        # Action prediction components
        if self.config.use_vae:
            # Use TransformerEncoder from transformer.py
            vae_encoder_layer = TransformerEncoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.feedforward_activation,
                normalize_before=config.pre_norm
            )
            vae_encoder_norm = nn.LayerNorm(config.dim_model) if config.pre_norm else None
            self.vae_encoder = TransformerEncoder(
                encoder_layer=vae_encoder_layer,
                num_layers=config.n_vae_encoder_layers,
                norm=vae_encoder_norm
            )
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature and self.config.use_lowdim:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    state_dim, config.dim_model)
            else:
                self.vae_encoder_robot_state_input_proj = None
            self.vae_encoder_action_input_proj = nn.Linear(
                action_dim, config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(
                config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size + \
                (1 if (self.config.robot_state_feature and self.config.use_lowdim) else 0)
            self.register_buffer("vae_encoder_pos_enc", create_sinusoidal_pos_embedding(
                num_input_token_encoder, config.dim_model).unsqueeze(0))

        self.transformer = Transformer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_encoder_layers,
            num_decoder_layers=config.n_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            no_pos=False,  # Use positional encoding
            activation=config.feedforward_activation,
            normalize_before=config.pre_norm,
            return_intermediate_dec=False
        )

        # Encoder input projections
        self.encoder_latent_input_proj = nn.Linear(
            config.latent_dim, config.dim_model)
        if self.config.robot_state_feature and self.config.use_lowdim:
            self.encoder_robot_state_input_proj = nn.Linear(
                state_dim, config.dim_model)
        else:
            self.encoder_robot_state_input_proj = None

        self.readout_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of transformer parameters."""
        # Initialize transformer parameters (already initialized in Transformer.__init__)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _remove_time_dim(self, tensor: Tensor) -> Tensor:
        """Remove time dimension from tensor."""
        tensor = tensor.squeeze(1)
        return tensor

    def forward_action(self, img_dict: dict[str, Tensor], mask_outputs: dict, lowdims: Tensor, actions: Tensor = None) -> dict:
        """Transformer-based action prediction for MaskACT (aligned with ACT forward logic)."""
        B = img_dict["features"].shape[0] if "features" in img_dict else lowdims.shape[0]

        # VAE latent sampling (training mode with actions)
        if self.config.use_vae and actions is not None and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=B)

            robot_state_embed = None
            if self.config.robot_state_feature and self.config.use_lowdim and self.vae_encoder_robot_state_input_proj is not None:
                state = lowdims
                if state.dim() == 3:
                    state = state[:, -1]
                robot_state_embed = self.vae_encoder_robot_state_input_proj(
                    state).unsqueeze(1)

            action = actions
            if len(action.shape) == 2:
                action = action.view(
                    action.shape[0] // self.config.chunk_size, self.config.chunk_size, -1)
            action_embed = self.vae_encoder_action_input_proj(action)

            vae_encoder_input_list = [cls_embed]
            if robot_state_embed is not None:
                vae_encoder_input_list.append(robot_state_embed)
            vae_encoder_input_list.append(action_embed)
            vae_encoder_input = torch.cat(vae_encoder_input_list, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            action_is_pad = torch.zeros(
                B, self.config.chunk_size, dtype=torch.bool, device=lowdims.device)
            num_prefix_tokens = 1 + (1 if robot_state_embed is not None else 0)
            key_padding_mask = torch.cat([
                torch.full((B, num_prefix_tokens),
                           False, device=lowdims.device),
                action_is_pad
            ], axis=1)

            # TransformerEncoder expects (seq_len, batch, dim) format
            vae_encoder_input_seq = vae_encoder_input.permute(1, 0, 2)  # (seq_len, B, D)
            vae_pos_embed_seq = pos_embed.permute(1, 0, 2)  # (seq_len, B, D)
            
            # TransformerEncoder returns (output, attn_maps), we only need output
            vae_encoder_output, _ = self.vae_encoder(
                src=vae_encoder_input_seq,
                pos=vae_pos_embed_seq,
                src_key_padding_mask=key_padding_mask,
                need_weights=False
            )
            # Extract CLS token (first token)
            cls_token_out = vae_encoder_output[0]  # (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(
                cls_token_out)
            mu = latent_pdf_params[:, :self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]
            latent_sample = mu + \
                log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # Inference mode or no VAE
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(B, self.config.latent_dim,
                                        dtype=torch.float32, device=lowdims.device)

        # Build encoder inputs - follow ACT order: [latent, state, images]
        # Format: (B, L, D) for unified Transformer
        encoder_in_tokens = []
        encoder_in_pos_embed = []

        # Latent token
        latent_embed = self.encoder_latent_input_proj(latent_sample)  # (B, D)
        encoder_in_tokens.append(latent_embed)
        latent_modality_idx = 1 if not self.config.use_lowdim else 2
        latent_modality_emb = self.modality_embed(torch.tensor(
            latent_modality_idx, device=lowdims.device))
        latent_pos = latent_modality_emb.unsqueeze(0).expand(B, -1)  # (B, D)
        encoder_in_pos_embed.append(latent_pos)

        # State token 
        if self.config.robot_state_feature and self.config.use_lowdim and self.encoder_robot_state_input_proj is not None:
            state = lowdims
            if state.dim() == 3:
                state = state[:, -1]
            state_embed = self.encoder_robot_state_input_proj(state)  # (B, D)
            encoder_in_tokens.append(state_embed)
            state_modality_emb = self.modality_embed(torch.tensor(
                1, device=lowdims.device))
            state_pos = state_modality_emb.unsqueeze(0).expand(B, -1)  # (B, D)
            encoder_in_pos_embed.append(state_pos)

        # Get image features
        img_features = img_dict["features"]  # (B, N, D)
        img_pos = img_dict["pos"]  # (B, N, D)
        img_modality_emb = self.modality_embed(torch.tensor(
            0, device=lowdims.device))
        img_pos = img_pos + img_modality_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, D)
        B = img_features.shape[0]
        N = img_features.shape[1]
        T = getattr(self.config, 'n_obs_steps', 1)

        # Add label embedding to image pos if using mask
        if self.config.use_mask and self.label_embed is not None and mask_outputs is not None:
            patch_labels_all = mask_outputs['patch_labels'].reshape(B, T * N)
            label_emb = self.label_embed(patch_labels_all.long())  # (B, N, D)
            img_pos = img_pos + label_emb

        # Concatenate all tokens: (B, L, D) format
        # Order: [latent, state(optional), image_patches...]
        if encoder_in_tokens:
            prefix_tokens = torch.stack(encoder_in_tokens, dim=1)  # (B, num_prefix, D)
            prefix_pos = torch.stack(encoder_in_pos_embed, dim=1)  # (B, num_prefix, D)
            src = torch.cat([prefix_tokens, img_features], dim=1)  # (B, L, D)
            pos_concat = torch.cat([prefix_pos, img_pos], dim=1)  # (B, L, D)
        else:
            src = img_features
            pos_concat = img_pos

        # Build label-based attention mask if using mask
        label_attn_mask = None
        if self.config.use_mask and self.config.label_attention_rules and mask_outputs is not None:
            patch_labels_all = mask_outputs['patch_labels'].reshape(B, T * N)
            num_prefix = len(encoder_in_tokens)
            
            # Build complete label sequence: [latent_label, state_label(optional), image_labels...]
            label_seq_list = []
            label_seq_list.append(torch.full((B, 1), self.lowdim_label,
                                             device=patch_labels_all.device,
                                             dtype=patch_labels_all.dtype))
            if self.config.robot_state_feature and self.config.use_lowdim and self.encoder_robot_state_input_proj is not None:
                label_seq_list.append(torch.full((B, 1), self.lowdim_label,
                                                device=patch_labels_all.device,
                                                dtype=patch_labels_all.dtype))
            label_seq_list.append(patch_labels_all)
            
            label_seq = torch.cat(label_seq_list, dim=1)  # (B, L)
            
            label_attn_mask = build_label_based_attention_mask(
                label_seq, self.config.label_attention_rules)

        padding_mask = torch.zeros(B, src.shape[1], dtype=torch.bool, device=src.device)
        query_embed = self.readout_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, chunk_size, D)
        
        if label_attn_mask is not None:
            hs, _ = self.transformer(
                src, padding_mask, query_embed, pos_concat,
                src_mask=label_attn_mask
            )
        else:
            hs, _ = self.transformer(
                src, padding_mask, query_embed, pos_concat
            )

        action_pred = self.action_head(hs)  # (B, chunk_size, action_dim)

        if actions is not None:
            # Training mode - compute loss
            action_is_pad = torch.zeros(
                B, self.config.chunk_size, dtype=torch.bool, device=actions.device)

            l1_loss = (F.l1_loss(
                actions, action_pred, reduction="none") * ~action_is_pad.unsqueeze(-1)).mean()

            if self.config.use_vae and mu is not None and log_sigma_x2 is not None:
                mean_kld = (-0.5 * (1 + log_sigma_x2 -
                            mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
                total_loss = l1_loss + mean_kld * self.config.kl_weight
            else:
                total_loss = l1_loss

            return {'loss': total_loss}
        else:
            # Inference mode
            return {'action': action_pred}

    def compute_joint_loss(self, action_outputs: dict, mask_outputs: dict, gt_masks: Tensor = None) -> dict:
        """Compute MaskACT joint loss."""
        losses = {}

        if 'loss' in action_outputs:
            losses['action_loss'] = action_outputs['loss']
        else:
            raise ValueError("Action outputs must contain 'loss' key")

        if gt_masks is not None and hasattr(self, 'config') and self.config.use_mask:
            B, T = mask_outputs['pixel_logits'].shape[:2]
            pixel_logits_flat = mask_outputs['pixel_logits'].reshape(
                B * T, *mask_outputs['pixel_logits'].shape[2:])
            gt_masks_flat = gt_masks.reshape(B * T, *gt_masks.shape[2:])
            mask_loss = self.compute_mask_loss(
                pixel_logits_flat, gt_masks_flat, self.config.num_mask)
            losses['loss'] = mask_loss

        return losses

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()
