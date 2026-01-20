"""Vision Encoder for MaskPolicy."""

import torch
import torch.nn as nn

from maskpolicy.policies.act.dino.config import *
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class Sine2DPositionalEncoding(nn.Module):
    """2D Sinusoidal positional encoding for image features."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, height, width, device):
        """Generate 2D positional embeddings.

        Args:
            height: Height of the feature map
            width: Width of the feature map
            device: Device to place tensors on

        Returns:
            pos_embed: (height, width, hidden_dim) positional embeddings
        """
        y_embed = torch.arange(
            height, device=device).unsqueeze(1).repeat(1, width)
        x_embed = torch.arange(width, device=device).unsqueeze(
            0).repeat(height, 1)

        dim_t = torch.arange(self.hidden_dim // 2,
                             dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (self.hidden_dim // 2))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack(
            [pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack(
            [pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)

        return torch.cat([pos_y, pos_x], dim=2)


class DINOEncoder(nn.Module):
    """DINOv3 encoder using only the last layer."""

    def __init__(
        self,
        model_name: str,
        output_dim: int = 512,
        freeze: bool = True,
        model_dir: str = "dinov3-vits",
        use_lora: bool = False,
        lora_config: dict = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze = freeze
        self.model_dir = model_dir
        self.use_lora = use_lora
        self.lora_config = lora_config

        assert model_name in MODEL_TO_NUM_LAYERS, f"Model name {model_name} not in {MODEL_TO_NUM_LAYERS}"

        try:
            self.dino = AutoModel.from_pretrained(self.model_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 model '{model_name}'. Error: {e}")

        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        if use_lora:
            if self.lora_config is None:
                lora_config = {
                    'r': 16,
                    'lora_alpha': 16,
                    'target_modules': ["q_proj", "k_proj", "v_proj"],
                    'lora_dropout': 0.1,
                    'bias': 'none',
                }
            else:
                lora_config = self.lora_config

            peft_config = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['lora_alpha'],
                target_modules=lora_config['target_modules'],
                lora_dropout=lora_config['lora_dropout'],
                bias=lora_config['bias'],
                use_rslora=True,
            )
            self.dino = get_peft_model(self.dino, peft_config)
            print(f"Applied LoRA to DINOv3 with r={lora_config['r']}")

        self.dino_dim = MODEL_TO_HIDDEN_DIM[self.model_name]
        self.projection = nn.Conv2d(self.dino_dim, output_dim, kernel_size=1)

        self.fc = nn.Module()
        self.fc.in_features = output_dim

    def forward(self, x: torch.Tensor) -> dict:
        """Args: x: (B, C, H, W). Returns: dict with feature_map."""
        B, C, H, W = x.shape

        if self.freeze:
            if self.use_lora:
                outputs = self.dino(pixel_values=x, output_hidden_states=False)
            else:
                with torch.no_grad():
                    outputs = self.dino(
                        pixel_values=x, output_hidden_states=False)
        else:
            outputs = self.dino(pixel_values=x, output_hidden_states=False)

        last_hidden_state = outputs.last_hidden_state
        patch_features = last_hidden_state[:, 5:, :]  # Remove CLS token

        B, num_patches, dino_dim = patch_features.shape
        patch = getattr(self.dino.config, "patch_size", 16)
        H_feat = H // patch
        W_feat = W // patch

        expected_patches = H_feat * W_feat
        if num_patches != expected_patches:
            raise ValueError(
                f"Patch count mismatch: expected {expected_patches} patches (H={H_feat}, W={W_feat}) "
                f"from input size ({H}, {W}) with patch_size={patch}, "
                f"but got {num_patches} patches from DINO output."
            )

        patch_features = patch_features.transpose(
            1, 2).reshape(B, self.dino_dim, H_feat, W_feat)
        feature_map = self.projection(patch_features)

        return {"feature_map": feature_map}

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and not self.use_lora:
            if hasattr(self.dino, 'base_model'):
                self.dino.base_model.eval()
            else:
                self.dino.eval()
        return self


class VisionEncoder(nn.Module):
    """Vision encoder using DINOv3 last layer only."""

    def __init__(
        self,
        hidden_dim=512,
        num_obs=1,
        vision_backbone="dinov3_vitl16",
        freeze_backbone=True,
        dino_model_dir=None,
        use_lora=True,
        lora_config=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_obs = num_obs
        self.vision_backbone = vision_backbone

        if vision_backbone.startswith("dinov3"):
            self.backbone = DINOEncoder(
                model_name=vision_backbone,
                freeze=freeze_backbone,
                output_dim=hidden_dim,
                use_lora=use_lora,
                lora_config=lora_config,
                model_dir=dino_model_dir,
            )
        else:
            raise ValueError(
                f"VisionEncoder only supports DINOv3, got {vision_backbone}")

        self.pos_encoding = Sine2DPositionalEncoding(hidden_dim)

    def forward(self, images, batch_size=None):
        """Args: images: (B*T, C, H, W) or (B, T, C, H, W), batch_size: batch size. Returns: dict with features, pos, padding_mask."""
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.reshape(B * T, C, H, W)
        else:
            B = batch_size if batch_size is not None else images.shape[0]
            T = 1
            _, C, H, W = images.shape

        dino_output = self.backbone(images)
        feature_map = dino_output["feature_map"]

        BT, D, H_feat, W_feat = feature_map.shape
        features = feature_map.flatten(2).transpose(1, 2)
        N = H_feat * W_feat
        features = features.reshape(B, T * N, D)

        pos = self.pos_encoding(H_feat, W_feat, images.device)
        pos = pos.reshape(1, H_feat * W_feat, D)
        pos = pos.repeat(B, T, 1)

        padding_mask = torch.zeros(
            B, features.shape[1], dtype=torch.bool, device=images.device)

        return {
            "features": features,
            "pos": pos,
            "padding_mask": padding_mask,
        }
