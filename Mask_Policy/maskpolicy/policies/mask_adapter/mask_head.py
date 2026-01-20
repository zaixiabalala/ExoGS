"""Mask Head: Pixel-level mask prediction head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rates for each ASPP branch.
        in_channels (int): Input feature channels.
        channels (int): Output channels for each branch.
    """

    def __init__(self, dilations, in_channels, channels):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels

        # Create parallel convolutions with different dilations
        self.convs = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                # 1x1 convolution (no dilation)
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, channels,
                                  kernel_size=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                # 3x3 convolution with dilation
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, channels, kernel_size=3,
                                  dilation=dilation, padding=dilation, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True)
                    )
                )

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map of shape (B, in_channels, H, W).

        Returns:
            list[Tensor]: List of ASPP branch outputs, each of shape (B, channels, H, W).
        """
        return [conv(x) for conv in self.convs]


class ASPPHead(nn.Module):
    """Standalone ASPP Head for semantic segmentation.

    This implementation is equivalent to mmsegmentation's ASPPHead but doesn't
    require mmseg dependencies. Based on DeepLabV3.

    Args:
        in_channels (int): Input feature channels.
        channels (int): Intermediate channels (default: 256).
        num_classes (int): Number of output classes.
        dilations (tuple[int]): Dilation rates for ASPP module (default: (1, 6, 12, 18)).
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 256,
        num_classes: int = 4,
        dilations: tuple = (1, 6, 12, 18),
    ):
        super().__init__()
        assert isinstance(dilations, (list, tuple))
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dilations = dilations

        # Image-level pooling branch (global average pooling + 1x1 conv)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # ASPP module with multiple parallel branches
        self.aspp_modules = ASPPModule(dilations, in_channels, channels)

        # Bottleneck to fuse all ASPP outputs
        # Total channels = image_pool (channels) + len(dilations) * channels
        total_channels = (len(dilations) + 1) * channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_channels, channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Classification head
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def _transform_inputs(self, inputs):
        """Transform inputs to single feature map.

        Args:
            inputs: Can be a list/tuple of tensors or a single tensor.

        Returns:
            Tensor: Single feature map of shape (B, in_channels, H, W).
        """
        if isinstance(inputs, (list, tuple)):
            # If multiple feature maps, use the last one (highest resolution)
            return inputs[-1]
        return inputs

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs: Input feature map(s). Can be:
                - Tensor of shape (B, in_channels, H, W)
                - List/Tuple of tensors (uses the last one)

        Returns:
            Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        # Transform inputs to single feature map
        x = self._transform_inputs(inputs)

        # Image-level pooling branch
        # Pool to 1x1, then upsample to match spatial size
        img_pool = self.image_pool(x)
        img_pool = F.interpolate(
            img_pool,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # ASPP parallel branches
        aspp_outs = self.aspp_modules(x)

        # Concatenate all branches: [img_pool] + aspp_outs
        aspp_outs = [img_pool] + aspp_outs
        aspp_outs = torch.cat(aspp_outs, dim=1)

        # Bottleneck fusion
        feats = self.bottleneck(aspp_outs)

        # Classification
        output = self.cls_seg(feats)

        return output


class MaskHead(nn.Module):
    """Pixel-level mask prediction head using standalone ASPPHead."""

    def __init__(
        self,
        hidden_dim=512,
        num_classes=4,
        patch_size=16,
        image_size=480,
        importance_weights=None,
        use_aux_head=False,
    ):
        """Initialize MaskHead.

        Args:
            hidden_dim: Backbone output dimension
            num_classes: Number of mask classes
            patch_size: Patch size for converting pixels to patches
            image_size: Target image size
            importance_weights: Voting weights for classes
            use_aux_head: Whether to use auxiliary head
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_aux_head = use_aux_head

        if importance_weights is None:
            importance_weights = [1.0] * num_classes
        assert len(importance_weights) == num_classes
        self.importance_weights = importance_weights

        self.feat_size = image_size // patch_size

        # Use standalone ASPPHead implementation
        self.head = ASPPHead(
            in_channels=hidden_dim,
            channels=256,
            num_classes=num_classes,
            dilations=(1, 6, 12, 18),
        )

        if self.use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )
        else:
            self.aux_head = None

    def forward(self, feature_map, return_pixel_pred=False, return_aux=False):
        """Forward pass through mask head.

        Args:
            feature_map: (B, D, H_feat, W_feat) input features
            return_pixel_pred: Whether to return pixel predictions
            return_aux: Whether to return auxiliary predictions

        Returns:
            patch_labels: (B, N) patch-level labels
            optionally pixel_logits: (B, num_classes, H, W)
            optionally aux_logits: auxiliary predictions
        """
        B, D, H_feat, W_feat = feature_map.shape

        features = feature_map
        if isinstance(features, dict):
            features = features.get("features", features.get("feature_map"))
        if isinstance(features, (list, tuple)):
            features = features[-1]

        pixel_logits = self.head([features])

        if pixel_logits.shape[-2:] != (self.image_size, self.image_size):
            pixel_logits = F.interpolate(
                pixel_logits,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )

        aux_logits = None
        if self.use_aux_head and return_aux:
            aux_logits = self.aux_head(feature_map)

        patch_labels = self._pixel_to_patch_labels(
            pixel_logits, H_feat, W_feat)

        if return_pixel_pred and return_aux and aux_logits is not None:
            return patch_labels, pixel_logits, aux_logits
        elif return_pixel_pred:
            return patch_labels, pixel_logits
        return patch_labels

    def _pixel_to_patch_labels(self, pixel_logits, H_feat, W_feat):
        """Convert pixel predictions to patch labels using weighted voting."""
        B, num_classes, H, W = pixel_logits.shape

        pixel_logits = torch.clamp(pixel_logits, min=-50.0, max=50.0)
        pixel_probs = F.softmax(pixel_logits, dim=1)

        h_patches, w_patches = H_feat, W_feat
        assert H % h_patches == 0 and W % w_patches == 0
        patch_h, patch_w = H // h_patches, W // w_patches

        pixel_probs = pixel_probs.view(
            B, num_classes, h_patches, patch_h, w_patches, patch_w)
        pixel_probs = pixel_probs.permute(0, 2, 4, 1, 3, 5).contiguous()
        pixel_probs = pixel_probs.view(
            B, h_patches * w_patches, num_classes, patch_h * patch_w)

        patch_weights = pixel_probs.mean(dim=3)
        patch_weights = patch_weights / \
            (patch_weights.sum(dim=2, keepdim=True) + 1e-8)

        importance_weights = torch.tensor(
            self.importance_weights,
            device=pixel_logits.device,
            dtype=patch_weights.dtype
        )
        importance_weights = torch.clamp(
            importance_weights, min=0.01, max=10.0)
        weighted_weights = patch_weights * \
            importance_weights.unsqueeze(0).unsqueeze(0)
        weighted_weights = weighted_weights / \
            (weighted_weights.sum(dim=2, keepdim=True) + 1e-8)

        patch_labels = torch.argmax(weighted_weights, dim=2)
        return patch_labels

    def compute_pixel_loss(self, pixel_logits, gt_masks):
        """Compute pixel-level cross-entropy loss."""
        return F.cross_entropy(pixel_logits, gt_masks.long())
