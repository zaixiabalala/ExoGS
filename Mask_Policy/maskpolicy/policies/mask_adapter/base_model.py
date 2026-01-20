import torch
import torch.nn.functional as F
from torch import Tensor, nn
from maskpolicy.policies.mask_adapter.transformer import build_label_based_attention_mask


class BaseMaskModel(nn.Module):
    """Base model class with common mask-related functionality."""

    def compute_mask_loss(self, pixel_logits: Tensor, gt_masks: Tensor, num_mask: int) -> Tensor:
        """Compute pixel-level cross-entropy loss for mask prediction.

        Args:
            pixel_logits: (B*T, num_classes, H, W) pixel-level logits
            gt_masks: (B*T, H, W) ground truth mask labels
            num_mask: number of mask classes

        Returns:
            loss: computed cross-entropy loss
        """
        # Compute class weights for imbalanced data
        class_counts = torch.bincount(
            gt_masks.flatten(), minlength=num_mask)
        total_pixels = gt_masks.numel()
        min_count = max(1.0, total_pixels * 0.01)
        class_counts = class_counts.float() + min_count
        class_weights = total_pixels / (num_mask * class_counts)

        # Clip weights to prevent extreme values
        max_weight = 10.0 * (total_pixels / (num_mask * min_count))
        class_weights = torch.clamp(class_weights, min=0.1, max=max_weight)
        class_weights = class_weights.to(pixel_logits.device)

        loss = F.cross_entropy(
            pixel_logits, gt_masks,
            weight=class_weights,
            ignore_index=-1)

        # Clip loss to prevent NaN/Inf
        loss = torch.clamp(loss, min=0.0, max=100.0)
        return loss

    def get_image_features(self, images: Tensor) -> dict[str, Tensor]:
        """Get image features from vision encoder.

        Args:
            images: (B, T, C, H, W) input images

        Returns:
            dict with 'features', 'pos'
        """
        encoder_output = self.vision_encoder(images)
        features = encoder_output["features"]
        pos = encoder_output["pos"]
        return {
            "features": features,
            "pos": pos
        }

    def forward_mask(self, img_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Shared mask prediction logic for all models.

        Args:
            img_dict: dict with 'images', 'pos'

        Returns:
            dict with 'pixel_logits', 'pixel_preds', 'patch_labels'
        """
        features = img_dict["features"]
        B, TN, D = features.shape
        T = getattr(self, 'num_obs', None) or (
            getattr(self.config, 'n_obs_steps', 1) if hasattr(self, 'config') else 1)
        N = TN // T
        H_feat = W_feat = int(N ** 0.5)

        if self.mask_head is None:
            image_size = getattr(self, 'image_size', None) or (getattr(
                self.config, 'image_size', None) if hasattr(self, 'config') else None) or (H_feat * 16)
            num_mask = getattr(self, 'num_mask', None) or (
                getattr(self.config, 'num_mask', 1) if hasattr(self, 'config') else 1)

            pixel_logits = torch.zeros(B, T, num_mask, image_size, image_size,
                                       device=features.device, dtype=features.dtype)
            pixel_preds = torch.zeros(B, T, image_size, image_size,
                                      device=features.device, dtype=torch.long)
            patch_labels = torch.zeros(B, T, N,
                                       device=features.device, dtype=torch.long)

            return {
                'pixel_logits': pixel_logits,
                'pixel_preds': pixel_preds,
                'patch_labels': patch_labels
            }

        features_reshaped = features.reshape(B, T, N, D)
        pixel_logits_list = []
        patch_labels_list = []

        for t in range(T):
            feature_map_t = features_reshaped[:, t, :, :].transpose(
                1, 2).reshape(B, D, H_feat, W_feat)
            patch_labels_t, pixel_logits_t = self.mask_head(
                feature_map_t, return_pixel_pred=True)
            pixel_logits_list.append(pixel_logits_t)
            patch_labels_list.append(patch_labels_t)

        pixel_logits_all = torch.stack(
            pixel_logits_list, dim=1)  # (B, T, num_classes, H, W)
        pixel_preds = torch.argmax(
            pixel_logits_all, dim=2)       # (B, T, H, W)
        patch_labels_all = torch.stack(patch_labels_list, dim=1)  # (B, T, N)

        return {
            'pixel_logits': pixel_logits_all,
            'pixel_preds': pixel_preds,
            'patch_labels': patch_labels_all
        }

    def forward_stage1(self, images, gt_masks, return_predictions=False):
        """Stage 1: Mask prediction only.

        Args:
            images: (B, T, C, H, W) input images
            gt_masks: (B, T, H, W) or (B, H, W) ground truth masks
            return_predictions: If True, return predictions even when gt_masks is provided

        Returns:
            If training and gt_masks provided: loss
            If return_predictions=True: dict with 'pixel_logits', 'pixel_preds', 'patch_labels'
            Otherwise: None
        """
        use_mask = getattr(self, 'use_mask', None) or (
            getattr(self.config, 'use_mask', True) if hasattr(self, 'config') else True)
        if not use_mask:
            raise RuntimeError("Mask prediction is disabled.")

        img_dict = self.get_image_features(images)
        mask_outputs = self.forward_mask(img_dict)

        if gt_masks is not None:
            B, T = mask_outputs['pixel_logits'].shape[:2]
            pixel_logits_flat = mask_outputs['pixel_logits'].reshape(
                B * T, *mask_outputs['pixel_logits'].shape[2:])
            gt_masks_flat = gt_masks.reshape(B * T, *gt_masks.shape[2:])
            loss = self.compute_mask_loss(
                pixel_logits_flat, gt_masks_flat, self.num_mask)

            if return_predictions:
                return {**mask_outputs, 'loss': loss, 'gt_masks': gt_masks}
            return loss

        return mask_outputs if return_predictions else None

    def forward_action(self, img_dict: dict[str, Tensor], mask_outputs: dict, lowdims: Tensor, actions: Tensor = None) -> dict:
        """Abstract method for action prediction logic. Implemented by subclasses.

        Args:
            img_dict: dict with 'features', 'pos'
            mask_outputs: dict from forward_mask()
            lowdims: (B, T, state_dim) state features
            actions: (B, chunk_size, action_dim) actions for training

        Returns:
            dict with action predictions and losses
        """
        raise NotImplementedError

    def compute_joint_loss(self, action_outputs: dict, mask_outputs: dict, gt_masks: Tensor = None) -> dict:
        """Abstract method for computing joint loss. Implemented by subclasses.

        Args:
            action_outputs: dict from forward_action()
            mask_outputs: dict from forward_mask()
            gt_masks: ground truth masks

        Returns:
            dict with 'action_loss' and 'mask_loss'
        """
        raise NotImplementedError

    def format_stage2_output(self, action_outputs: dict, mask_outputs: dict,
                             return_patch_labels: bool) -> dict | Tensor:
        """Format stage2 output. Can be overridden by subclasses."""
        if self.training:
            result = {'action_loss': action_outputs.get(
                'loss'), 'mask_loss': mask_outputs.get('loss', torch.tensor(0., dtype=torch.float32))}
            if return_patch_labels:
                result.update({'patch_labels': mask_outputs['patch_labels'],
                               'pixel_preds': mask_outputs['pixel_preds'],
                               'pixel_logits': mask_outputs['pixel_logits']})
            return result
        else:
            result = action_outputs['action']
            if return_patch_labels:
                result = {'action': result, 'patch_labels': mask_outputs['patch_labels'],
                          'pixel_preds': mask_outputs['pixel_preds'],
                          'pixel_logits': mask_outputs['pixel_logits']}
            return result

    def forward_stage2(self, images, lowdims, actions=None, gt_masks=None, return_patch_labels=False):
        """Stage 2: Joint training with action and mask loss.

        Args:
            images: (B, T, C, H, W) input images
            lowdims: (B, T, state_dim) or (B, state_dim) state features
            actions: (B, chunk_size, action_dim) actions for training
            gt_masks: (B, T, H, W) or (B, H, W) ground truth masks
            return_patch_labels: If True, return patch_labels for visualization

        Returns:
            If training: dict with 'action_loss' and 'mask_loss'
            If inference: action prediction or dict with additional mask predictions
        """
        # Get mask predictions
        img_dict = self.get_image_features(images)
        mask_outputs = self.forward_mask(img_dict)

        # Get action predictions (implemented by subclasses)
        action_outputs = self.forward_action(
            img_dict, mask_outputs, lowdims, actions)

        # Compute joint loss only during training (when actions are provided)
        if actions is not None:
            loss_outputs = self.compute_joint_loss(
                action_outputs, mask_outputs, gt_masks)
        else:
            loss_outputs = {}

        # Format output
        return self.format_stage2_output(action_outputs, {**mask_outputs, **loss_outputs}, return_patch_labels)
