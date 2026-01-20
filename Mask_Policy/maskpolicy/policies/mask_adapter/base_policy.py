"""Base policy class with common mask-related functionality."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from maskpolicy.constants import ACTION, OBS_IMAGES
from maskpolicy.policies.pretrained import PreTrainedPolicy


class BaseMaskPolicy(PreTrainedPolicy):
    """Base policy class with common mask-related functionality."""

    # These must be overridden by subclasses
    # Using placeholder values to satisfy PreTrainedPolicy's __init_subclass__ check
    config_class = "__abstract__"  # Must be overridden by subclasses
    name = "__abstract__"  # Must be overridden by subclasses

    def _prepare_image_and_mask_sequences(self, batch: dict[str, Tensor]):
        """Return resized image sequence tensor and its paired mask sequence."""
        """ !!! only one image and one mask are supported !!! """

        image_keys = [
            k for k in self.config.image_features if "mask" not in k.lower()]
        if not image_keys:
            raise ValueError("No image features configured")

        image_key = image_keys[0]
        image_tensor = batch[image_key]
        assert image_tensor.dim(
        ) == 5, f"Images must be (B, T, C, H, W), got {image_tensor.shape}"
        assert (image_tensor.shape[-2], image_tensor.shape[-1]) == (self.config.image_size,
                                                                    self.config.image_size), f"Images must be resized to {self.config.image_size}x{self.config.image_size}"

        if not self.config.use_mask:
            return image_tensor, None, []

        mask_key = self._expected_mask_key(image_key)
        if mask_key not in batch:
            raise ValueError(f"Image '{image_key}' requires mask '{mask_key}'")
        mask_tensor = batch[mask_key]
        assert mask_tensor.shape == image_tensor.shape, f"Mask must have the same shape as image, got {mask_tensor.shape} and {image_tensor.shape}"

        return image_tensor, mask_tensor, [mask_key]

    def _expected_mask_key(self, image_key: str) -> str:
        """Derive expected mask key from image key."""
        if "." not in image_key:
            raise ValueError(f"Image key '{image_key}' must contain '.'")
        prefix, suffix = image_key.rsplit(".", 1)
        return f"{prefix}.mask_{suffix}"

    def _prepare_state_sequence(self, batch: dict[str, Tensor], images: Tensor) -> Tensor:
        """Ensure states share the same temporal axis as images."""
        state_keys = [k for k in self.config.input_features if "state" in k]
        n_obs = self.config.n_obs_steps

        if not state_keys:
            # Default state dimension if no encoder provided
            default_state_dim = 8
            if hasattr(self.model, 'lowdims_encoder') and self.model.lowdims_encoder is not None:
                state_dim = self.model.lowdims_encoder[0].in_features
            else:
                state_dim = default_state_dim
            return torch.zeros(images.shape[0], n_obs, state_dim,
                               device=images.device, dtype=images.dtype)

        state_tensors = []
        for key in state_keys:
            if key not in batch:
                raise ValueError(f"State key '{key}' missing from batch")
            state_tensors.append(batch[key].reshape(
                batch[key].shape[0], batch[key].shape[1], -1))

        return torch.cat(state_tensors, dim=-1)

    def _prepare_gt_masks(self, mask_sequence: Tensor | None) -> Tensor | None:
        """Convert mask sequence to integer class labels.

        This function also handles remapping of raw dataset labels to the
        semantic classes expected by the policy. By default we assume the
        raw masks use the following convention:

        - 0: background
        - 1: robot arm
        - 2,3,4,...: object / blocks (potentially multiple instance ids)

        We merge all non-background, non-arm ids into a single "object"
        class so that the final semantic masks have exactly 3 classes:

        - 0: background
        - 1: robot arm
        - 2: object (all blocks)
        """

        if mask_sequence is None:
            return None
        mask_sequence = mask_sequence.float()
        mask_sequence = mask_sequence[:, :, 0, :, :]
        # Convert from stored mask format to integer label ids
        raw_labels = (mask_sequence * 255.0).round().long()

        semantic_masks = torch.zeros_like(raw_labels)

        arm_mask = raw_labels == 1
        object_mask = raw_labels >= 2

        semantic_masks[arm_mask] = 1
        semantic_masks[object_mask] = 2

        return semantic_masks

    @torch.no_grad()
    def predict_action_chunk_and_mask(
        self,
        batch: dict[str, Tensor],
        return_patch_labels: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Predict action chunk and optionally mask predictions.

        Args:
            batch: Input batch with images and states
            return_patch_labels: If True, return mask predictions as well

        Returns:
            If return_patch_labels=False:
                action: (B, chunk_size, action_dim) action tensor
            If return_patch_labels=True:
                tuple of (action, mask_dict) where:
                    - action: (B, chunk_size, action_dim) action tensor
                    - mask_dict: dict containing 'patch_labels', 'pixel_preds', 'pixel_logits', 'action'
        """
        self.eval()
        batch = self.normalize_inputs(batch)
        images, _, _ = self._prepare_image_and_mask_sequences(batch)
        states = self._prepare_state_sequence(batch, images)
        if return_patch_labels:
            result = self.model.forward_stage2(
                images, states, actions=None, gt_masks=None, return_patch_labels=True)
            action_pred = result['action']
            return self.unnormalize_outputs({ACTION: action_pred})[ACTION], result
        else:
            action_pred = self.model.forward_stage2(
                images, states, actions=None)
            return self.unnormalize_outputs({ACTION: action_pred})[ACTION]

    @torch.no_grad()
    def predict_masks_stage1(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict masks for stage1 evaluation/visualization.

        Args:
            batch: Input batch with images and optional gt_masks

        Returns:
            dict with 'pixel_logits', 'pixel_preds', 'patch_labels', and optionally 'gt_masks'
        """
        self.eval()
        batch = self.normalize_inputs(batch)
        images, mask_sequence, _ = self._prepare_image_and_mask_sequences(
            batch)
        gt_masks = self._prepare_gt_masks(mask_sequence)
        return self.model.forward_stage1(images, gt_masks, return_predictions=True)

    @torch.no_grad()
    def get_mask_pred_stage2(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Get mask predictions for stage2 visualization.

        Args:
            batch: Input batch with images and states

        Returns:
            dict containing:
                - 'patch_labels': (B, T*N) patch-level mask labels
                - 'pixel_preds': (B, T, H, W) pixel-level mask predictions
                - 'pixel_logits': (B, T, num_classes, H, W) pixel-level logits
                - 'action': (B, chunk_size, action_dim) predicted actions
        """
        _, mask_dict = self.predict_action_chunk_and_mask(
            batch, return_patch_labels=True)
        return mask_dict
