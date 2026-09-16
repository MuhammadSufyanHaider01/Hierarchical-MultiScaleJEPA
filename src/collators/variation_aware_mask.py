"""Sample-adaptive, variation-aware multi-block masking.

This collator keeps the existing MultiBlockMask batch contract while changing
only the prediction-block location rule.  The importance map is deliberately
non-learnable and is computed on the full cached channel set before JEPA's
training step performs any channel subsampling.
"""

from __future__ import annotations

from logging import getLogger
from multiprocessing import Value

import torch
import torch.nn.functional as F


logger = getLogger()


class VariationAwareMultiBlockMask(object):
    """Choose prediction blocks from temporal/frequency variation scores."""

    def __init__(
        self,
        input_size,
        patch_size,
        aspect_ratio_type,
        aspect_ratio,
        enc_mask_scale,
        pred_mask_scale,
        num_enc_masks,
        num_pred_masks,
        min_keep,
        allow_overlap,
        alpha,
        exploration_eta,
        epsilon,
    ):
        super().__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        if len(input_size) != 2:
            raise ValueError(f"input_size must contain height and width, got {input_size!r}")
        if aspect_ratio_type != "span_time":
            raise ValueError(
                "variation-aware masking currently requires aspect_ratio_type='span_time'"
            )
        if patch_size < 1:
            raise ValueError("patch_size must be positive")

        self.patch_size = int(patch_size)
        self.height = int(input_size[0]) // self.patch_size
        self.width = int(input_size[1]) // self.patch_size
        if self.height < 1 or self.width < 1:
            raise ValueError("input_size must contain at least one patch in each axis")
        if int(input_size[0]) % self.patch_size or int(input_size[1]) % self.patch_size:
            raise ValueError("input_size must be divisible by patch_size")

        self.aspect_ratio_type = aspect_ratio_type
        self.aspect_ratio = aspect_ratio
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = int(num_enc_masks)
        self.npred = int(num_pred_masks)
        self.min_keep = int(min_keep)
        self.allow_overlap = bool(allow_overlap)
        self.alpha = float(alpha)
        self.exploration_eta = float(exploration_eta)
        self.epsilon = float(epsilon)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= self.exploration_eta <= 1.0:
            raise ValueError("exploration_eta must be in [0, 1]")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.nenc < 1 or self.npred < 1:
            raise ValueError("num_enc_masks and num_pred_masks must be positive")
        if self.min_keep < 1:
            raise ValueError("min_keep must be positive")

        # The collator can be shared by DataLoader workers.
        self._itr_counter = Value("i", -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            value = i.value
        return value

    def _sample_block_size(self, generator, scale):
        """Sample the same span-time geometry convention as MultiBlockMask."""

        random_value = torch.rand(1, generator=generator).item()
        min_scale, max_scale = scale
        mask_scale = min_scale + random_value * (max_scale - min_scale)
        max_keep = int(self.height * self.width * mask_scale)
        width = self.width
        height = max_keep // width
        height = max(1, min(height, self.height))
        return height, width

    def _variation_maps(self, samples):
        """Return temporal and frequency variation maps shaped ``[B, H, W]``."""

        if samples.ndim != 4:
            raise ValueError(f"samples must have shape [B, C, H, W], got {tuple(samples.shape)}")
        samples = samples.to(dtype=torch.float32)
        temporal_difference = torch.zeros_like(samples)
        temporal_difference[..., 1:] = (samples[..., 1:] - samples[..., :-1]).abs()
        frequency_difference = torch.zeros_like(samples)
        frequency_difference[:, :, 1:, :] = (
            samples[:, :, 1:, :] - samples[:, :, :-1, :]
        ).abs()
        return temporal_difference.mean(dim=1), frequency_difference.mean(dim=1)

    def _patch_importance(self, samples):
        """Compute the full-channel variation score on the latent patch grid."""

        expected_hw = (self.height * self.patch_size, self.width * self.patch_size)
        if tuple(samples.shape[-2:]) != expected_hw:
            raise ValueError(
                f"samples have spatial shape {tuple(samples.shape[-2:])}, expected {expected_hw}"
            )
        temporal, frequency = self._variation_maps(samples)
        importance = self.alpha * temporal + (1.0 - self.alpha) * frequency
        return F.avg_pool2d(
            importance.unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(1)

    def _candidate_blocks(self, block_size):
        """Return flattened patch indices for every valid block location."""

        height, width = (int(block_size[0]), int(block_size[1]))
        if height < 1 or width < 1 or height > self.height or width > self.width:
            raise ValueError(f"invalid block size {block_size!r} for grid {self.height}x{self.width}")
        rows = torch.arange(height, dtype=torch.long)
        columns = torch.arange(width, dtype=torch.long)
        blocks = []
        for top in range(self.height - height + 1):
            for left in range(self.width - width + 1):
                block = (rows[:, None] + top) * self.width + (columns[None, :] + left)
                blocks.append(block.reshape(-1))
        return torch.stack(blocks, dim=0)

    def _candidate_scores(self, patch_importance, block_size):
        """Average patch importance over each valid rectangular candidate."""

        if patch_importance.ndim != 3:
            raise ValueError(
                f"patch_importance must have shape [B, H, W], got {tuple(patch_importance.shape)}"
            )
        height, width = (int(block_size[0]), int(block_size[1]))
        if tuple(patch_importance.shape[-2:]) != (self.height, self.width):
            raise ValueError(
                "patch_importance grid does not match the collator grid: "
                f"{tuple(patch_importance.shape[-2:])} != {(self.height, self.width)}"
            )
        if not torch.isfinite(patch_importance).all():
            raise ValueError("patch importance contains non-finite values")
        scores = F.avg_pool2d(
            patch_importance.unsqueeze(1),
            kernel_size=(height, width),
            stride=1,
        ).flatten(1)
        return scores

    def _candidate_probabilities(self, candidate_scores):
        """Convert nonnegative candidate scores to an exploration mixture."""

        if candidate_scores.ndim != 2 or candidate_scores.shape[1] < 1:
            raise ValueError(
                "candidate_scores must have shape [B, candidates] with at least one candidate"
            )
        if not torch.isfinite(candidate_scores).all():
            raise ValueError("candidate scores contain non-finite values")
        if (candidate_scores < 0).any():
            raise ValueError("candidate scores must be nonnegative")
        scores = candidate_scores.to(dtype=torch.float32)
        candidate_count = scores.shape[1]
        adaptive = (scores + self.epsilon) / (scores.sum(dim=1, keepdim=True) + self.epsilon * candidate_count)
        uniform = torch.full_like(adaptive, 1.0 / candidate_count)
        probabilities = (1.0 - self.exploration_eta) * adaptive + self.exploration_eta * uniform
        return probabilities / probabilities.sum(dim=1, keepdim=True)

    @staticmethod
    def _mask_complement(mask, height, width):
        complement = torch.ones((height, width), dtype=torch.int32)
        complement.reshape(-1)[mask] = 0
        return complement

    def _sample_prediction_mask(self, probabilities, candidate_blocks, generator):
        selected = torch.multinomial(
            probabilities,
            num_samples=self.npred,
            replacement=True,
            generator=generator,
        )
        return [candidate_blocks[index] for index in selected.tolist()]

    def _sample_context_mask(self, block_size, generator, acceptable_regions=None):
        """Sample context blocks while preserving MultiBlockMask overlap rules."""

        height, width = block_size

        def constrain_mask(mask, tries=0):
            count = max(int(len(acceptable_regions) - tries), 0)
            for index in range(count):
                mask *= acceptable_regions[index]

        tries = 0
        timeout = original_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, self.height - height + 1, (1,), generator=generator).item()
            left = torch.randint(0, self.width - width + 1, (1,), generator=generator).item()
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + height, left : left + width] = 1
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = original_timeout
                    logger.warning(
                        "Mask generator could not find a valid context mask; "
                        "decreasing acceptable-region constraints [%s]",
                        tries,
                    )
        mask = mask.squeeze()
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + height, left : left + width] = 0
        return mask, mask_complement

    def __call__(self, batch):
        """Create encoder and prediction masks with the standard collator contract."""

        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)
        prediction_size = self._sample_block_size(generator, self.pred_mask_scale)
        encoder_size = self._sample_block_size(generator, self.enc_mask_scale)

        patch_importance = self._patch_importance(collated_batch)
        candidate_scores = self._candidate_scores(patch_importance, prediction_size)
        candidate_probabilities = self._candidate_probabilities(candidate_scores)
        candidate_blocks = self._candidate_blocks(prediction_size)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for batch_index in range(batch_size):
            masks_pred = self._sample_prediction_mask(
                candidate_probabilities[batch_index], candidate_blocks, generator
            )
            masks_complement = [
                self._mask_complement(mask, self.height, self.width) for mask in masks_pred
            ]
            min_keep_pred = min(min_keep_pred, *(len(mask) for mask in masks_pred))
            collated_masks_pred.append(masks_pred)

            acceptable_regions = None if self.allow_overlap else masks_complement
            masks_enc = []
            for _ in range(self.nenc):
                mask, _ = self._sample_context_mask(
                    encoder_size,
                    generator,
                    acceptable_regions=acceptable_regions,
                )
                masks_enc.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_enc)

        collated_masks_pred = [
            [mask[:min_keep_pred] for mask in masks] for masks in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = [
            [mask[:min_keep_enc] for mask in masks] for masks in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
