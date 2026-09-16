import torch
import math

import torch.nn as nn

from functools import partial
from torch.nn.init import trunc_normal_
from .utils import get_2d_sincos_pos_embed, apply_masks, repeat_interleave_batch
from .components import Block


class Predictor(nn.Module):
    def __init__(
            self, 
            num_patches, 
            encoder_embed_dim,
            output_dim,
            predictor_embed_dim, 
            depth, 
            num_heads, 
            mlp_ratio=4, 
            qkv_bias=False, 
            qk_scale=None, 
            drop_rate=0., 
            attn_drop_rate=0.,
            drop_path_rate=0.,
            init_std=0.02,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ):
        super().__init__()
        self.embed = nn.Linear(encoder_embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, predictor_embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, predictor_embed_dim), requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, output_dim, bias=True)
        
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, channels, masks_enc, masks_pred):
        assert (masks_enc is not None) and (masks_pred is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]

        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        # -- Batch Size
        B = len(x) // len(masks_enc)
        C = channels
        unmasked_patches = x.size(1) // C

        # -- map from encoder-dim to pedictor-dim
        x = self.embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.pos_embed.repeat(B, C, 1, 1)
        x_pos_embed = apply_masks(x_pos_embed, masks_enc)

        x = x.reshape(x.size(0), C, unmasked_patches, -1) + x_pos_embed
        x = x.reshape(x.size(0), -1, x.shape[-1])

        N_ctxt = x.shape[1]

        # -- concat mask tokens to x
        pos_embs = self.pos_embed.repeat(B, C, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_pred)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_enc))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), pos_embs.size(2), 1)
        # --
        pred_tokens = pred_tokens + pos_embs
        pred_tokens = pred_tokens.reshape(pred_tokens.size(0), -1, pred_tokens.size(-1))
        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

    def _region_positional_embeddings(self, region_indices, region_sizes=None):
        """Average valid fine positional embeddings into one embedding per region."""
        if region_indices.ndim != 4:
            raise ValueError(
                "region indices must have shape [M, B, R, K], "
                f"got {tuple(region_indices.shape)}"
            )
        if torch.any(region_indices < 0) or torch.any(region_indices >= self.pos_embed.size(2)):
            raise ValueError("region indices are outside the predictor positional-embedding grid")
        region_indices = region_indices.to(device=self.pos_embed.device, dtype=torch.long)
        positions = self.pos_embed[0, 0][region_indices]
        if region_sizes is None:
            return positions.mean(dim=3)
        if region_sizes.ndim != 3 or tuple(region_sizes.shape) != tuple(region_indices.shape[:3]):
            raise ValueError("region sizes must have shape [M, B, R] matching region indices")
        region_sizes = region_sizes.to(device=positions.device, dtype=torch.long)
        if torch.any(region_sizes < 1) or torch.any(region_sizes > region_indices.size(-1)):
            raise ValueError("region sizes must be within the padded fine-patch axis")
        valid_members = torch.arange(region_indices.size(-1), device=positions.device).reshape(1, 1, 1, -1)
        valid_members = valid_members < region_sizes.unsqueeze(-1)
        positions = (positions * valid_members.unsqueeze(-1).to(dtype=positions.dtype)).sum(dim=3)
        return positions / region_sizes.unsqueeze(-1).to(dtype=positions.dtype)

    def forward_regions(self, x, channels, masks_enc, region_indices, region_sizes=None):
        """Predict one latent vector per mask-relative P2 target region.

        ``region_indices`` has explicit
        ``[prediction_mask, batch, region, fine_patch]`` axes. Context tokens
        follow the pinned P0 path; only the prediction queries differ, using a
        learned mask token plus the mean positional embedding of each region's
        valid member fine patches.
        """

        if masks_enc is None:
            raise ValueError("cannot run region predictor without encoder-mask indices")
        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]
        if region_indices.ndim != 4:
            raise ValueError(
                "region indices must have shape [M, B, R, K], "
                f"got {tuple(region_indices.shape)}"
            )

        num_context_masks = len(masks_enc)
        if len(x) % num_context_masks:
            raise ValueError("context token batch is not divisible by the number of encoder masks")
        batch_size = len(x) // num_context_masks
        mask_count, region_batch_size, regions, _ = region_indices.shape
        if region_batch_size != batch_size:
            raise ValueError(
                f"region batch size {region_batch_size} does not match context batch size {batch_size}"
            )
        if channels < 1 or x.size(1) % channels:
            raise ValueError(f"context token count {x.size(1)} is incompatible with channels={channels}")

        unmasked_patches = x.size(1) // channels
        x = self.embed(x)
        context_pos_embed = self.pos_embed.repeat(batch_size, channels, 1, 1)
        context_pos_embed = apply_masks(context_pos_embed, masks_enc)
        x = x.reshape(x.size(0), channels, unmasked_patches, -1) + context_pos_embed
        x = x.reshape(x.size(0), -1, x.shape[-1])
        context_tokens = x.size(1)

        region_pos_embed = self._region_positional_embeddings(region_indices, region_sizes).to(device=x.device)
        region_pos_embed = region_pos_embed.unsqueeze(1).unsqueeze(3).expand(
            mask_count,
            num_context_masks,
            batch_size,
            channels,
            regions,
            -1,
        )
        mask_tokens = self.mask_token.reshape(1, 1, 1, 1, 1, -1).expand_as(region_pos_embed)
        prediction_tokens = (mask_tokens + region_pos_embed).reshape(
            mask_count * num_context_masks * batch_size,
            channels * regions,
            -1,
        )
        repeated_context = x.unsqueeze(0).expand(mask_count, -1, -1, -1).reshape(
            mask_count * num_context_masks * batch_size,
            context_tokens,
            -1,
        )
        x = torch.cat([repeated_context, prediction_tokens], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x[:, context_tokens:])
        return x.reshape(
            mask_count,
            num_context_masks,
            batch_size,
            channels,
            regions,
            -1,
        )


class PredictorCLS(Predictor):
    def forward(self, x, channels,masks_enc, masks_pred):
        assert (masks_enc is not None) and (masks_pred is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]

        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        # -- Batch Size
        B = len(x) // len(masks_enc)
        C = channels
        unmasked_patches = (x.size(1) - 1) // C

        # -- map from encoder-dim to pedictor-dim
        x = self.embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.pos_embed.repeat(B, C, 1, 1)
        x_pos_embed = apply_masks(x_pos_embed, masks_enc)


        cls_token = x[:, :1]
        patch_tokens = x[:, 1:]
        patch_tokens = patch_tokens.reshape(patch_tokens.size(0), C, unmasked_patches, -1) + x_pos_embed
        patch_tokens = patch_tokens.reshape(patch_tokens.size(0), -1, patch_tokens.size(-1))
        x = torch.cat([cls_token, patch_tokens], dim=1)

        N_ctxt = x.shape[1]

        # -- concat mask tokens to x
        pos_embs = self.pos_embed.repeat(B, C, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_pred)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_enc))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), pos_embs.size(2), 1)
        # --
        pred_tokens = pred_tokens + pos_embs
        pred_tokens = pred_tokens.reshape(pred_tokens.size(0), -1, pred_tokens.size(-1))
        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = torch.cat([x[:, :1], x[:, N_ctxt:]], dim=1)
        x = self.predictor_proj(x)

        return x
