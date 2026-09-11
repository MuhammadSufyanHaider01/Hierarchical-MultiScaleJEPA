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
