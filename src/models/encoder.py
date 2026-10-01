import torch
import math

import torch.nn as nn

from functools import partial
from torch.nn.init import trunc_normal_
from .utils import get_2d_sincos_pos_embed, apply_masks
from .components import Block, PatchEmbed


class Encoder(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        self.base_grid_size = (
            int(img_size[0]) // self.patch_embed.patch_size,
            int(img_size[1]) // self.patch_embed.patch_size,
        )
        if min(self.base_grid_size) < 1:
            raise ValueError(f"img_size must be at least one patch in each dimension, got {img_size}")
        num_patches = self.base_grid_size[0] * self.base_grid_size[1]
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            self.base_grid_size,
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
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
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    @property
    def logical_depth(self):
        return len(self.blocks)

    @property
    def exit_embed_dim(self):
        return int(self.pos_embed.shape[-1])

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

    def _prepare_tokens(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        if x.ndim != 4:
            raise ValueError(f"expected input with shape [B, C, H, W], got {tuple(x.shape)}")
        height, width = int(x.shape[-2]), int(x.shape[-1])
        patch_size = self.patch_embed.patch_size
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"input spatial dimensions must be divisible by patch_size={patch_size}, got {(height, width)}"
            )
        grid_size = (height // patch_size, width // patch_size)

        # -- patchify x
        x = self.patch_embed(x)

        # -- add positional embedding to x
        if grid_size != self.base_grid_size:
            pos_embed = torch.tensor(
                get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=False),
                device=x.device,
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        x = x.reshape(x.size(0), -1, x.shape[-1])  # (B, C*P, D)

        return x

    def _forward_layers(self, x, masks, return_layers, *, normalize_exits=True, stop_at_last_requested=False):
        requested = tuple(dict.fromkeys(int(layer) for layer in return_layers))
        if not requested:
            return {}
        if any(layer < 1 or layer > len(self.blocks) for layer in requested):
            raise ValueError(f"exit layers must be in [1, {len(self.blocks)}], got {requested}")

        x = self._prepare_tokens(x, masks=masks)
        outputs = {}
        last_requested = max(requested)
        for layer_index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if layer_index in requested:
                # The pinned encoder applies this same final normalization after
                # its last block. Reuse it at optional exits without feeding the
                # normalized tensor into subsequent blocks.
                outputs[layer_index] = self.norm(x) if normalize_exits else x
            # ``forward_to_layer`` is a deployment early exit. Generic
            # ``return_layers`` remains a full-depth tap API for the target
            # branch, whose P2 contract requires full EMA encoding.
            if stop_at_last_requested and layer_index == last_requested:
                break
        return outputs

    def forward_to_layer(self, x, exit_layer):
        """Run full-input inference through one frozen encoder exit.

        ``exit_layer`` is one-indexed: L2 means the first two transformer
        blocks, while L8 is the complete pinned encoder. The output is always
        full-input tokens with the encoder's final normalization applied.
        """
        return self._forward_layers(
            x,
            masks=None,
            return_layers=[exit_layer],
            stop_at_last_requested=True,
        )[int(exit_layer)]

    def forward(self, x, masks=None, return_layers=None, *, normalize_exits=True):
        if return_layers is not None:
            return self._forward_layers(
                x,
                masks=masks,
                return_layers=return_layers,
                normalize_exits=normalize_exits,
            )

        x = self._prepare_tokens(x, masks=masks)

        # -- fwd prop
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


class EncoderCLS(Encoder):
    """ Vision Transformer """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.pos_embed.shape[-1]))
        trunc_normal_(self.cls_token, std=self.init_std)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)

        # -- add positional embedding to x
        if x.shape[2] != self.pos_embed.shape[2]:
            pos_embed = torch.tensor(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(x.shape[2]**.5), cls_token=False), device=x.device, dtype=torch.float32)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        x = x.reshape(x.size(0), -1, x.shape[-1])  # (B, C*P, D)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)

        return x
