import torch.nn as nn

from .encoder import Encoder


class LatentWFM(Encoder):
    def __init__(self, num_outputs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.LayerNorm(self.pos_embed.shape[-1]),
            nn.Linear(self.pos_embed.shape[-1], num_outputs)
        )

    def forward(self, x):
        out = super().forward(x)
        return self.head(out.mean(dim=1))

    def freeze_encoder(self, num_blocks: int | None = None):
        if num_blocks is None:
            for p in self.blocks.parameters():
                p.requires_grad = False
        else:
            for p in self.blocks[:num_blocks].parameters():
                p.requires_grad = False

        for p in self.norm.parameters():
            p.requires_grad = False

    def freeze_encoder_lora(self):
        self.freeze_encoder()

        # Unfreeze LoRA layers
        for block in self.blocks:
            for param in block.attn.qkv.lora_q.parameters():
                param.requires_grad = True
            for param in block.attn.qkv.lora_v.parameters():
                param.requires_grad = True

    def freeze_tokenizer(self):
        for p in self.patch_embed.parameters():
            p.requires_grad = False


def get_latent_wfm(num_outputs, img_size, patch_size):    
    return LatentWFM(
        num_outputs=num_outputs,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=256,
        depth=8,
        num_heads=8,
    )