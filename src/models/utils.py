import torch

import torch.nn.functional as F

import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), C (channels), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(1).unsqueeze(-1).repeat(1, x.size(1), 1, x.size(-1))
        all_x += [torch.gather(x, dim=2, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x


@torch.no_grad()
def get_effective_rank(latent_reps):
    singular_values = torch.linalg.svdvals(latent_reps)
    p = singular_values / singular_values.sum()
    p = p[p > 1e-8]
    effective_rank = torch.exp(-torch.sum(p * torch.log(p))).item()
    return effective_rank


def sample_channel_subsets(x, C_prime):
    """
    Args:
        x: (B, C_max, F, T) input tensor
        C_prime: int, number of channels to keep per sample
    Returns:
        x_sub: (B, C_prime, F, T) with per-sample random channel subsets
    """
    B, C_max, F, T = x.shape
    
    noise = torch.rand(B, C_max, device=x.device)
    perms = noise.argsort(dim=1)
    selected = perms[:, :C_prime]
    
    batch_idx = torch.arange(B, device=x.device)[:, None]
    x_sub = x[batch_idx, selected]
    
    return x_sub


def simclr_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.shape[0]
    
    labels = torch.arange(batch_size, device=z1.device)
    masks = torch.eye(batch_size, device=z1.device)

    logits_aa = z1 @ z1.T / temperature
    logits_aa = logits_aa - masks * 1e9
    logits_bb = z2 @ z2.T / temperature
    logits_bb = logits_bb - masks * 1e9
    logits_ab = z1 @ z2.T / temperature
    logits_ba = z2 @ z1.T / temperature

    loss_a = F.cross_entropy(torch.concat([logits_ab, logits_aa], 1), labels)
    loss_b = F.cross_entropy(torch.concat([logits_ba, logits_bb], 1), labels)
    loss = (loss_a + loss_b) / 2

    return loss 
