import torch
import copy
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from configs import JEPAConfig
from models import Encoder, Predictor
from .utils import apply_masks, repeat_interleave_batch, get_effective_rank, sample_channel_subsets
from .validation import ValidationModule
from schedulers import WarmupCosineSchedule, CosineWDSchedule


class JEPA(ValidationModule):
    def __init__(self, config: JEPAConfig, iterations_per_epoch: int, eval_fns: list[callable], metrics: list[str]):
        super().__init__(eval_fns, metrics)
        CONFIG = config
        self.context_encoder = Encoder(
            img_size=CONFIG.encoder.img_size,
            patch_size=CONFIG.encoder.patch_size,
            embed_dim=CONFIG.encoder.embed_dim,
            depth=CONFIG.encoder.depth,
            num_heads=CONFIG.encoder.num_heads,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = Predictor(
            num_patches=(CONFIG.encoder.img_size[0] // CONFIG.encoder.patch_size) ** 2,
            encoder_embed_dim=CONFIG.encoder.embed_dim,
            output_dim=CONFIG.encoder.embed_dim,
            predictor_embed_dim=CONFIG.predictor.embed_dim,
            depth=CONFIG.predictor.depth,
            num_heads=CONFIG.predictor.num_heads,
        )
        self.loss = nn.MSELoss()
        self.iterations_per_epoch = iterations_per_epoch

        self.ema_start = CONFIG.optimization.ema[0]
        self.ema_end = CONFIG.optimization.ema[1]
        self.total_steps = iterations_per_epoch * CONFIG.optimization.epochs
        self.num_patches = (CONFIG.encoder.img_size[0] // CONFIG.encoder.patch_size) ** 2

        self._freeze_target_encoder()
        self.config = CONFIG

    def _freeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, masks_enc, masks_pred):
        C = x.size(1)

        context_emb = self.context_encoder(x, masks_enc)
        latent_mask_pred = self.predictor(context_emb, C, masks_enc, masks_pred)

        with torch.no_grad():
            target_out = self.target_encoder(x)
            latent_mask_target = F.layer_norm(target_out, (target_out.size(-1),))
            B, _, D = latent_mask_target.shape

            latent_mask_target = latent_mask_target.reshape(B, C, -1, D)
            latent_mask_target = apply_masks(latent_mask_target, masks_pred)
            latent_mask_target = repeat_interleave_batch(latent_mask_target, B=B, repeat=len(masks_enc))
            latent_mask_target = latent_mask_target.reshape(latent_mask_target.size(0), -1, D)

        return latent_mask_pred, latent_mask_target, target_out

    def training_step(self, batches, _):
        losses = []
        mask_percentages = []
        latent_reps = []
        for batch_name, batch in batches.items():
            loss, mask_percentage, rep = self.training_step_per_patch(batch_name, batch)
            losses.append(loss)
            mask_percentages.append(mask_percentage)
            latent_reps.append(rep)

        total_loss = torch.stack(losses).mean()
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True)

        self.log(f"mask_percentage", np.mean(mask_percentages), on_step=True, on_epoch=True, logger=True)
    
        latent_reps = torch.cat(latent_reps, dim=0)
        effective_rank = get_effective_rank(latent_reps)
        self.log(f"effective_rank", effective_rank, on_step=True, on_epoch=True, logger=True)

        with torch.no_grad():
            per_dim_std = latent_reps.std(dim=0)
            mean_std = per_dim_std.mean().item()
            self.log(f"mean_std", mean_std, on_step=True, on_epoch=True, logger=True)

        return total_loss
        
    def training_step_per_patch(self, name, batch):
        x, masks_enc, masks_pred = batch
        C_max = x.size(1)
        if C_max > 1:
            C_prime = torch.randint(2, C_max + 1, (1,)).item()
            x = sample_channel_subsets(x, C_prime)
            self.log(f"num_channels_{name}", C_prime, on_step=True, on_epoch=True, logger=True)

        latent_mask_pred, latent_mask_target, target_out = self(x, masks_enc, masks_pred)
        
        loss = self.loss(latent_mask_pred, latent_mask_target)
        self.log(f'train_loss_{name}', loss, on_step=True, on_epoch=True, logger=True)

        mask_percentage = 1 - len(masks_enc[0][0]) / self.num_patches

        latent_rep = target_out.mean(dim=1).detach()
        
        return loss, mask_percentage, latent_rep
    
    def on_train_batch_start(self, batch, batch_idx):
        wd = self.optimizers().param_groups[0]['weight_decay']
        self.log('weight_decay', wd, on_step=True, on_epoch=False, logger=True) 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            m = self.ema_start + (self.global_step - 1) * (self.ema_end - self.ema_start) / self.total_steps
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data) 
            
        self.log("ema_momentum", m, on_step=True, on_epoch=False, logger=True)

    def _get_validation_tokens_reps(self, x):
        return self.target_encoder(x)

    def _pool_validation_reps(self, token_latent_reps):
        return token_latent_reps.mean(dim=1)

    def configure_optimizers(self):
        CONFIG = self.config
        param_groups = [
            {
                'params': (p for n, p in self.context_encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.context_encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=CONFIG.optimization.start_lr, weight_decay=CONFIG.optimization.ref_weight_decay)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(CONFIG.optimization.warmup * self.iterations_per_epoch),
            start_lr=CONFIG.optimization.start_lr,
            ref_lr=CONFIG.optimization.ref_lr,
            final_lr=CONFIG.optimization.final_lr,
            T_max=int(CONFIG.optimization.epochs * self.iterations_per_epoch)
        )
        wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=CONFIG.optimization.ref_weight_decay,
            final_wd=CONFIG.optimization.final_weight_decay,
            T_max=int(CONFIG.optimization.epochs * self.iterations_per_epoch)
        )                
        return [optimizer], [
            {
                'scheduler': scheduler,
                'interval': 'step',
            }, 
            {
                'scheduler': wd_scheduler,
                'interval': 'step',
            }
        ]
