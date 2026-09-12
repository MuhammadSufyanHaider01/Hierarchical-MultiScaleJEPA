import torch
import copy
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from configs import JEPAConfig
from models import Encoder, Predictor
from .utils import apply_masks, repeat_interleave_batch, get_effective_rank, sample_channel_subsets
from .regions import P2_LAYERS, build_mask_relative_region_layout, pool_target_regions
from .validation import ValidationModule
from checkpointing import build_checkpoint_metadata
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
        hierarchy = getattr(CONFIG, "hierarchy", None)
        self.hierarchy_enabled = bool(getattr(hierarchy, "enabled", False))
        self.hierarchy_mode = getattr(hierarchy, "mode", "same_scale")
        self.hierarchy_layers = tuple(int(layer) for layer in getattr(hierarchy, "layers", (2, 4, 6)))
        self.hierarchy_lambda = float(getattr(hierarchy, "lambda_h", 1.0))
        self.hierarchy_warmup_epochs = int(getattr(hierarchy, "warmup_epochs", 12))
        self.include_l8_coarse = bool(getattr(hierarchy, "include_l8_coarse", False))
        self._validate_hierarchy_config()
        self.auxiliary_layers = self.hierarchy_layers
        if self.hierarchy_enabled and self.include_l8_coarse:
            self.auxiliary_layers = self.hierarchy_layers + (len(self.context_encoder.blocks),)
        active_hierarchy_layers = self.auxiliary_layers if self.hierarchy_enabled else ()
        self.aux_predictors = nn.ModuleDict({
            str(layer): Predictor(
                num_patches=(CONFIG.encoder.img_size[0] // CONFIG.encoder.patch_size) ** 2,
                encoder_embed_dim=CONFIG.encoder.embed_dim,
                output_dim=CONFIG.encoder.embed_dim,
                predictor_embed_dim=CONFIG.predictor.embed_dim,
                depth=CONFIG.predictor.depth,
                num_heads=CONFIG.predictor.num_heads,
            )
            for layer in active_hierarchy_layers
        })
        self.loss = nn.MSELoss()
        self.iterations_per_epoch = iterations_per_epoch

        self.ema_start = CONFIG.optimization.ema[0]
        self.ema_end = CONFIG.optimization.ema[1]
        self.total_steps = iterations_per_epoch * CONFIG.optimization.epochs
        self.num_patches = (CONFIG.encoder.img_size[0] // CONFIG.encoder.patch_size) ** 2

        self._freeze_target_encoder()
        self.config = CONFIG
        self.checkpoint_metadata = build_checkpoint_metadata(CONFIG, seed=CONFIG.seed)

    def _validate_hierarchy_config(self):
        if not self.hierarchy_enabled:
            return
        supported_modes = {"same_scale", "mask_relative_fine_to_coarse"}
        if self.hierarchy_mode not in supported_modes:
            raise ValueError(f"unsupported hierarchy mode: {self.hierarchy_mode!r}")
        if self.hierarchy_warmup_epochs < 0:
            raise ValueError("hierarchy warmup_epochs must be non-negative")
        if self.hierarchy_lambda < 0:
            raise ValueError("hierarchy lambda must be non-negative")
        if len(set(self.hierarchy_layers)) != len(self.hierarchy_layers):
            raise ValueError(f"hierarchy layers must be unique, got {self.hierarchy_layers}")
        depth = len(self.context_encoder.blocks)
        if any(layer < 1 or layer >= depth for layer in self.hierarchy_layers):
            raise ValueError(
                f"same-scale auxiliary layers must be in [1, {depth - 1}], got {self.hierarchy_layers}"
            )
        if self.hierarchy_mode == "mask_relative_fine_to_coarse" and self.hierarchy_layers != P2_LAYERS:
            raise ValueError(
                f"P2 requires auxiliary layers {P2_LAYERS}, got {self.hierarchy_layers}"
            )
        if self.include_l8_coarse and (
            not self.hierarchy_enabled or self.hierarchy_mode != "mask_relative_fine_to_coarse"
        ):
            raise ValueError("include_l8_coarse is an opt-in P2-only ablation")

    def _freeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @staticmethod
    def _target_for_prediction_masks(target_out, channels, masks_enc, masks_pred):
        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]
        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        target_out = F.layer_norm(target_out, (target_out.size(-1),))
        batch, _, dim = target_out.shape
        target_out = target_out.reshape(batch, channels, -1, dim)
        target_out = apply_masks(target_out, masks_pred)
        target_out = repeat_interleave_batch(target_out, B=batch, repeat=len(masks_enc))
        return target_out.reshape(target_out.size(0), -1, dim)

    def hierarchy_weight(self, epoch: int | None = None) -> float:
        """Return the epoch-level P1 auxiliary-loss coefficient."""
        if not self.hierarchy_enabled:
            return 0.0
        if self.hierarchy_warmup_epochs == 0:
            return self.hierarchy_lambda
        epoch = self.current_epoch if epoch is None else int(epoch)
        progress = min(1.0, max(0.0, (epoch + 1) / self.hierarchy_warmup_epochs))
        return self.hierarchy_lambda * progress

    def _hierarchy_total_loss(self, anchor_loss, auxiliary_losses, hierarchy_weight):
        """Combine the unchanged anchor with the mean auxiliary loss."""
        auxiliary_mean = torch.stack(
            [auxiliary_losses[layer] for layer in self.auxiliary_layers]
        ).mean()
        return anchor_loss + hierarchy_weight * auxiliary_mean

    def _p1_total_loss(self, anchor_loss, auxiliary_losses, hierarchy_weight):
        """Compatibility wrapper for the P1 objective helper."""
        return self._hierarchy_total_loss(anchor_loss, auxiliary_losses, hierarchy_weight)

    def forward(self, x, masks_enc, masks_pred, *, return_hierarchy=False):
        C = x.size(1)
        hierarchy_requested = self.hierarchy_enabled and return_hierarchy
        p2_layout = None

        if hierarchy_requested:
            context_layers = tuple(sorted(set(self.auxiliary_layers + (len(self.context_encoder.blocks),))))
            context_outputs = self.context_encoder(x, masks_enc, return_layers=context_layers)
            context_emb = context_outputs[len(self.context_encoder.blocks)]
        else:
            context_outputs = None
            context_emb = self.context_encoder(x, masks_enc)
        latent_mask_pred = self.predictor(context_emb, C, masks_enc, masks_pred)

        with torch.no_grad():
            target_out = self.target_encoder(x)
            latent_mask_target = self._target_for_prediction_masks(
                target_out, C, masks_enc, masks_pred
            )
            auxiliary_targets = {}
            if hierarchy_requested:
                if self.hierarchy_mode == "same_scale":
                    target_outputs = self.target_encoder(x, return_layers=self.auxiliary_layers)
                    auxiliary_targets = {
                        layer: self._target_for_prediction_masks(
                            target_outputs[layer], C, masks_enc, masks_pred
                        )
                        for layer in self.auxiliary_layers
                    }
                else:
                    p2_layout = build_mask_relative_region_layout(
                        masks_pred,
                        grid_size=int(self.num_patches ** 0.5),
                    )
                    target_outputs = self.target_encoder(
                        x,
                        return_layers=self.auxiliary_layers,
                        normalize_exits=False,
                    )
                    num_context_masks = len(masks_enc) if isinstance(masks_enc, list) else 1
                    auxiliary_targets = {
                        layer: pool_target_regions(
                            target_outputs[layer],
                            channels=C,
                            layout=p2_layout,
                            # L8 coarse supervision pools the same complete
                            # masked block as L6, but from L8 target features.
                            layer=6 if layer == len(self.context_encoder.blocks) else layer,
                        )
                        .unsqueeze(1)
                        .expand(-1, num_context_masks, -1, -1, -1, -1)
                        for layer in self.auxiliary_layers
                    }

        if not hierarchy_requested:
            return latent_mask_pred, latent_mask_target, target_out

        if self.hierarchy_mode == "same_scale":
            auxiliary_predictions = {
                layer: self.aux_predictors[str(layer)](
                    context_outputs[layer], C, masks_enc, masks_pred
                )
                for layer in self.auxiliary_layers
            }
        else:
            auxiliary_predictions = {
                layer: self.aux_predictors[str(layer)].forward_regions(
                    context_outputs[layer],
                    C,
                    masks_enc,
                    p2_layout.for_layer(6 if layer == len(self.context_encoder.blocks) else layer),
                )
                for layer in self.auxiliary_layers
            }
        return latent_mask_pred, latent_mask_target, target_out, auxiliary_predictions, auxiliary_targets

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

        if self.hierarchy_enabled:
            (
                latent_mask_pred,
                latent_mask_target,
                target_out,
                auxiliary_predictions,
                auxiliary_targets,
            ) = self(x, masks_enc, masks_pred, return_hierarchy=True)
        else:
            latent_mask_pred, latent_mask_target, target_out = self(x, masks_enc, masks_pred)
        
        anchor_loss = self.loss(latent_mask_pred, latent_mask_target)
        loss = anchor_loss
        if self.hierarchy_enabled:
            hierarchy_weight = self.hierarchy_weight()
            self.log("hierarchy_weight", hierarchy_weight, on_step=True, on_epoch=True, logger=True)
            self.log(f"train_loss_{name}_anchor", anchor_loss, on_step=True, on_epoch=True, logger=True)
            auxiliary_losses = {}
            for layer in self.auxiliary_layers:
                auxiliary_prediction = auxiliary_predictions[layer]
                auxiliary_target = auxiliary_targets[layer]
                auxiliary_loss = self.loss(auxiliary_prediction, auxiliary_target)
                auxiliary_losses[layer] = auxiliary_loss
                self.log(
                    f"train_loss_{name}_L{layer}",
                    auxiliary_loss,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    f"valid_target_count_L{layer}",
                    float(self._target_vector_count(auxiliary_target)),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    f"target_rms_L{layer}",
                    self._rms(auxiliary_target),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    f"predictor_rms_L{layer}",
                    self._rms(auxiliary_prediction),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
            loss = self._hierarchy_total_loss(anchor_loss, auxiliary_losses, hierarchy_weight)
        self.log(f'train_loss_{name}', loss, on_step=True, on_epoch=True, logger=True)

        mask_percentage = 1 - len(masks_enc[0][0]) / self.num_patches

        latent_rep = target_out.mean(dim=1).detach()
        
        return loss, mask_percentage, latent_rep

    @staticmethod
    def _rms(value):
        return value.detach().square().mean().sqrt()

    @staticmethod
    def _target_vector_count(value):
        count = 1
        for dimension in value.shape[:-1]:
            count *= dimension
        return count

    def on_after_backward(self):
        if not self.hierarchy_enabled:
            return
        for layer, predictor in self.aux_predictors.items():
            gradients = [parameter.grad.detach().square().sum() for parameter in predictor.parameters() if parameter.grad is not None]
            if gradients:
                gradient_norm = torch.stack(gradients).sum().sqrt()
            else:
                gradient_norm = torch.tensor(0.0, device=self.device)
            self.log(
                f"predictor_grad_norm_L{layer}",
                gradient_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )
    
    def on_train_batch_start(self, batch, batch_idx):
        wd = self.optimizers().param_groups[0]['weight_decay']
        self.log('weight_decay', wd, on_step=True, on_epoch=False, logger=True) 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # The pinned path has one optimizer update per batch. Phase 5 uses
        # gradient accumulation to retain its effective batch size on the
        # available GPU. Do not apply an EMA update to a target branch when
        # the context branch has not yet taken an optimizer step. With
        # accumulation=1 this guard is never taken, preserving P0 behavior.
        if self.trainer.fit_loop._should_accumulate():
            return
        with torch.no_grad():
            m = self.ema_start + (self.global_step - 1) * (self.ema_end - self.ema_start) / self.total_steps
            self.update_target_encoder(m)
            
        self.log("ema_momentum", m, on_step=True, on_epoch=False, logger=True)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Apply the original LatentWave EMA update to target parameters."""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["metadata"] = self.checkpoint_metadata

    def _get_validation_tokens_reps(self, x):
        # Version 1 downstream evaluation uses the learned context branch;
        # the EMA target branch is pretraining-only.
        return self.context_encoder(x)

    def _pool_validation_reps(self, token_latent_reps):
        return token_latent_reps.mean(dim=1)

    def configure_optimizers(self):
        CONFIG = self.config
        predictor_modules = [self.predictor, *self.aux_predictors.values()]

        def predictor_parameters(exclude_weight_decay):
            for predictor in predictor_modules:
                for name, parameter in predictor.named_parameters():
                    no_weight_decay = ('bias' in name) or (len(parameter.shape) == 1)
                    if no_weight_decay == exclude_weight_decay:
                        yield parameter

        param_groups = [
            {
                'params': (p for n, p in self.context_encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': predictor_parameters(exclude_weight_decay=False)
            }, {
                'params': (p for n, p in self.context_encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': predictor_parameters(exclude_weight_decay=True),
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
