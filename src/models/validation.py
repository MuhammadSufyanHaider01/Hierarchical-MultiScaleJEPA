from abc import ABC, abstractmethod

import torch

from lightning import LightningModule


class ValidationModule(LightningModule, ABC):
    def __init__(self, eval_fns: list[callable], metrics: list[str]):
        super().__init__()
        self.validation_embeddings = [[] for _ in range(len(metrics))]
        self.labels = [[] for _ in range(len(metrics))]
        self.metrics = metrics
        self.eval_fns = eval_fns

    @abstractmethod
    def _get_validation_tokens_reps(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _pool_validation_reps(self, token_latent_reps: torch.Tensor) -> torch.Tensor:
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 2:
            x, y, _ = batch
        else:
            x, y = batch

        token_latent_reps = self._get_validation_tokens_reps(x)
        sample_reps = self._pool_validation_reps(token_latent_reps)
        self.validation_embeddings[dataloader_idx].append(sample_reps.detach().cpu())
        self.labels[dataloader_idx].append(y.cpu())

    def on_validation_epoch_end(self):
        if self.global_step == 0:
            for i in range(len(self.metrics)):
                self.validation_embeddings[i].clear()
                self.labels[i].clear()
            return

        for val_embeddings, labels, eval_fn, metric in zip(
            self.validation_embeddings, self.labels, self.eval_fns, self.metrics
        ):
            perf = eval_fn(val_embeddings, labels)
            self.log(metric, perf, on_epoch=True, logger=True)

        for i in range(len(self.metrics)):
            self.validation_embeddings[i].clear()
            self.labels[i].clear()
