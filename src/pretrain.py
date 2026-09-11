import torch
import random
import typer
import wandb

import numpy as np
import lightning as L

from torch.backends import cudnn
from torch.utils.data import DataLoader, RandomSampler
from typing import Annotated, Literal
from lightning.pytorch import callbacks
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.loggers import WandbLogger
from configs import (
    init_jepa_config,
    init_jepa_simclr_config, 
    init_mae_config,
    init_jepa_mae_config,
    init_simclr_config,
    init_vicreg_config
)
from models import JEPA, JEPASimCLR, MAE, JEPAMAE, SimCLR, VICReg
from datasets import PretrainCache, ImageDataset, DeepMIMO
from collators import get_collator, TwoViewsCollator
from evaluation import evaluate_class, evaluate_pos


torch.set_float32_matmul_precision('high')

app = typer.Typer(pretty_exceptions_show_locals=False)

INIT_CONFIG = {
    "jepa": init_jepa_config,
    "jepa_simclr": init_jepa_simclr_config,
    "mae": init_mae_config,
    "jepa_mae": init_jepa_mae_config,
    "simclr": init_simclr_config,
    "vicreg": init_vicreg_config
}

MODEL = {
    "jepa": JEPA,
    "jepa_simclr": JEPASimCLR,
    "mae": MAE,
    "jepa_mae": JEPAMAE,
    "simclr": SimCLR,
    "vicreg": VICReg
}


@app.command()
def main(method: Annotated[Literal["jepa", "jepa_simclr", "mae", "jepa_mae", "simclr", "vicreg"], typer.Option()], config_path: Annotated[str, typer.Option()]):
    CONFIG = INIT_CONFIG.get(method)(config_path)

    torch.manual_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    cudnn.benchmark = True

    wandb.login(key=CONFIG.wandb.key)
    wandb_logger = WandbLogger(project=CONFIG.wandb.project_name)

    checkpoint_filename = wandb_logger.experiment.name + "-{epoch}"
    train_callbacks = [
        callbacks.ModelCheckpoint(every_n_epochs=CONFIG.checkpoint.every_n_epochs, dirpath=CONFIG.checkpoint.dir, filename=checkpoint_filename, save_top_k=-1),
        callbacks.LearningRateMonitor(logging_interval='step'),
        callbacks.RichProgressBar(),
    ]

    if method in ["simclr", "vicreg"]:
        collator = TwoViewsCollator(
            output_size=CONFIG.transforms.output_size,
            crop=CONFIG.transforms.crop,
            crop_scale=CONFIG.transforms.crop_scale,
            n_time_masks=CONFIG.transforms.n_time_masks,
            time_mask_max_width=CONFIG.transforms.time_mask_max_width,
            n_freq_masks=CONFIG.transforms.n_freq_masks,
            freq_mask_max_width=CONFIG.transforms.freq_mask_max_width,
            add_noise=CONFIG.transforms.add_noise,
            noise_std=CONFIG.transforms.noise_std
        )
    else:
        collator_class = get_collator(CONFIG.masks.type)
        collator = collator_class(
            input_size=CONFIG.encoder.img_size,
            patch_size=CONFIG.encoder.patch_size,
            **getattr(CONFIG.masks, CONFIG.masks.type).model_dump()
        )

    dataloaders = {}
    for dataset_config in CONFIG.data.pretrain_datasets:
        dataset = PretrainCache(dataset_config.path, dataset_config.key)
        sampler_train = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset, 
            batch_size=CONFIG.data.batch_size, 
            sampler=sampler_train, 
            num_workers=CONFIG.data.num_workers, 
            pin_memory=CONFIG.data.pin_mem,
            collate_fn=collator,
            persistent_workers=True,
        )
        dataloaders[dataset_config.name] = dataloader

    train_dataloader = CombinedLoader(dataloaders, mode="max_size_cycle")

    rfs_dataset = ImageDataset(CONFIG.data.rfs_path)
    pos_dataset = ImageDataset(CONFIG.data.pos_path, sample_key="features", label_dtype=torch.float32)
    beam_dataset = DeepMIMO(CONFIG.data.beam_path, label_key="label_beam", n_beams=16)
    val_datasets = [rfs_dataset, pos_dataset, beam_dataset]
    val_dataloders = []
    for val_dataset in val_datasets:
        val_dataloders.append(DataLoader(
            val_dataset, 
            batch_size=CONFIG.data.batch_size, 
            num_workers=CONFIG.data.num_workers, 
            pin_memory=CONFIG.data.pin_mem,
            shuffle=False,
        ))

    wandb_logger.log_hyperparams(CONFIG.model_dump())
    iter(train_dataloader)
    model = MODEL[method](
        CONFIG, 
        len(train_dataloader), 
        [evaluate_class, evaluate_pos, evaluate_class], 
        ["rfs_pca", "pos_error", "beam_pca"]
    )
    
    trainer = L.Trainer(
        max_epochs=CONFIG.optimization.epochs,
        callbacks=train_callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        check_val_every_n_epoch=CONFIG.checkpoint.every_n_epochs,
        gradient_clip_val=CONFIG.optimization.grad_clip
    )
    trainer.fit(model, train_dataloader, val_dataloders)


if __name__ == "__main__":
    app()
