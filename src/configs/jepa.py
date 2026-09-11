import yaml

from pydantic_settings import BaseSettings
from .components import EncoderConfig, PredictorConfig, WandBConfig, DatasetConfig, MasksConfig, CheckpointConfig


class OptimizationConfig(BaseSettings):
    ema: list[float]
    epochs: int
    warmup: int
    start_lr: float
    ref_lr: float
    final_lr: float
    ref_weight_decay: float
    final_weight_decay: float
    grad_clip: float | None


class Config(BaseSettings):
    encoder: EncoderConfig
    predictor: PredictorConfig
    wandb: WandBConfig
    data: DatasetConfig
    masks: MasksConfig
    optimization: OptimizationConfig
    checkpoint: CheckpointConfig
    seed: int


CONFIG: Config = None


def init_config(config_path: str) -> Config:
    global CONFIG
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    CONFIG = Config(**config_dict)
    return CONFIG


def get_config() -> Config:
    if CONFIG is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return CONFIG
