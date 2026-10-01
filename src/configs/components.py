import yaml

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class EncoderConfig(BaseSettings):
    img_size: tuple[int, int]
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    backbone: Literal["vit", "convnext_v2_hier"] = "vit"
    stage_depths: tuple[int, int, int, int] = (2, 2, 6, 2)
    stage_channels: tuple[int, int, int, int] = (48, 96, 192, 384)
    mask_value: float = 0.0
    gradient_checkpointing: bool = False


class PredictorConfig(BaseSettings):
    embed_dim: int
    depth: int
    num_heads: int


class HierarchyConfig(BaseSettings):
    """Optional P1 same-scale or P2 mask-relative JEPA supervision."""

    enabled: bool = False
    mode: Literal[
        "same_scale",
        "mask_relative_fine_to_coarse",
        "mask_relative_coarse_to_fine",
    ] = "same_scale"
    layers: list[int] = [2, 4, 6]
    lambda_h: float = Field(default=1.0, validation_alias="lambda")
    warmup_epochs: int = 12
    include_l8_coarse: bool = False


class ProjectorConfig(BaseSettings):
    num_blocks: int
    hidden_dim: int


class WandBConfig(BaseSettings):
    project_name: str
    key: str = Field(validation_alias="WANDB_KEY")


class PrettrainDatasetConfig(BaseSettings):
    name: str
    path: str
    key: str


class DatasetConfig(BaseSettings):
    pretrain_datasets: list[PrettrainDatasetConfig]
    rfs_path: str
    pos_path: str
    beam_path: str
    batch_size: int
    num_workers: int
    pin_mem: bool


class MultiBlockMaskConfig(BaseSettings):
    allow_overlap: bool
    aspect_ratio_type: Literal["span_freq", "span_time", "random"]
    aspect_ratio: list[float]
    enc_mask_scale: list[float]
    min_keep: int
    num_enc_masks: int
    num_pred_masks: int
    pred_mask_scale: list[float]


class RandomMaskConfig(BaseSettings):
    ratios: list[float]


class MasksConfig(BaseSettings):
    type: str
    multi_block: MultiBlockMaskConfig
    random: RandomMaskConfig


class TwoViewsTransformsConfig(BaseSettings):
  output_size: tuple[int, int]
  crop: bool
  crop_scale: tuple[float, float]
  n_time_masks: int
  time_mask_max_width: int
  n_freq_masks: int
  freq_mask_max_width: int
  add_noise: bool
  noise_std: float

  
class CheckpointConfig(BaseSettings):
    dir: str
    every_n_epochs: int
