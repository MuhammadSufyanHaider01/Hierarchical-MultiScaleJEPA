"""P0 model exports from the pinned LatentWave implementation.

Only the JEPA path is present in the pinned source tree. Optional model files
referenced by the upstream initializer are intentionally not imported here.
"""

from .encoder import Encoder, EncoderCLS
from .predictor import Predictor, PredictorCLS
from .jepa import JEPA
from .validation import ValidationModule
from .latentwfm import LatentWFM, get_latent_wfm
from .factory import build_encoder
from .exit_heads import (
    CausalFusionExitHead,
    FrozenEncoderFusionExit,
    FrozenEncoderFusionTeacherStudent,
    ResidualProjectedFusionExitHead,
    FrozenEncoderResidualFusionExit,
    ADAPTER_ABLATION_VARIANTS,
    ResidualFusionAblationExitHead,
    FrozenEncoderResidualFusionAblationExit,
    ExitHead,
    FrozenEncoderExit,
    load_context_encoder_checkpoint,
    load_target_encoder_checkpoint,
)

__all__ = [
    "Encoder",
    "EncoderCLS",
    "build_encoder",
    "Predictor",
    "PredictorCLS",
    "JEPA",
    "ValidationModule",
    "LatentWFM",
    "get_latent_wfm",
    "ExitHead",
    "FrozenEncoderExit",
    "CausalFusionExitHead",
    "FrozenEncoderFusionExit",
    "FrozenEncoderFusionTeacherStudent",
    "ResidualProjectedFusionExitHead",
    "FrozenEncoderResidualFusionExit",
    "ADAPTER_ABLATION_VARIANTS",
    "ResidualFusionAblationExitHead",
    "FrozenEncoderResidualFusionAblationExit",
    "load_context_encoder_checkpoint",
    "load_target_encoder_checkpoint",
]
