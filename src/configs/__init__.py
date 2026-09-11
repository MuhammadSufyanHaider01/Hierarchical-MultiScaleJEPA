"""Configuration exports available in the pinned LatentWave tree.

The upstream commit contains only the JEPA configuration implementation, while
its initializer imports optional baseline modules that are absent at that
commit. Keep the P0 import surface explicit instead of fabricating those
baselines or changing the JEPA configuration itself.
"""

from .jepa import Config as JEPAConfig
from .jepa import get_config as get_jepa_config
from .jepa import init_config as init_jepa_config

__all__ = ["JEPAConfig", "get_jepa_config", "init_jepa_config"]
