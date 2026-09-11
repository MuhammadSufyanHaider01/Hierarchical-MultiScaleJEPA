from .multi_block_mask import MultiBlockMask
from .random_mask import RandomMask
from .two_views import TwoViewsCollator

def get_collator(name):
    if name == "random":
        return RandomMask
    elif name == "multi_block":
        return MultiBlockMask
    else:
        raise ValueError(f"Unsupported collator: {name}")
