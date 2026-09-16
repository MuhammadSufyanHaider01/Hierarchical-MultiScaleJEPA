from .multi_block_mask import MultiBlockMask
from .random_mask import RandomMask
from .variation_aware_mask import VariationAwareMultiBlockMask

def get_collator(name):
    if name == "random":
        return RandomMask
    elif name == "multi_block":
        return MultiBlockMask
    elif name == "variation_aware_multi_block":
        return VariationAwareMultiBlockMask
    else:
        raise ValueError(f"Unsupported collator: {name}")
