from GTBenchmark.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('mask')
def set_cfg_mask(cfg):
    """Extend configuration with Attention mask&Graph and dense Data transform options.
    """

    # Argument group for each Positional Encoding class.
    cfg.mask = CN()

    cfg.mask.name = "full"
    
    