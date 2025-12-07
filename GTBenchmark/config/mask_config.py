from GTBenchmark.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('mask')
def set_cfg_mask(cfg):
    """Extend configuration with Attention mask&Graph and dense Data transform options.
    """

    # Argument group for each Positional Encoding class.
    cfg.mask = CN()

    cfg.mask.name = "full"
    cfg.mask.transadj = False #when Edge attr become attn_bias
    
    # METIS partitioning
    cfg.metis = CN()
    cfg.metis.enable = False
    cfg.metis.patches = 0
    cfg.metis.num_hops = 0
    cfg.metis.drop_rate = 0.3
    cfg.metis.online = True
    cfg.metis.patch_rw_dim = 0
    cfg.metis.patch_num_diff = -1
    cfg.metis.cut_type = "edge-cut"
    cfg.metis.mode = "subgraph"
