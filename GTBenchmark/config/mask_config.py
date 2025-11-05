from GTBenchmark.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('mask')
def set_cfg_mask(cfg):
    """Extend configuration with Attention mask&Graph and dense Data transform options.
    """

    # Argument group for each Positional Encoding class.
    cfg.mask = CN()

    cfg.mask.name = "full"
    cfg.mask.use_flex = False
    
    # METIS partitioning
    cfg.metis = CN()
    cfg.metis.enable = False
    cfg.metis.patches = 0
    cfg.metis.num_hops = 1
    cfg.metis.drop_rate = 0.3
    cfg.metis.online = True
    cfg.metis.patch_rw_dim = 0
    cfg.metis.patch_num_diff = -1




    """Extend configuration with positional encoding options.
    """

    # Argument group for each Positional Encoding class.
    cfg.reorder_RCM = CN()
    cfg.reorder_Slashburn = CN()
    cfg.reorder_Metis = CN()
    cfg.reorder_Random = CN()
    cfg.reorder_Degree = CN()
    cfg.reorder_BFS = CN()
    cfg.reorder_GOrder = CN()


    # Common arguments to all PE types.
    for name in [
        'reorder_Metis','reorder_RCM','reorder_Slashburn','reorder_Random',
        'reorder_Degree', 'reorder_BFS', 'reorder_GOrder'
    ]:
        recfg = getattr(cfg, name)

        # Use extended positional encodings
        recfg.enable = False
        recfg.block_size = 128




    cfg.reorder_Slashburn.k = 300
    cfg.reorder_GOrder.window = 32
    