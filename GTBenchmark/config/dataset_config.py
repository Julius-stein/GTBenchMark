from GTBenchmark.graphgym.register import register_config
from typing import Union

@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The entity to perform the task in an heterogeneous graph dataset
    cfg.dataset.task_entity = ""

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    cfg.dataset.undirected = False

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    cfg.dataset.rand_split = False

    cfg.dataset.preprocess = "None"

    cfg.dataset.add_self_loops = True

    # NAG preprocess
    cfg.dataset.hop = 3 

    cfg.dataset.heteroProcess = False

    # Expander preprocess
    cfg.dataset.add_edge_index= True
    cfg.dataset.dist_cutoff= 510
    cfg.dataset.dist_enable= False
    cfg.dataset.exp_algorithm= "Random-d"
    cfg.dataset.exp_count= 1
    cfg.dataset.exp_deg= 3
    cfg.dataset.exp_max_num_iters= 100
    cfg.dataset.num_virt_node= 0
    cfg.dataset.rb_order= 1
    cfg.dataset.use_exp_edges= True



    