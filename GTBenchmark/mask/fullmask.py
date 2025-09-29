import torch
import torch.nn as nn
import torch.nn.functional as F
from .basemask import Basemask
import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_mask
from GTBenchmark.transform.graph2dense import to_dense_batch,get_graph_sizes


@register_mask('full')
class FullMask(Basemask):
    def __init__(self,batch):
        super().__init__(batch)
                   

