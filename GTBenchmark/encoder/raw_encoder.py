import torch
import torch.nn as nn
from torch_geometric.data import Data

from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_node_encoder, register_edge_encoder

from GTBenchmark.graphgym.register import register_config
from yacs.config import CfgNode as CN

ENCODER_NAME = 'RawEncoder'

@register_config(ENCODER_NAME)
def set_cfg(cfg):
    setattr(cfg, ENCODER_NAME, CN())
    pecfg = getattr(cfg, ENCODER_NAME)


@register_node_encoder(ENCODER_NAME)
class RawNodeEncoder(torch.nn.Module):
    """
    The raw feature node encoder.

    Apply a linear transformation to each node feature to transform them into
    a unified node embedding sapce.

    Args:
        emb_dim (int): Output embedding dimension
        dataset (Any): A :class:`~torch_geometric.data.InMemoryDataset` dataset object.
    """
    def __init__(self,dim_out):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        # self.dim_h = cfg.gt.dim_hidden
        
        self.linear = nn.Linear(self.dim_in, dim_out)
        
        
    def forward(self, batch):
        batch.x = self.linear(batch.x)

        return batch


@register_edge_encoder(ENCODER_NAME)
class RawEdgeEncoder(torch.nn.Module):
    """
    The raw feature edge encoder.
    
    Apply a linear transformation to each edge feature to transform them into
    a unified edge embedding sapce.

    Args:
        emb_dim (int): Output embedding dimension
        dataset (Any): A :class:`~torch_geometric.data.InMemoryDataset` dataset object.
    """
    def __init__(self,dim_out):
        super().__init__()
        self.dim_in = cfg.share.edge_dim_in
        # self.dim_h = getattr(cfg, ENCODER_NAME).dim_h
        
        self.linear = nn.Linear(self.dim_in, dim_out)
        
        
    def forward(self, batch):
        batch.edge_attr = self.linear(batch.edge_attr)

        return batch