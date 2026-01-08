import torch
from torch_scatter import scatter

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg


# Pooling options (pool nodes into graph representations)
# pooling function takes in node embedding [num_nodes x emb_dim] and
# batch (indices) and outputs graph embedding [num_graphs x emb_dim].
def global_add_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def global_mean_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def global_max_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')



# ------------------------------------------------------------
# DenseFirst pooling primitives
# ------------------------------------------------------------

def _node_mask(num_nodes: torch.Tensor, M: int):
    """
    num_nodes: [B]
    return: mask [B, M, 1], True = valid node
    """
    device = num_nodes.device
    mask = torch.arange(M, device=device)[None, :] < num_nodes[:, None]
    return mask.unsqueeze(-1)


def dense_add_pool(x, num_nodes):
    """
    x: [B, M, F]
    num_nodes: [B]
    return: [B, F]
    """
    B, M, F = x.shape
    mask = _node_mask(num_nodes, M)
    return (x * mask).sum(dim=1)


def dense_mean_pool(x, num_nodes):
    """
    x: [B, M, F]
    num_nodes: [B]
    return: [B, F]
    """
    B, M, F = x.shape
    mask = _node_mask(num_nodes, M)
    return (x * mask).sum(dim=1) / num_nodes.unsqueeze(-1)


def dense_max_pool(x, num_nodes):
    """
    x: [B, M, F]
    num_nodes: [B]
    return: [B, F]
    """
    B, M, F = x.shape
    mask = _node_mask(num_nodes, M)
    x = x.masked_fill(~mask, float('-inf'))
    return x.max(dim=1).values



pooling_dict = {
    'add': global_add_pool,
    'mean': global_mean_pool,
    'max': global_max_pool,
    'add_df': dense_add_pool,
    'mean_df': dense_mean_pool,
    'max_df': dense_max_pool,
}

register.pooling_dict = {**register.pooling_dict, **pooling_dict}
