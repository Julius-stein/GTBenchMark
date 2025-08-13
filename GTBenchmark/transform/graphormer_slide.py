# graphormer_preprocess_light.py
import torch
from torch_geometric.data import Data

def graphormer_preprocess_light(d: Data, undirected: bool = True) -> Data:
    """只做安全字段：in_degrees/out_degrees（[N]）。绝不写 NxN。"""
    N = int(d.num_nodes)
    in_deg = torch.zeros(N, dtype=torch.long)
    out_deg = torch.zeros(N, dtype=torch.long)
    ones = torch.ones(d.edge_index.size(1), dtype=torch.long)
    in_deg.index_add_(0, d.edge_index[1], ones)
    out_deg.index_add_(0, d.edge_index[0], ones)
    if undirected:
        in_deg = out_deg = (in_deg + out_deg) // 2
    d.in_degrees = in_deg
    d.out_degrees = out_deg
    return d
