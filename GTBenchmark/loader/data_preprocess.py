from typing import List
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_scipy_sparse_matrix
from GTBenchmark.graphgym.config import cfg
import copy

def hop2token(dataset) -> InMemoryDataset:
    """
    完全复刻 NAG 中 re_features 的 Hop2Token 逻辑：
      - 不归一化 A，只做简单求和聚合
      - 输出 data.x 形状变为 (N, K+1, d)
    
    Args
    ----
    dataset : 原始 PyG InMemoryDataset
    K       : 传播步数（hops）
    
    Returns
    -------
    new_dataset : 与输入同类型、同切片结构，但 data.x 已替换为 Hop2Token 特征
    """
    processed: List[Data] = []
    K = cfg.dataset.hop
    for data in dataset:
        # 克隆 data，避免修改原对象
        data = copy.copy(data)
        
        x = data.x                     # (N, d)
        N, d = x.size()
        edge_index = data.edge_index   # (2, E)
        device = x.device
        
        # —— 1. 构邻接稠密矩阵 (N, N)，保持与官方代码“dense matmul”完全一致 ——
        adj = torch.zeros((N, N), dtype=x.dtype, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # —— 2. Hop2Token 聚合 —— 
        hop_feat = torch.zeros((N, K + 1, d), dtype=x.dtype, device=device)
        hop_feat[:, 0, :] = x  # hop 0
        
        x_prop = x.clone()
        for h in range(1, K + 1):
            x_prop = adj @ x_prop        # 无归一化求和
            hop_feat[:, h, :] = x_prop
        
        # —— 3. 写回 Data，并保存 —— 
        data.x = hop_feat                # (N, K+1, d)
        data.num_node_features = hop_feat.shape[-1] * (K + 1)  # 有需要再更新
        processed.append(data)
    
    # —— 4. collate 回新的 InMemoryDataset —— 
    new_data, new_slices = dataset.collate(processed)
    new_ds = copy.copy(dataset)
    new_ds.data, new_ds.slices = new_data, new_slices
    # cfg.share.dim_in = 
    return new_ds
