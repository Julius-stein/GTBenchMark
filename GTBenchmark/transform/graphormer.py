import torch
import numpy as np
from torch_geometric.data import Data
# from GTBenchmark.transform.gf_algos import algos_numba as nb
# graphormer_data.py



# 可选：替换为 torchscript 版（和官方一致）
@torch.jit.script
def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    # 支持 (N,) 或 (N, F)
    feature_num: int = x.size(1) if x.dim() > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    # (N, F) + (F,) = 广播，(N,) 情况上面 feature_num=1 也兼容
    return x.long() + feature_offset

# —— 优先用 Cython 实现，备用用 Numba BFS 实现 ——
try:
    # 来自 microsoft/Graphormer 仓库
    from GTBenchmark.transform.gf_algos import algos as cy_algos
    HAS_CYTHON = True
except Exception:
    cy_algos = None
    HAS_CYTHON = False
# TODO：BFS算法
try:
    from GTBenchmark.transform.gf_algos import algos_numba as nb_algos
    HAS_NUMBA = True
except Exception:
    nb_algos = None
    HAS_NUMBA = False


def graphormer_pre_processing(
    data: Data,
    max_spatial_dist: int = 5,
    algo: str = "bfs_numba",          # 可选: "floyd" | "bfs_numba"
    undirected: bool = True,
) -> Data:
    """
    以 Graphormer 官方 preprocess_item 为准，直接产出官方需要的字段：
    - x: convert_to_single_emb 后的节点离散特征 (与官方一致，后续 Embedding 使用)
    - attn_bias: [N+1, N+1]，含 graph token 的偏置基底（先置 0）
    - attn_edge_type: [N, N, F_e] 的整型张量，存放边特征（convert_to_single_emb + 1）
    - spatial_pos: [N, N] 最短路径距离（不可达按官方惯例会被 algos 置为 510）
    - edge_input: [N, N, D, F_e] 沿最短路逐跳的边特征（截断到 D）
    - in_degree / out_degree: 节点度（无向图两者一致）
    """
    assert data.edge_index is not None, "edge_index is required"
    N = int(data.num_nodes)

    # 1) 节点特征 -> 单向量离散编码（官方做法）
    if getattr(data, "x", None) is None:
        # 没有 x 就按 1 维哑变量处理（全 0），保证后续 Embedding 不崩
        data.x = torch.zeros((N, 1), dtype=torch.long)
    x = data.x
    if x.dim() == 1:
        x = x.unsqueeze(1)
    x = convert_to_single_emb(x)  # (N, F_x) long
    data.x = x

    # 2) 构造布尔邻接（必要时对称化）
    ei = data.edge_index
    adj = torch.zeros((N, N), dtype=torch.bool)
    adj[ei[0], ei[1]] = True
    if undirected:
        adj = adj | adj.t()

    # 3) 边特征拉到 [N,N,Fe]（convert_to_single_emb 后 +1，0 作为 padding）
    if getattr(data, "edge_attr", None) is None:
        edge_attr = torch.zeros((ei.size(1), 1), dtype=torch.long)
    else:
        edge_attr = data.edge_attr.long()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

    ea_single = convert_to_single_emb(edge_attr) + 1  # (E, Fe)
    Fe = int(ea_single.size(-1))
    attn_edge_type = torch.zeros((N, N, Fe), dtype=torch.long)
    attn_edge_type[ei[0], ei[1]] = ea_single
    if undirected:
        attn_edge_type[ei[1], ei[0]] = ea_single

    # 4) 用 Numba 生成 SPD 与多跳边特征
    adj_np = np.ascontiguousarray(adj.cpu().numpy(), dtype=np.int64)
    aetype_np = np.ascontiguousarray(attn_edge_type.cpu().numpy(), dtype=np.int64)
    sp_np, edge_input_np = nb_algos.bfs_numba_spatial_pos_and_edge_input(
        adj_np, aetype_np, int(max_spatial_dist)
    )
    # 注意：Numba 返回的 spatial_pos 不可达仍是 510；edge_input 形状 [N,N,D,Fe]，D=max_spatial_dist

    # 5) ——关键：改成“可 collate”的扁平存储——
    # 5.1 (i,j) 索引（行主序）
    idx = torch.arange(N, dtype=torch.long)
    ii = idx.repeat_interleave(N)    # [0,0,...,1,1,...]
    jj = idx.repeat(N)               # [0,1,2,...,0,1,2,...]
    graph_index = torch.stack([ii, jj], dim=0)              # [2, N*N]

    # 5.2 扁平化 spatial_pos 和 edge_input
    spatial_types = torch.from_numpy(sp_np.reshape(-1).astype(np.int64))  # [N*N]
    shortest_path_types = torch.from_numpy(
        edge_input_np.reshape(N * N, edge_input_np.shape[2], edge_input_np.shape[3]).astype(np.int64)
    )  # [N*N, D, Fe]  —— D=max_spatial_dist（全图一致），Fe 在数据集内应一致

    # 6) 度（无向图 in/out 相同）
    deg = adj.long().sum(dim=1).view(-1)
    data.in_degrees = deg
    data.out_degrees = deg

    # 7) 仅保存“可 collate”的字段；不要把 NxN 方阵塞进 Data！
    data.graph_index = graph_index              # [2, N*N]
    data.spatial_types = spatial_types          # [N*N]
    data.shortest_path_types = shortest_path_types  # [N*N, D, Fe]

    # 可选：如果后续还需要原子边特征（E,Fe_raw）做别的事，可以另外保存
    # data.edge_attr_single = ea_single          # (E, Fe)
    return data


# 若你是从其它地方 import 的，就改成对应路径：from graphormer.data import algos_numba as nb_algos

# def graphormer_pre_processing(
#     data: Data,
#     max_spatial_dist: int = 15,     # 多跳上限，Numba 会自动 clamp
#     undirected: bool = True,
# ) -> Data:
#     assert data.edge_index is not None
#     N = int(data.num_nodes)

#     # 1) 节点特征（离散合并）
#     x = data.x if getattr(data, "x", None) is not None else torch.zeros((N, 1), dtype=torch.long)
#     if x.dim() == 1:
#         x = x.unsqueeze(1)
#     data.x = convert_to_single_emb(x)

#     # 2) 邻接（布尔/0-1）
#     ei = data.edge_index
#     adj = torch.zeros((N, N), dtype=torch.bool)
#     adj[ei[0], ei[1]] = True
#     if undirected:
#         adj |= adj.t()

    