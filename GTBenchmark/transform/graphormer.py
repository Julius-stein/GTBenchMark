import torch
import numpy as np
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.transform.gf_algos.algos_numba import bfs_numba_spatial_pos_and_edge_input

def graphormer_pre_processing(data, distance: int):
    """Graphormer预处理：使用Numba BFS，产物与原BiasEncoder对齐：
       - graph_index: [2, M], M=N*N
       - spatial_types: [M]
       - shortest_path_types: [M, D] （二维！）
    """
    # 1) 入/出度（有向）
    row, col = data.edge_index
    num_nodes = int(data.num_nodes)
    in_degrees = torch.zeros(num_nodes, dtype=torch.long)
    out_degrees = torch.zeros(num_nodes, dtype=torch.long)

    # 避免 .tolist() 的额外开销
    for u, v in zip(row.cpu().numpy().tolist(), col.cpu().numpy().tolist()):
        out_degrees[u] += 1
        in_degrees[v] += 1

    data.in_degrees = in_degrees
    data.out_degrees = out_degrees

    max_in_degree = int(in_degrees.max())
    max_out_degree = int(out_degrees.max())
    if max_in_degree >= cfg.posenc_GraphormerBias.num_in_degrees:
        raise ValueError(
            f"Encountered in_degree: {max_in_degree}, set "
            f"posenc_GraphormerBias.num_in_degrees >= {max_in_degree + 1}"
        )
    if max_out_degree >= cfg.posenc_GraphormerBias.num_out_degrees:
        raise ValueError(
            f"Encountered out_degree: {max_out_degree}, set "
            f"posenc_GraphormerBias.num_out_degrees >= {max_out_degree + 1}"
        )

    if cfg.posenc_GraphormerBias.node_degrees_only:
        return data

    # 2) 邻接&边类型矩阵（注意一维/二维边特征的区分）
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int8)
    has_edge_attr = hasattr(data, "edge_attr") and (data.edge_attr is not None)

    if has_edge_attr:
        ea = data.edge_attr
        if ea.dim() == 1:
            edge_type_dim = 1
            ea = ea.view(-1, 1).long()
        else:
            edge_type_dim = int(ea.size(-1))
            ea = ea.long()
    else:
        edge_type_dim = 1  # 不用，但占位
        ea = None

    edge_type = torch.full((num_nodes, num_nodes, edge_type_dim), -1, dtype=torch.long)

    # 直接用 numpy 加速循环
    row_np = row.cpu().numpy()
    col_np = col.cpu().numpy()
    for idx in range(row_np.shape[0]):
        u = int(row_np[idx]); v = int(col_np[idx])
        adj_matrix[u, v] = 1
        if has_edge_attr:
            edge_type[u, v] = ea[idx]

    # 3) Numba 最短路（返回：[N,N] 与 [N,N,D,Fe]）
    #    保证传入是 C-contiguous 的 np.int64
    adj_np = np.ascontiguousarray(adj_matrix.cpu().numpy().astype(np.int64))
    et_np  = np.ascontiguousarray(edge_type.cpu().numpy().astype(np.int64))
    sp_np, edge_input_np = bfs_numba_spatial_pos_and_edge_input(
        adj_np, et_np, int(distance)
    )

    # 4) 打平成可 collate 的三元组
    # 4.1 索引对 [2, M]
    idx = torch.arange(num_nodes, dtype=torch.long)
    ii = idx.repeat_interleave(num_nodes)
    jj = idx.repeat(num_nodes)
    graph_index = torch.stack([ii, jj], dim=0)  # [2, M], M=N*N

    # 4.2 距离 [M]，可选裁剪到 distance 以内，避免超界
    spatial_types = torch.from_numpy(sp_np.reshape(-1).astype(np.int64))
    spatial_types = spatial_types.clamp_max(int(distance))

    data.graph_index = graph_index
    data.spatial_types = spatial_types

    # 5) shortest_path_types: 压回二维 [M, D]，负值→0（Embedding padding）
    if has_edge_attr:
        # edge_input_np: [N, N, D, Fe] -> [M, D, Fe]
        N, _, D, Fe = edge_input_np.shape
        spt = edge_input_np.reshape(N * N, D, Fe).astype(np.int64)

        if Fe == 1:
            spt = spt[..., 0]  # [M, D]
        else:
            # 最稳妥：沿 Fe 聚合（sum/mean 均可）→ 单一 id 空间
            # 若你想与“单通道 id”完全一致，也可以只取 spt[..., 0]
            spt = spt.sum(axis=-1)  # [M, D]

        # 负值→0 作为 padding
        spt = np.maximum(spt, 0)
        data.shortest_path_types = torch.from_numpy(spt)
    # 若无边特征：不写入 shortest_path_types，BiasEncoder 会只用 spatial 部分

    return data



@torch.jit.script
def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    """把多维离散特征映射到统一 embedding 空间（与 Graphormer 官方一致）"""
    feature_num = x.size(1) if x.dim() > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    return x + feature_offset

def graphormer_pre_processing_NEW(data, distance: int):
    """
    对齐 Graphormer 官方 preprocess 输出的预处理（Numba BFS 版）：
      产出字段（全部在 CPU 上）：
        - data.x              : 节点离散特征 id，已 convert_to_single_emb（形状 [N, A] 或 [N, 1]）
        - data.in_degree      : [N]  long
        - data.out_degree     : [N]  long
        - data.attn_bias      : [N+1, N+1] float32（含 graph token）
        - data.spatial_pos    : [N, N]     long，最短路，已 clamp 到 max_dist
        - data.attn_edge_type : [N, N, Fe] long，边类型 id（convert_to_single_emb(edge_attr)+1）
        - data.edge_input     : [N, N, D, Fe] long，多跳路径边类型序列（负值→0 用作 padding）
    说明：
        - 这里保留 edge_input 的 Fe 维度，以适配 Graphormer 官方 multi-hop 路径
          （GraphAttnBias 里会对 Fe 做 embedding 再沿 Fe 做 mean）。
    """
    N = int(data.num_nodes)
    row, col = data.edge_index  # [2, E]

    # 1) 度数（更快：bincount）
    in_deg  = torch.bincount(col, minlength=N)
    out_deg = torch.bincount(row, minlength=N)
    data.in_degree = in_deg.to(torch.long)
    data.out_degree = out_deg.to(torch.long)

    if cfg.posenc_GraphormerBias.node_degrees_only:
        return data

    # 2) 节点特征：convert_to_single_emb（确保整型 & 2D）
    if getattr(data, "x", None) is not None:
        if data.x.dim() == 1:
            data.x = data.x.view(-1, 1)
        data.x = convert_to_single_emb(data.x.to(torch.long))

    # 3) 构建邻接与边类型（NumPy 上构建，传给 Numba）
    row_np = row.cpu().numpy()
    col_np = col.cpu().numpy()

    # 邻接矩阵：int64（与 numba 侧一致，避免隐式转换）
    adj_np = np.zeros((N, N), dtype=np.int64)
    adj_np[row_np, col_np] = 1

    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        ea = edge_attr
        if ea.dim() == 1:
            ea = ea[:, None]
        # 先在 torch 中 convert，再转 numpy（保持 id 对齐，并 +1）
        ea_id = (convert_to_single_emb(ea.to(torch.long)) + 1).cpu().numpy()  # [E, Fe]
        Fe = int(ea_id.shape[1])
        attn_edge_type_np = np.zeros((N, N, Fe), dtype=np.int64)
        # 直接一次性赋值（避免列循环）
        attn_edge_type_np[row_np, col_np] = ea_id
    else:
        Fe = 1
        attn_edge_type_np = np.zeros((N, N, 1), dtype=np.int64)

    # 4) Numba BFS：最短路 + 多跳边类型（保留 4D Fe 维）
    sp_np, edge_input_np = bfs_numba_spatial_pos_and_edge_input(
        np.ascontiguousarray(adj_np),
        np.ascontiguousarray(attn_edge_type_np),
        distance
    )
    # 现在：
    #   sp_np         : [N, N]
    #   edge_input_np : [N, N, D, Fe]   （D = max_dist；或实现里与你的 get_full_path 约定）

    # 5) 回写到 Data（一次性转换；负值→0 作为 padding）
    data.spatial_pos    = torch.from_numpy(sp_np).long().clamp_max(distance)        # [N, N]
    ei = torch.from_numpy(edge_input_np).long()
    ei = torch.clamp_min(ei, 0)  # 负值→0
    data.edge_input     = ei                                                                          # [N, N, D, Fe]
    data.attn_edge_type = torch.from_numpy(attn_edge_type_np).long()                                  # [N, N, Fe]
    data.attn_bias      = torch.zeros((N + 1, N + 1), dtype=torch.float32)                            # [N+1, N+1]

    return data

#NEW搭配新的collector
# pair_store_ragged.py
from typing import List, Dict, Any, Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater

DENSE_KEYS = ("attn_bias", "spatial_pos", "attn_edge_type", "edge_input")

def _flatten_dense_sample(d: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """
    返回：
      flat_ab, flat_sp, flat_aet, flat_ei （均为 1D contiguous Tensor）
      N, Fe, D
    允许任意一个键缺失（返回空 Tensor）
    """
    N = int(d.num_nodes)
    ab  = getattr(d, "attn_bias", None)        # [N+1, N+1] float
    sp  = getattr(d, "spatial_pos", None)      # [N, N] long
    aet = getattr(d, "attn_edge_type", None)   # [N, N, Fe] long
    ei  = getattr(d, "edge_input", None)       # [N, N, D] long

    Fe = int(aet.size(-1)) if (aet is not None and aet.dim() == 3) else 1
    D  = int(ei.size(-1))  if (ei  is not None and ei.dim()  == 3) else 1

    flat_ab  = ab.reshape(-1).contiguous() if ab  is not None else torch.empty(0, dtype=torch.float32)
    flat_sp  = sp.reshape(-1).contiguous() if sp  is not None else torch.empty(0, dtype=torch.long)
    flat_aet = aet.reshape(-1).contiguous() if aet is not None else torch.empty(0, dtype=torch.long)
    flat_ei  = ei.reshape(-1).contiguous() if ei  is not None else torch.empty(0, dtype=torch.long)
    return flat_ab, flat_sp, flat_aet, flat_ei, N, Fe, D

def build_pair_store_ragged(dataset, data_list: List[Data]) -> Dict[str, Any]:
    """
    - 把四个方阵键扁平化拼成“大连续存储” + offsets（前缀和）
    - 把样本的 N/Fe/D 记录下来
    - 从 Data 中删除这四个键（避免 InMemoryDataset.collate 报错）
    - 返回 pair_store（字典）；并把其挂到 dataset 上
    """
    ab_list, sp_list, aet_list, ei_list = [], [], [], []
    Ns, Fes, Ds = [], [], []
    sample_idx_tensors = []

    # 扁平化收集
    for i, d in enumerate(data_list):
        d.sample_idx = torch.tensor([i], dtype=torch.long)  # 保证存在
        sample_idx_tensors.append(d.sample_idx)

        flat_ab, flat_sp, flat_aet, flat_ei, N, Fe, D = _flatten_dense_sample(d)
        ab_list.append(flat_ab); sp_list.append(flat_sp)
        aet_list.append(flat_aet); ei_list.append(flat_ei)
        Ns.append(N); Fes.append(Fe); Ds.append(D)

        # 删掉原字段，避免 InMemoryDataset 拼接它们
        for k in DENSE_KEYS:
            if hasattr(d, k):
                delattr(d, k)

    # 前缀和 offsets（长度=样本数+1）
    def _cat_with_offsets(lst: List[torch.Tensor], dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        sizes = [t.numel() for t in lst]
        offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
        if sizes:
            offsets[1:] = torch.cumsum(torch.tensor(sizes, dtype=torch.long), dim=0)
        if offsets[-1].item() == 0:
            return torch.empty(0, dtype=dtype), offsets
        buf = torch.empty(int(offsets[-1]), dtype=dtype)
        pos = 0
        for t in lst:
            n = t.numel()
            if n:
                buf[pos:pos+n] = t.to(dtype)
                pos += n
        return buf, offsets

    ab_buf,  ab_off  = _cat_with_offsets(ab_list,  torch.float32)
    sp_buf,  sp_off  = _cat_with_offsets(sp_list,  torch.long)
    aet_buf, aet_off = _cat_with_offsets(aet_list, torch.long)
    ei_buf,  ei_off  = _cat_with_offsets(ei_list,  torch.long)

    # 样本元信息
    Ns = torch.tensor(Ns, dtype=torch.long)
    Fes = torch.tensor(Fes, dtype=torch.long)
    Ds = torch.tensor(Ds, dtype=torch.long)

    pair_store = {
        "ab_buf": ab_buf,   "ab_off": ab_off,
        "sp_buf": sp_buf,   "sp_off": sp_off,
        "aet_buf": aet_buf, "aet_off": aet_off,
        "ei_buf": ei_buf,   "ei_off": ei_off,
        "Ns": Ns, "Fes": Fes, "Ds": Ds,
    }

    # 让 InMemoryDataset 正常 collate 剩余键（含 sample_idx）
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

    # 方便 DataLoader 的 collate 使用
    dataset._pair_store = pair_store
    dataset._dense_keys = DENSE_KEYS
    return pair_store

def make_collate_from_ragged(pair_store: dict):
    """
    根据 ragged pair_store 生成 collate_fn：
      - 先用 PyG Collater 合并普通键
      - 再用 offsets 从大缓冲切片，view 成 [N,N,*]，按本 batch 的 Nmax/Fe/D 做一次性 padding
    """
    ab_buf,  ab_off  = pair_store["ab_buf"],  pair_store["ab_off"]
    sp_buf,  sp_off  = pair_store["sp_buf"],  pair_store["sp_off"]
    aet_buf, aet_off = pair_store["aet_buf"], pair_store["aet_off"]
    ei_buf,  ei_off  = pair_store["ei_buf"],  pair_store["ei_off"]
    Ns, Fes, Ds = pair_store["Ns"], pair_store["Fes"], pair_store["Ds"]

    def collate_fn(batch_list: List[Data]):
        collater = Collater(follow_batch=[])
        batch = collater(batch_list)

        idxs = [int(d.sample_idx.item()) for d in batch_list]
        Ns_b  = Ns[idxs]
        Fes_b = Fes[idxs]
        Ds_b  = Ds[idxs]

        B     = len(batch_list)
        Nmax  = int(Ns_b.max().item()) if B > 0 else 1
        Fe    = int(Fes_b.max().item()) if B > 0 else 1
        D     = int(Ds_b.max().item())  if B > 0 else 1

        attn_bias      = torch.zeros(B, Nmax+1, Nmax+1, dtype=torch.float32)
        spatial_pos    = torch.zeros(B, Nmax,   Nmax,   dtype=torch.long)
        attn_edge_type = torch.zeros(B, Nmax,   Nmax,   Fe, dtype=torch.long)
        edge_input     = torch.zeros(B, Nmax,   Nmax,   D,  dtype=torch.long)
        node_mask           = torch.zeros(B, Nmax,   dtype=torch.bool)
        node_with_tok_mask  = torch.zeros(B, Nmax+1, dtype=torch.bool)

        for i, idx in enumerate(idxs):
            N  = int(Ns[idx].item())
            Fe_i = int(Fes[idx].item())
            D_i  = int(Ds[idx].item())

            # 1) attn_bias
            lo, hi = int(ab_off[idx].item()), int(ab_off[idx+1].item())
            if hi > lo:
                ab_flat = ab_buf[lo:hi]
                attn_bias[i, :N+1, :N+1] = ab_flat.view(N+1, N+1)

            # 2) spatial_pos
            lo, hi = int(sp_off[idx].item()), int(sp_off[idx+1].item())
            if hi > lo:
                sp_flat = sp_buf[lo:hi]
                spatial_pos[i, :N, :N] = sp_flat.view(N, N)

            # 3) attn_edge_type
            lo, hi = int(aet_off[idx].item()), int(aet_off[idx+1].item())
            if hi > lo:
                aet_flat = aet_buf[lo:hi]
                attn_edge_type[i, :N, :N, :Fe_i] = aet_flat.view(N, N, Fe_i)

            # 4) edge_input
            lo, hi = int(ei_off[idx].item()), int(ei_off[idx+1].item())
            if hi > lo:
                ei_flat = ei_buf[lo:hi]
                edge_input[i, :N, :N, :D_i] = ei_flat.view(N, N, D_i)

            node_mask[i, :N] = True
            node_with_tok_mask[i, :N+1] = True


        batch.attn_bias = attn_bias.contiguous()
        batch.spatial_pos = spatial_pos.contiguous()
        batch.attn_edge_type = attn_edge_type.contiguous()
        batch.edge_input = edge_input.contiguous()
        batch.node_mask = node_mask
        batch.node_with_tok_mask = node_with_tok_mask
        batch.num_nodes_list = Ns_b
        batch._pad_info = {"Nmax": Nmax, "Fe": Fe, "D": D}
        return batch

    return collate_fn