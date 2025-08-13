# side_channel.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from GTBenchmark.graphgym.config import cfg
from torch import Tensor
from torch_geometric.data import Batch, Data

# 优先用你仓库里的 Numba；没有再尝试 Cython；都没有就报错
try:
    from GTBenchmark.transform.gf_algos import algos_numba as nb_algos
    HAS_NUMBA = True
except Exception:
    nb_algos = None
    HAS_NUMBA = False

try:
    from GTBenchmark.transform.gf_algos import algos as cy_algos
    HAS_CYTHON = True
except Exception:
    cy_algos = None
    HAS_CYTHON = False


# -------- 通用旁路容器 --------
@dataclass
class SideBatch:
    """不参与 PyG collate 的 batch 级稠密张量容器。"""
    tensors: Dict[str, Tensor]   # e.g. {'spatial_pos': [B,N,N], 'attn_edge_type':[B,N,N,Fe], 'edge_input':[B,N,N,D,Fe]}
    sizes: List[int]             # 每图节点数 [n1, n2, ...]
    Nmax: int                    # 本 batch 的最大 N

    def to(self, device=None, dtype=None):
        self.tensors = {k: (v.to(device=device, dtype=dtype) if (device or dtype) else v)
                        for k, v in self.tensors.items()}
        return self

    def __contains__(self, k): return k in self.tensors
    def __getitem__(self, k):  return self.tensors[k]


def attach_side(batch: Batch, side: SideBatch) -> Batch:
    """把 SideBatch 掛到 Batch 上，供后续模块读取。"""
    batch._side = side
    return batch


# -------- Graphormer：把离散多列合并为单列 ID（与官方一致）--------
@torch.jit.script
def _convert_to_single_emb(x: Tensor, offset: int = 512) -> Tensor:
    feature_num: int = x.size(1) if x.dim() > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    return x.long() + feature_offset


def _build_adj_and_aetype(d: Data, undirected: bool = True, NoNeedPlus:bool = False) -> Tuple[Tensor, Tensor]:
    """返回 adj:[N,N] bool；attn_edge_type:[N,N,Fe] long（convert_to_single_emb+1）"""
    assert d.edge_index is not None, "edge_index is required"
    N = int(d.num_nodes)

    # 1) 节点特征 -> 单向量离散编码（官方做法）
    if getattr(d, "x", None) is None:
        # 没有 x 就按 1 维哑变量处理（全 0），保证后续 Embedding 不崩
        d.x = torch.zeros((N, 1), dtype=torch.long)
    x = d.x
    if x.dim() == 1:
        x = x.unsqueeze(1)
    x = _convert_to_single_emb(x)  # (N, F_x) long
    d.x = x

    # 2) 构造布尔邻接（必要时对称化）
    ei = d.edge_index
    adj = torch.zeros((N, N), dtype=torch.bool)
    adj[ei[0], ei[1]] = True
    if undirected:
        adj = adj | adj.t()

    # 边特征
    if getattr(d, "edge_attr", None) is None:
        edge_attr = torch.zeros((ei.size(1), 1), dtype=torch.long)
    else:
        edge_attr = d.edge_attr.long()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
    if NoNeedPlus:
        ea_single = _convert_to_single_emb(edge_attr)  # (E, Fe)
    else:
        ea_single = _convert_to_single_emb(edge_attr) + 1  # (E, Fe)
    Fe = int(ea_single.size(-1))
    aetype = torch.zeros((N, N, Fe), dtype=torch.long)
    aetype[ei[0], ei[1]] = ea_single
    if undirected:
        aetype[ei[1], ei[0]] = ea_single
    return adj, aetype


def _sp_edgeinput_single(adj: Tensor, aetype: Tensor, max_dist: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 spatial_pos:[N,N] int64（不可达=510），edge_input:[N,N,D,Fe] int64（D=max_dist）"""
    N = int(adj.size(0))
    adj_np = np.ascontiguousarray(adj.cpu().numpy(), dtype=np.int64)
    aetype_np = np.ascontiguousarray(aetype.cpu().numpy(), dtype=np.int64)

    if HAS_NUMBA:
        sp_np, edge_input_np = nb_algos.bfs_numba_spatial_pos_and_edge_input(adj_np, aetype_np, int(max_dist))
        return sp_np, edge_input_np

    if HAS_CYTHON:
        sp_np, path_np = cy_algos.floyd_warshall(adj_np)
        D = min(int(np.amax(sp_np)), int(max_dist))
        edge_input_np = cy_algos.gen_edge_input(D, path_np, aetype_np)
        if D < max_dist:  # pad 到固定 D，pad=-1
            pad = np.full((N, N, max_dist - D, aetype_np.shape[-1]), -1, dtype=np.int64)
            edge_input_np = np.concatenate([edge_input_np, pad], axis=2)
        return sp_np, edge_input_np

    raise RuntimeError("No shortest-path backend (numba/cython) available.")


def _pad_stack_nd(items: List[Tensor], Nmax: int, pad_value: int) -> Tensor:
    """将若干 (Ni,Ni,...) 张量按 Nmax pad 成 (B,Nmax,Nmax,...)。"""
    B = len(items)
    assert B > 0
    tail = tuple(items[0].shape[2:])
    dtype = items[0].dtype
    device = items[0].device
    out = torch.full((B, Nmax, Nmax, *tail), pad_value, dtype=dtype, device=device)
    for b, t in enumerate(items):
        n = t.size(0)
        assert t.size(1) == n, f"expect square [N,N,...], got {tuple(t.shape)}"
        assert tuple(t.shape[2:]) == tail, "inconsistent tail dims"
        out[b, :n, :n, ...] = t
    return out


def collate_graphormer_with_side(
    items: List[Data],
    max_spatial_dist: int = 15,
    undirected: bool = True,
) -> Tuple[Batch, SideBatch]:
    """
    Graphormer 专用 collate：
      - 每张图现算 spatial_pos / attn_edge_type / edge_input（不写回 Data）
      - 同时补 in_degrees/out_degrees（写回 Data，因其是 [N]，PyG 可安全 collate）
      - 返回 (Batch, SideBatch)
    """
    B = len(items)
    Ns = [int(d.num_nodes) for d in items]
    Nmax = max(Ns)

    sp_list: List[Tensor] = []
    ae_list: List[Tensor] = []
    ei_list: List[Tensor] = []

    Fe_ref: Optional[int] = None
    D_ref: Optional[int] = None

    stripped: List[Data] = []
    for d in items:
        # 度（[N]，可安全交给 PyG collate）
        N = int(d.num_nodes)
        in_deg = torch.zeros(N, dtype=torch.long)
        out_deg = torch.zeros(N, dtype=torch.long)
        ones = torch.ones(d.edge_index.size(1), dtype=torch.long)
        in_deg.index_add_(0, d.edge_index[1], ones)
        out_deg.index_add_(0, d.edge_index[0], ones)
        if undirected:
            in_deg = out_deg = (in_deg + out_deg) // 2
        d = d.clone()
        d.in_degrees = in_deg
        d.out_degrees = out_deg
        if d.edge_attr.min() == 0:
            print("!")
        # NxN 三件套（仅留在本函数，不写回 Data）
        adj, aetype = _build_adj_and_aetype(d, undirected=undirected,NoNeedPlus=True)
        sp_np, ei_np = _sp_edgeinput_single(adj, aetype, max_spatial_dist)

        Fe = int(aetype.size(-1))
        D = int(ei_np.shape[2])
        if Fe_ref is None: Fe_ref = Fe
        if D_ref is None:  D_ref  = D
        assert Fe_ref == Fe, f"Edge feature dim mismatch: {Fe} vs {Fe_ref}"
        assert D_ref  == D,  f"Max spatial depth mismatch: {D} vs {D_ref}"

        sp_list.append(torch.from_numpy(sp_np).long())  # [N,N]
        ae_list.append(aetype.long())                   # [N,N,Fe]
        ei_list.append(torch.from_numpy(ei_np).long())  # [N,N,D,Fe]
        stripped.append(d)

    Fe = int(Fe_ref) if Fe_ref is not None else 1
    D  = int(D_ref)  if D_ref  is not None else int(max_spatial_dist)

    # pad → batch
    spatial_pos   = _pad_stack_nd(sp_list, Nmax, pad_value=0)                # [B,N,N]
    attn_edge_type= _pad_stack_nd(ae_list, Nmax, pad_value=0)                # [B,N,N,Fe]
    edge_input    = _pad_stack_nd(ei_list, Nmax, pad_value=-1)               # [B,N,N,D,Fe]

    side = SideBatch(
        tensors={
            "spatial_pos": spatial_pos.squeeze(-1) if spatial_pos.dim()==4 and spatial_pos.size(-1)==1 else spatial_pos[..., 0] if spatial_pos.dim()==4 and spatial_pos.size(-1)==1 else spatial_pos,
            "attn_edge_type": attn_edge_type,
            "edge_input": edge_input,
        },
        sizes=Ns,
        Nmax=Nmax,
    )

    batch = Batch.from_data_list(stripped)
    return batch, side
