# graph_partition.py
# ============================================================
# GraphPartitionTransformV2 — unified Dataset output
# - mode="full":   Dataset(len=1), item=full graph + patch attrs
# - mode="subgraph": Dataset(len≈n_patches), items=subgraphs
# - Supports: metis/random partition, drop_rate (partition-only),
#             num_hops expansion (overlapping membership),
#             patch-level RW-PE & diffusion, cache.
# ============================================================

import os
import random
from typing import List, Dict, Optional, Literal, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import coalesce
from torch_sparse import SparseTensor

# ---- try both python-metis & pymetis ----
_METIS_BACKEND = None
try:
    import metis as _metis_backend  # python-metis
    _METIS_BACKEND = "metis"
except Exception:
    try:
        import pymetis as _metis_backend  # pymetis
        _METIS_BACKEND = "pymetis"
    except Exception:
        _metis_backend = None
        _METIS_BACKEND = None


# -------------------- core helpers --------------------
def _to_undirected(edge_index: Tensor, num_nodes: int) -> Tensor:
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    ei, _ = coalesce(ei, None, num_nodes, num_nodes)
    return ei


def _build_adjlist(edge_index: Tensor, num_nodes: int) -> List[List[int]]:
    row, col = edge_index
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    # dedup
    return [list(set(nbs)) for nbs in adj]


def _safe_index_select(x: Optional[Tensor], idx: Tensor) -> Optional[Tensor]:
    if x is None:
        return None
    return x.index_select(0, idx)


def _is_node_level(vec: Tensor, num_nodes: int) -> bool:
    return vec.dim() >= 1 and vec.size(0) == num_nodes


# -------------------- k-hop reachability (boolean N x N) --------------------
def k_hop_reachability(edge_index: Tensor, num_nodes: int, num_hops: int) -> Tensor:
    """
    Return dense boolean matrix R [N,N]: R[i,j] = True if j within <=k hops of i (including i).
    Uses SparseTensor power/matmul; identical to your original semantics.
    """
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    eye0 = torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)
    hop_masks = [eye0]  # <=0-hop (self)
    # indicator tracks first discovered hop; not strictly needed here
    for _ in range(num_hops):
        nxt = adj.matmul(hop_masks[-1].float()) > 0
        hop_masks.append(nxt)
    # union of <=k hop
    reach = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    for m in hop_masks:
        reach |= m
    return reach  # [N,N] bool


# -------------------- partition core --------------------
def _metis_node_partition(edge_index: Tensor, num_nodes: int, nparts: int) -> Tensor:
    if _METIS_BACKEND is None:
        raise ImportError(
            "No METIS backend found. Please install `python-metis` (preferred) or `pymetis`."
        )
    adj = _build_adjlist(edge_index, num_nodes)

    if _METIS_BACKEND == "metis":
        # python-metis: metis.part_graph(adj_list, nparts=?)
        _, parts = _metis_backend.part_graph(adj, nparts=nparts)
    else:
        # pymetis: pymetis.part_graph(nparts, adjacency=adj_list)
        _, parts = _metis_backend.part_graph(nparts, adjacency=adj)

    parts = torch.tensor(parts, dtype=torch.long)
    if parts.numel() < num_nodes:
        pad = torch.full((num_nodes - parts.numel(),), nparts - 1, dtype=torch.long)
        parts = torch.cat([parts, pad], dim=0)
    return parts[:num_nodes]


def _random_node_partition(num_nodes: int, nparts: int, device=None) -> Tensor:
    return torch.randint(0, max(nparts, 1), (num_nodes,), device=device)


# -------------------- subgraph builders --------------------
def _build_subgraphs_unique(
    data: Data, node_parts: Tensor, nparts: int
) -> List[Data]:
    """
    Edge-cut with unique assignment (fast path when num_hops == 0):
    Keep only edges with endpoints in the same part; nodes unique to one subgraph.
    """
    N = data.num_nodes
    E = data.edge_index.size(1)
    row, col = data.edge_index
    same = node_parts[row] == node_parts[col]
    keep_e = torch.nonzero(same, as_tuple=False).view(-1)
    edge_part = node_parts[row[keep_e]]

    subgraphs: List[Data] = []
    for p in range(nparts):
        n_ids = torch.where(node_parts == p)[0]
        e_ids = keep_e[edge_part == p]

        g2l = -torch.ones(N, dtype=torch.long, device=row.device)
        if n_ids.numel() > 0:
            g2l[n_ids] = torch.arange(n_ids.numel(), device=row.device)

        if e_ids.numel() > 0:
            e_src = g2l[row[e_ids]]
            e_dst = g2l[col[e_ids]]
            ei = torch.stack([e_src, e_dst], dim=0)
        else:
            ei = torch.empty((2, 0), dtype=torch.long, device=row.device)

        x = _safe_index_select(getattr(data, "x", None), n_ids)
        # y: if node-level
        y = (
            _safe_index_select(getattr(data, "y", None), n_ids)
            if getattr(data, "y", None) is not None and data.y.size(0) == N
            else getattr(data, "y", None)
        )

        edge_attr = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.size(0) == E:
            edge_attr = data.edge_attr.index_select(0, e_ids)

        # map 1D node-level masks as well
        def _map_mask(name: str) -> Optional[Tensor]:
            if not hasattr(data, name):
                return None
            vec = getattr(data, name)
            if torch.is_tensor(vec) and _is_node_level(vec, N) and vec.dim() == 1:
                return vec.index_select(0, n_ids)
            return vec

        sg = Data(
            x=x,
            edge_index=ei,
            edge_attr=edge_attr,
            y=y,
            orig_n_id=n_ids,
            orig_e_id=e_ids,
        )
        for mk in ("train_mask", "val_mask", "test_mask"):
            mv = _map_mask(mk)
            if mv is not None:
                setattr(sg, mk, mv)
        subgraphs.append(sg)

    return subgraphs


def _build_subgraphs_from_masks(
    data: Data, node_masks: Tensor
) -> List[Data]:
    """
    General builder when patches can overlap (e.g., after k-hop expansion).
    node_masks: [P, N] bool, True if node belongs to patch p.
    """
    N = data.num_nodes
    E = data.edge_index.size(1)
    row, col = data.edge_index
    P = node_masks.size(0)

    subgraphs: List[Data] = []
    for p in range(P):
        n_mask = node_masks[p]  # [N]
        n_ids = torch.where(n_mask)[0]
        if n_ids.numel() == 0:
            # keep empty to preserve count
            sg = Data(
                x=torch.empty_like(getattr(data, "x", torch.empty(0,))),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=row.device),
                orig_n_id=n_ids,
                orig_e_id=torch.empty((0,), dtype=torch.long, device=row.device),
            )
            subgraphs.append(sg)
            continue

        # edges whose endpoints both in this patch
        e_mask = n_mask[row] & n_mask[col]
        e_ids = torch.where(e_mask)[0]

        g2l = -torch.ones(N, dtype=torch.long, device=row.device)
        g2l[n_ids] = torch.arange(n_ids.numel(), device=row.device)

        if e_ids.numel() > 0:
            e_src = g2l[row[e_ids]]
            e_dst = g2l[col[e_ids]]
            ei = torch.stack([e_src, e_dst], dim=0)
        else:
            ei = torch.empty((2, 0), dtype=torch.long, device=row.device)

        x = _safe_index_select(getattr(data, "x", None), n_ids)
        y = (
            _safe_index_select(getattr(data, "y", None), n_ids)
            if getattr(data, "y", None) is not None and data.y.size(0) == N
            else getattr(data, "y", None)
        )
        edge_attr = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.size(0) == E:
            edge_attr = data.edge_attr.index_select(0, e_ids)

        def _map_mask(name: str) -> Optional[Tensor]:
            if not hasattr(data, name):
                return None
            vec = getattr(data, name)
            if torch.is_tensor(vec) and _is_node_level(vec, N) and vec.dim() == 1:
                return vec.index_select(0, n_ids)
            return vec

        sg = Data(
            x=x,
            edge_index=ei,
            edge_attr=edge_attr,
            y=y,
            orig_n_id=n_ids,
            orig_e_id=e_ids,
        )
        for mk in ("train_mask", "val_mask", "test_mask"):
            mv = _map_mask(mk)
            if mv is not None:
                setattr(sg, mk, mv)

        subgraphs.append(sg)

    return subgraphs


# -------------------- patch utils --------------------
def get_node_mask_from_parts(node_parts: Tensor, n_parts: int) -> Tensor:
    num_nodes = node_parts.numel()
    mask = torch.zeros((n_parts, num_nodes), dtype=torch.bool, device=node_parts.device)
    mask[node_parts, torch.arange(num_nodes, device=node_parts.device)] = True
    return mask


def get_coarsened_adj_from_node_mask(node_mask: Tensor) -> Tensor:
    A = node_mask.float() @ node_mask.float().t()
    A.fill_diagonal_(0)
    return A  # [P,P]


def random_walk_powers(A: Tensor, k: int) -> Tensor:
    if k <= 0:
        return torch.empty((A.size(0), 0), dtype=A.dtype, device=A.device)
    deg = A.sum(dim=-1).clamp(min=1)
    RW = A / deg.unsqueeze(-1)
    M = RW
    pe = [torch.diag(M)]
    for _ in range(k - 1):
        M = M @ RW
        pe.append(torch.diag(M))
    return torch.stack(pe, dim=-1)  # [P,k]


def _pad_patch_index_lists(node_masks: Tensor, num_nodes: int) -> Tensor:
    """
    Turn boolean [P,N] into padded index table [P, max_len], using N as pad value (like your original).
    """
    P, N = node_masks.size()
    lists = [torch.where(node_masks[p])[0] for p in range(P)]
    max_len = max((len(l) for l in lists), default=0)
    if max_len == 0:
        return torch.full((P, 0), num_nodes, dtype=torch.long, device=node_masks.device)
    rows = []
    for l in lists:
        if len(l) < max_len:
            rows.append(torch.cat([l, torch.full((max_len - len(l),), num_nodes, device=node_masks.device)]))
        else:
            rows.append(l)
    return torch.stack(rows, dim=0)  # [P, max_len]


# -------------------- unified Dataset wrapper --------------------
class GraphPartitionDataset(InMemoryDataset):
    """
    Holds either:
    - len=1, item=full graph with patch-level attributes (mode='full'), or
    - len≈P, items=subgraphs (mode='subgraph').
    `meta` contains 'node_parts' and bookkeeping.
    """
    def __init__(self, datalist: List[Data], meta: Dict):
        super().__init__(".")
        self.data, self.slices = self.collate(datalist)
        self.meta = meta

    def len(self):
        if self.slices is not None:
            any_key = next(iter(self.slices))
            return int(self.slices[any_key].numel() - 1)
        else:
            return 1


# -------------------- main transform (always returns Dataset) --------------------
class GraphPartitionTransformV2:
    def __init__(
        self,
        n_patches: int,
        algo: Literal["metis", "random"] = "metis",
        cut_type: str = "edge-cut",                 # reserved for future; currently edge-cut path
        mode: Literal["full", "subgraph"] = "full", # unified Dataset output
        drop_rate: float = 0.0,                     # partition-only dropout
        num_hops: int = 0,                          # k-hop patch expansion (overlapping allowed)
        patch_rw_dim: int = 0,                      # patch-level RW-PE dim
        patch_num_diff: int = 0,                    # diffusion steps on patch-graph (RW^t)
        cache_dir: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.n_patches = n_patches
        self.algo = algo
        self.cut_type = cut_type
        self.mode = mode
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(self, data: Data) -> GraphPartitionDataset:
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

        # --- cache key ---
        cache_path = None
        if self.cache_dir:
            name = getattr(data, "name", "graph")
            cache_path = os.path.join(
                self.cache_dir, f"{name}_{self.algo}_{self.cut_type}_k{self.n_patches}_h{self.num_hops}_{self.mode}.pt"
            )

        if cache_path and os.path.exists(cache_path):
            datalist, meta = torch.load(cache_path, weights_only=False)
            return GraphPartitionDataset(datalist, meta)

        # --- undirected view for partition ---
        ei_all = _to_undirected(data.edge_index, data.num_nodes)

        # partition-only edge dropout (like your original augmentation)
        ei_for_part = ei_all
        if self.drop_rate > 0.0:
            E = ei_all.size(1)
            keep = (torch.rand(E, device=ei_all.device) > self.drop_rate)
            ei_for_part = ei_all[:, keep]

        # --- node assignment via algo ---
        if self.algo == "metis":
            node_parts = _metis_node_partition(ei_for_part, data.num_nodes, self.n_patches)
        elif self.algo == "random":
            node_parts = _random_node_partition(data.num_nodes, self.n_patches, device=data.edge_index.device)
        else:
            raise ValueError(f"Unknown algo: {self.algo}")

        meta: Dict = dict(
            algo=self.algo,
            cut_type=self.cut_type,
            node_parts=node_parts,
            n_patches=self.n_patches,
            drop_rate=self.drop_rate,
            num_hops=self.num_hops,
        )

        # --- build subgraphs list (two paths) ---
        if self.num_hops <= 0:
            # fast path: unique membership (no overlap)
            datalist = _build_subgraphs_unique(data, node_parts, self.n_patches)
            # For full-mode, attach patch-level attrs to a single Data item:
            if self.mode == "full":
                node_masks = get_node_mask_from_parts(node_parts, self.n_patches)
                coarsen_adj = get_coarsened_adj_from_node_mask(node_masks)
                full = data.clone()
                full.coarsen_adj = coarsen_adj
                if self.patch_rw_dim > 0:
                    full.patch_pe = random_walk_powers(coarsen_adj, self.patch_rw_dim)
                if self.patch_num_diff > 0:
                    # diffusion on patch graph via RW^t
                    deg = coarsen_adj.sum(-1).clamp(min=1)
                    RW = coarsen_adj / deg.unsqueeze(-1)
                    M = RW.clone()
                    for _ in range(self.patch_num_diff - 1):
                        M = M @ RW
                    full.coarsen_adj_diffused = M.unsqueeze(0)
                # patch index table (padded with N)
                full.patch = _pad_patch_index_lists(node_masks, data.num_nodes)
                # swap out datalist to length-1 dataset for full mode
                datalist = [full]

        else:
            # general path: overlapping membership via k-hop expansion
            # 1) initial mask from unique parts
            base_mask = get_node_mask_from_parts(node_parts, self.n_patches)  # [P,N] bool
            # 2) k-hop reachability on original (undirected) graph
            R = k_hop_reachability(ei_all, data.num_nodes, self.num_hops)     # [N,N] bool
            # 3) expand: node_masks = (base_mask @ R) > 0   (boolean semiring)
            node_masks = (base_mask.float() @ R.float()) > 0                   # [P,N] bool

            if self.mode == "subgraph":
                datalist = _build_subgraphs_from_masks(data, node_masks)
            else:
                # full mode: keep original full graph, attach patch-level attrs from masks
                coarsen_adj = get_coarsened_adj_from_node_mask(node_masks)
                full = data.clone()
                full.coarsen_adj = coarsen_adj
                if self.patch_rw_dim > 0:
                    full.patch_pe = random_walk_powers(coarsen_adj, self.patch_rw_dim)
                if self.patch_num_diff > 0:
                    deg = coarsen_adj.sum(-1).clamp(min=1)
                    RW = coarsen_adj / deg.unsqueeze(-1)
                    M = RW.clone()
                    for _ in range(self.patch_num_diff - 1):
                        M = M @ RW
                    full.coarsen_adj_diffused = M.unsqueeze(0)
                full.patch = _pad_patch_index_lists(node_masks, data.num_nodes)
                datalist = [full]

        ds = GraphPartitionDataset(datalist, meta)

        if cache_path:
            torch.save((datalist, meta), cache_path)

        return ds
