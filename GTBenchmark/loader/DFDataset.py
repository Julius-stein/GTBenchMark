
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
from torch_geometric.data import Data, Batch
from GTBenchmark.graphgym.config import cfg

class MetaInfo:
    def __init__(
        self,
        name: str,
        kind: str,                  # "dense" | "edge_list"
        dtype: torch.dtype,
        shape_rule: Optional[Callable[[int, Dict[str, Any]], Tuple[int, ...]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.kind = kind
        self.dtype = dtype
        self.shape_rule = shape_rule   # (N, extra) -> target shape (允许 -1)
        self.extra = extra or {}

    def __repr__(self) -> str:
        return f"MetaInfo(name={self.name}, kind={self.kind}, dtype={self.dtype}, rule={self.shape_rule})"


def _get_D_from_cfg() -> int:
    # D = cfg.posenc_GraphormerBias.multi_hop_max_dist （缺省回退为 1）
    try:
        return int(getattr(cfg.posenc_GraphormerBias, "multi_hop_max_dist"))
    except Exception:
        return 1


# —— 仅这些 key 会被抽出成 side data（按需增减）——
_SIDE_META: Dict[str, MetaInfo] = {
    # Graphormer: (N+1, N+1)
    "attn_bias": MetaInfo(
        "attn_bias", kind="dense", dtype=torch.float32,
        shape_rule=lambda N, _: (N + 1, N + 1),
    ),
    # Graphormer: (N, N)
    "spatial_pos": MetaInfo(
        "spatial_pos", kind="dense", dtype=torch.long,
        shape_rule=lambda N, _: (N, N),
    ),
    # Graphormer: (N, N, Fe) —— Fe 不固定，用 -1 自适应
    "attn_edge_type": MetaInfo(
        "attn_edge_type", kind="dense", dtype=torch.long,
        shape_rule=lambda N, extra: (N, N, -1),
    ),
    # Graphormer: (N, N, D, Fe) —— D 固定来自 cfg，Fe 用 -1 自适应
    "edge_input": MetaInfo(
        "edge_input", kind="dense", dtype=torch.long,
        shape_rule=lambda N, extra: (N, N, extra["D"], -1),
        extra={"D": _get_D_from_cfg()},
    ),
    # NodeFormer: list of edge_index（每 hop：[2, Eh]）
    "adjs": MetaInfo(
        "adjs", kind="edge_list", dtype=torch.long, shape_rule=None,
    ),
}


def register_side_meta(name: str, meta: MetaInfo):
    """外部新增 side data 时调用：明确 key、kind、dtype 和 shape_rule（可含 -1）"""
    _SIDE_META[name] = meta


# ============================================================
# Dense-First Batch (Intermediate Representation)
# ============================================================

class DenseFirstBatch:
    """
    Dense-First batch intermediate representation (IR).

    Attributes:
        x          : Tensor [B, M, F]
        num_nodes  : Tensor [B]
        edge_index : list[Tensor [2, Ei]]
        edge_attr  : list[Tensor [Ei, ...]] | None
        y          : Tensor [B, ...] | None
    """

    def __init__(
        self,
        x: torch.Tensor,
        num_nodes: torch.Tensor,
        # edge_index: List[torch.Tensor],
        # edge_attr: Optional[List[torch.Tensor]],
        y: Optional[torch.Tensor],
    ):
        self.x = x
        self.num_nodes = num_nodes
        # self.edge_index = edge_index
        # self.edge_attr = edge_attr
        self.y = y

    # --------------------------------------------------------
    # Device transfer
    # --------------------------------------------------------

    def to(self, device,non_blocking=True):
        self.x = self.x.to(device,non_blocking=non_blocking)
        self.num_nodes = self.num_nodes.to(device,non_blocking=non_blocking)

        if self.y is not None:
            self.y = self.y.to(device,non_blocking=non_blocking)

        # self.edge_index = [ei.to(device) for ei in self.edge_index]

        # if self.edge_attr is not None:
            # self.edge_attr = [ea.to(device) for ea in self.edge_attr]

        return self
    
    def key_padding_mask(self):
        """
        Returns key padding mask of shape [B, M]
        True means masked (invalid node)
        """
        B, M = self.x.size(0), self.x.size(1)
        device = self.x.device

        arange = torch.arange(M, device=device)
        return arange[None, :] >= self.num_nodes[:, None]

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.num_nodes = self.num_nodes.pin_memory()
        if self.y is not None:
            self.y = self.y.pin_memory()
        return self
    # --------------------------------------------------------
    # (Optional) Convert to PyG Batch (graph-level)
    # --------------------------------------------------------
    def to_pyg_batch(self):
        """
        Convert DenseFirstBatch to torch_geometric.data.Batch.
        Intended for GNN backends (MessagePassing).
        """
        from torch_geometric.data import Data, Batch

        data_list = []
        B = self.x.size(0)

        for i in range(B):
            n = int(self.num_nodes[i].item())
            data = Data(
                x=self.x[i, :n],
                edge_index=self.edge_index[i],
                edge_attr=None if self.edge_attr is None else self.edge_attr[i],
                y=None if self.y is None else self.y[i],
            )
            data_list.append(data)

        return Batch.from_data_list(data_list)


# ============================================================
# Dense-First Dataset (runtime, no persistence)
# ============================================================
class DenseFirstDataset(torch.utils.data.Dataset):
    """
    Runtime dataset that stores raw PyG Data objects.
    No padding, no materialization.
    """

    def __init__(self, ctx, data_list):
        self.ctx = ctx
        self.data_list = data_list
        self.index_tensors = self._load_split_indices(ctx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        return {
            "data": d
        }

    # --------------------------------------------------------
    # split handling
    # --------------------------------------------------------
    def _load_split_indices(self, ctx):
        split = getattr(ctx, "split_idx", None)
        out = {}

        if split is None:
            return out

        if isinstance(split, dict):
            if "train" in split:
                out["train"] = torch.as_tensor(split["train"])
            if "valid" in split or "val" in split:
                out["val"] = torch.as_tensor(split.get("valid", split.get("val")))
            if "test" in split:
                out["test"] = torch.as_tensor(split["test"])

        elif isinstance(split, (list, tuple)):
            out["train"] = torch.as_tensor(split[0])
            out["val"]   = torch.as_tensor(split[1])
            out["test"]  = torch.as_tensor(split[2])

        return out


# class DenseFirstDataset(Dataset):
#     r"""
#     Dense-First runtime dataset.

#     Materializes a PyG-style dataset into a dense-first layout:

#     - x          : [G, M, F]
#     - num_nodes  : [G]
#     - edge_index : list[Tensor]
#     - edge_attr  : list[Tensor] | None
#     - y          : [G, ...] | None
#     - index_tensors : dataset-level splits (train/val/test)

#     Notes:
#         - Positional encodings (PE) must be applied beforehand
#         - No save/load (runtime-only, training-time layout)
#     """
    
#     def __init__(self,ctx,data_list:List):
#         self.ctx = ctx
#         self.data_list = data_list 
#         self._materialize()

#     # --------------------------------------------------------
#     # Core materialization
#     # --------------------------------------------------------

#     def _materialize(self):
#         data_list = self.data_list
#         G = len(data_list)

#         # ---- node features (dense-first) ----
#         self.num_nodes = torch.tensor(
#             [d.num_nodes for d in data_list], dtype=torch.long
#         )
#         M = int(self.num_nodes.max().item())

#         x0 = data_list[0].x
#         F = x0.size(-1)

#         self.x = torch.zeros((G, M, F), dtype=x0.dtype)
#         for i, d in enumerate(data_list):
#             self.x[i, : d.num_nodes] = d.x

#         # ---- labels ----
#         self.y = None
#         if hasattr(data_list[0], "y") and data_list[0].y is not None:
#             try:
#                 self.y = torch.stack([d.y for d in data_list])
#             except Exception:
#                 self.y = None

#         # ---- structure (ragged on purpose) ----
#         # self.edge_index = [d.edge_index for d in data_list]

#         # self.edge_attr = None
#         # if hasattr(data_list[0], "edge_attr") and data_list[0].edge_attr is not None:
#             # self.edge_attr = [d.edge_attr for d in data_list]

#         # ---- dataset-level split indices ----
#         self.index_tensors = self._load_split_indices(self.ctx)

#     # --------------------------------------------------------
#     # Split handling (robust to PyG / OGB styles)
#     # --------------------------------------------------------

#     def _load_split_indices(self, ctx) -> Dict[str, torch.Tensor]:
#         split = None
#         if hasattr(ctx, "split_idx"):
#             split = ctx.split_idx

#         index_tensors: Dict[str, torch.Tensor] = {}

#         if split is None:
#             return index_tensors

#         # dict-style splits
#         if isinstance(split, dict):
#             if "train" in split:
#                 index_tensors["train"] = torch.as_tensor(split["train"])
#             if "valid" in split:
#                 index_tensors["val"] = torch.as_tensor(split["valid"])
#             elif "val" in split:
#                 index_tensors["val"] = torch.as_tensor(split["val"])
#             if "test" in split:
#                 index_tensors["test"] = torch.as_tensor(split["test"])

#         # list/tuple-style splits
#         elif isinstance(split, (list, tuple)):
#             assert len(split) == 3, "split_idxs must be [train, val, test]"
#             index_tensors["train"] = torch.as_tensor(split[0])
#             index_tensors["val"]   = torch.as_tensor(split[1])
#             index_tensors["test"]  = torch.as_tensor(split[2])

#         else:
#             raise TypeError(f"Unsupported split type: {type(split)}")

#         return index_tensors

#     # --------------------------------------------------------
#     # Dataset interface
#     # --------------------------------------------------------

#     def __len__(self) -> int:
#         return self.x.size(0)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         return {
#             "x": self.x[idx],                     # [M, F]
#             "num_nodes": self.num_nodes[idx],     # scalar
#             # "edge_index": self.edge_index[idx],   # [2, E]
#             # "edge_attr": None if self.edge_attr is None else self.edge_attr[idx],
#             "y": None if self.y is None else self.y[idx],
#         }


# ============================================================
# Collate function
# ============================================================

# def densefirst_collate(batch_list: List[Dict[str, Any]]) -> DenseFirstBatch:
#     """
#     Collate function for DenseFirstDataset.

#     Args:
#         batch_list: list of samples from DenseFirstDataset.__getitem__

#     Returns:
#         DenseFirstBatch
#     """

#     x = torch.stack([b["x"] for b in batch_list], dim=0)
#     num_nodes = torch.stack([b["num_nodes"] for b in batch_list], dim=0)

#     y = None
#     if batch_list[0]["y"] is not None:
#         y = torch.stack([b["y"] for b in batch_list], dim=0)

#     # edge_index = [b["edge_index"] for b in batch_list]

#     # edge_attr = None
#     # if batch_list[0]["edge_attr"] is not None:
#     #     edge_attr = [b["edge_attr"] for b in batch_list]

#     return DenseFirstBatch(
#         x=x,
#         num_nodes=num_nodes,
#         # edge_index=edge_index,
#         # edge_attr=edge_attr,
#         y=y,
#     )
def densefirst_collate(batch_list):
    """
    Materialize DenseFirst batch *inside collate*.

    batch_list: List[{"data": PyG.Data}]
    """

    data_list = [b["data"] for b in batch_list]

    B = len(data_list)
    num_nodes = torch.tensor([d.num_nodes for d in data_list], dtype=torch.long)
    M = int(num_nodes.max().item())

    # infer feature dim
    x0 = data_list[0].x
    F = x0.size(-1)

    device = x0.device
    dtype = x0.dtype

    # ---------- materialize X ----------
    x = torch.zeros((B, M, F), dtype=dtype)

    for i, d in enumerate(data_list):
        n = d.num_nodes
        x[i, :n] = d.x

    # ---------- labels ----------
    y = None
    if hasattr(data_list[0], "y") and data_list[0].y is not None:
        try:
            y = torch.stack([d.y for d in data_list])
        except Exception:
            y = None

    return DenseFirstBatch(
        x=x,
        num_nodes=num_nodes,
        y=y,
    )
