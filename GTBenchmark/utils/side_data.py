# ============================================================
# sidedata_global.py —— key 注册 + shape_rule(-1) 驱动的通用 SideData
# ============================================================
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
from torch_geometric.data import Data, Batch
from GTBenchmark.graphgym.config import cfg


# ============================================================
# 1) Meta 注册（按 key 控制是否作为 side data）
#    kind ∈ {"dense", "edge_list"}；shape_rule 支持 -1 自适应维
# ============================================================

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
# 2) 全局对象（给 DataLoader 使用）
# ============================================================

_GLOBAL_SIDE: Optional["SideData"] = None

def set_global_side(side: "SideData") -> None:
    global _GLOBAL_SIDE
    _GLOBAL_SIDE = side

def get_global_side() -> "SideData":
    if _GLOBAL_SIDE is None:
        raise RuntimeError("Global SideData not set. Call build_and_attach_global(...) first.")
    return _GLOBAL_SIDE

def clear_global_side() -> None:
    global _GLOBAL_SIDE
    _GLOBAL_SIDE = None

def global_collate(batch_list: List[Data]):
    return get_global_side().collate(batch_list)

def build_and_attach_global(dataset, data_list: List[Data]) -> "SideData":
    side = SideData.from_data_list(data_list)
    side.attach_to(dataset, data_list=data_list)
    set_global_side(side)
    return side


# ============================================================
# 3) 通用辅助
# ============================================================

def _flatten_or_empty(t: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    if t is None:
        return torch.empty(0, dtype=dtype)
    return t.reshape(-1).to(dtype).contiguous()

def _concat_with_offsets(lst: List[torch.Tensor], dtype: torch.dtype):
    sizes = [int(x.numel()) for x in lst]
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
    if sizes:
        offsets[1:] = torch.cumsum(torch.tensor(sizes, dtype=torch.long), dim=0)
    total = int(offsets[-1])
    out = torch.empty(total, dtype=dtype)
    pos = 0
    for x in lst:
        n = int(x.numel())
        if n:
            out[pos:pos+n] = x.to(dtype)
            pos += n
    return out, offsets

def _slice_flat(buf: torch.Tensor, off: torch.Tensor, idx: int) -> torch.Tensor:
    lo = int(off[idx]); hi = int(off[idx + 1])
    return buf[lo:hi] if hi > lo else buf.new_empty(0, dtype=buf.dtype)

def _resolve_shape_with_neg1(target_shape: Tuple[int, ...], flat_numel: int) -> Tuple[int, ...]:
    """
    将 target_shape 中的 -1 维度用 flat_numel 推断出来，要求恰好整除。
    例如：flat=14580, shape=(27,27,20,-1) -> -1=14580/(27*27*20)
    """
    known_prod = 1
    neg1_count = 0
    for s in target_shape:
        if s == -1:
            neg1_count += 1
        else:
            known_prod *= s if s > 0 else 1
    if neg1_count == 0:
        if known_prod != flat_numel:
            raise RuntimeError(f"shape {target_shape} product {known_prod} != flat {flat_numel}")
        return target_shape
    if neg1_count > 1:
        raise RuntimeError(f"Only one -1 is supported in shape_rule, got {target_shape}")
    if known_prod == 0:
        raise RuntimeError(f"Invalid shape {target_shape} with zero product")
    if flat_numel % known_prod != 0:
        raise RuntimeError(f"Cannot infer -1 dim: flat {flat_numel} not divisible by known_prod {known_prod} for shape {target_shape}")
    inferred = flat_numel // known_prod
    return tuple(inferred if s == -1 else s for s in target_shape)

def _pad_dense_batch(tensors: List[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """将 list[tensor] 按各维最大值 pad 成单个 batch Tensor。"""
    if not tensors:
        return torch.empty(0, dtype=dtype)
    rank = len(tensors[0].shape)
    max_shape = [max(t.shape[i] for t in tensors) for i in range(rank)]
    out = torch.zeros(len(tensors), *max_shape, dtype=dtype)
    for i, t in enumerate(tensors):
        slices = tuple(slice(0, s) for s in t.shape)
        out[i][slices] = t
    return out


# ============================================================
# 4) SideData 主体
# ============================================================

class SideData:
    @classmethod
    def from_data_list(cls, data_list: List[Data]) -> "SideData":
    # ------------------ 新增：空数据检查 ------------------
        if not data_list or len(data_list) == 0:
            print("[SideData] Empty dataset detected, disabling shared side.")
            cfg.share.side = False
            # 返回空 SideData 对象
            return cls(
                buffers={}, offsets={}, Ns=torch.tensor([], dtype=torch.long), hop_lens={}
            )

        Ns: List[int] = []
        buffers: Dict[str, List[torch.Tensor]] = {}
        offsets: Dict[str, torch.Tensor] = {}
        list_hoplens: Dict[str, List[List[int]]] = {}
        flag = True
        for i, d in enumerate(data_list):
            if not hasattr(d, "sample_idx"):
                d.sample_idx = torch.tensor([i], dtype=torch.long)

            N = int(d.num_nodes)
            Ns.append(N)

            for key, meta in _SIDE_META.items():
                if not hasattr(d, key):
                    continue
                val = getattr(d, key)
                flag = False
                if meta.kind == "dense":
                    flat = _flatten_or_empty(val, meta.dtype)
                    buffers.setdefault(key, []).append(flat)

                elif meta.kind == "edge_list":
                    if not isinstance(val, list):
                        raise TypeError(f"{key} must be list[Tensor], got {type(val)}")
                    parts = [x.reshape(-1).to(torch.long) for x in val]
                    flat = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long)
                    hoplens = [int(p.numel()) for p in parts]
                    buffers.setdefault(key, []).append(flat)
                    list_hoplens.setdefault(key, []).append(hoplens)

                delattr(d, key)
        # ------------------ 新增：无任何 side data 的情况 ------------------
        if flag:
            print("[SideData] No side data found, disabling shared side.")
            cfg.share.side = False
            return cls(
                buffers={}, offsets={}, Ns=torch.tensor(Ns, dtype=torch.long), hop_lens={}
            )
        
        cfg.share.side = True
        # 拼接 buffer
        buf_final, off_final = {}, {}
        for key, lst in buffers.items():
            dtype = _SIDE_META[key].dtype
            buf_final[key], off_final[key] = _concat_with_offsets(lst, dtype)

        hoplens_final: Dict[str, List[List[int]]] = {}
        for key, lst in list_hoplens.items():
            hoplens_final[key] = lst

        return cls(
            buffers=buf_final,
            offsets=off_final,
            Ns=torch.tensor(Ns, dtype=torch.long),
            hop_lens=hoplens_final,
        )

    def __init__(self, *, buffers, offsets, Ns: torch.Tensor, hop_lens: Dict[str, List[List[int]]]):
        self.buffers = buffers
        self.offsets = offsets
        self.Ns = Ns
        self.hop_lens = hop_lens  # 仅 edge_list 使用

    # --------------------------------------------------------

    def attach_to(self, dataset, data_list: Optional[List[Data]] = None) -> None:
        dataset._side = self
        dataset._side_meta = _SIDE_META
        if data_list is not None:
            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

    # --------------------------------------------------------

    def collate(self, batch_list: List[Data]) -> Batch:
        batch = Batch.from_data_list(batch_list)
        idxs = [int(d.sample_idx.item()) for d in batch_list]

        for key, meta in _SIDE_META.items():
            if key not in self.buffers:
                continue

            buf, off = self.buffers[key], self.offsets[key]

            if meta.kind == "dense":
                tensors = []
                for idx in idxs:
                    flat = _slice_flat(buf, off, idx)
                    N = int(self.Ns[idx])
                    # 组合 extra（允许 meta.extra 覆盖/提供超参，例如 D）
                    extra = dict(meta.extra) if meta.extra else {}
                    # shape_rule 中的 -1 在这里用 flat.numel() 推断
                    target_shape = meta.shape_rule(N, extra) if meta.shape_rule else ()
                    if not target_shape:
                        raise RuntimeError(f"dense key '{key}' must provide shape_rule")
                    shape = _resolve_shape_with_neg1(target_shape, int(flat.numel()))
                    tensors.append(flat.view(*shape))
                batch_tensor = _pad_dense_batch(tensors, meta.dtype)
                setattr(batch, key, batch_tensor)

            elif meta.kind == "edge_list":
                restored_all: List[List[torch.Tensor]] = []
                hoplens_list = self.hop_lens.get(key, [])
                for bi, idx in enumerate(idxs):
                    flat = _slice_flat(buf, off, idx)
                    # 对应样本的 hop lens（长度不一定等于别的样本）
                    hoplens = hoplens_list[idx] if bi < len(hoplens_list) else []
                    base = 0
                    hops: List[torch.Tensor] = []
                    for ln in hoplens:
                        ln = int(ln)
                        if ln == 0:
                            hops.append(torch.empty(2, 0, dtype=torch.long))
                        else:
                            seg = flat[base: base + ln]
                            if ln % 2 != 0:
                                raise RuntimeError(f"{key}: hop flattened length must be even (2*E), got {ln}")
                            hops.append(seg.view(2, -1))
                        base += ln
                    restored_all.append(hops)
                setattr(batch, key, restored_all)

        batch.num_nodes_list = self.Ns[idxs]
        return batch
