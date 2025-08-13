# side_channel.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

@dataclass
class SideFieldSpec:
    """定义从 Data 里抽取的字段，以及 pad 的填充值。"""
    name: str          # Data 上的属性名（单图）
    pad_value: int     # pad 的数值（e.g., spatial=0, aetype=0, edge_input=-1）

@dataclass
class SideBatch:
    """旁路 batch 容器：不参与 PyG collate。"""
    tensors: Dict[str, Tensor]   # {'spatial_pos': [B,N,N], 'edge_input':[B,N,N,D,Fe], ...}
    sizes: List[int]             # 每图节点数 [n1, n2, ...]
    Nmax: int                    # 本 batch 的 Nmax

    def to(self, device=None, dtype=None):
        self.tensors = {k: (v.to(device=device, dtype=dtype) if (device or dtype) else v)
                        for k, v in self.tensors.items()}
        return self

    def __contains__(self, k): return k in self.tensors
    def __getitem__(self, k):  return self.tensors[k]

def _pad_stack_nd(
    items: List[Tensor],
    Nmax: int,
    pad_value: int,
) -> Tensor:
    """
    将若干 (Ni,Ni,...) 的单图张量按 Nmax pad 成 (B,Nmax,Nmax,...)。
    要求所有尾维一致。
    """
    B = len(items)
    assert B > 0
    tail = tuple(items[0].shape[2:])
    dtype = items[0].dtype
    device = items[0].device
    out = torch.full((B, Nmax, Nmax, *tail), pad_value, dtype=dtype, device=device)
    for b, t in enumerate(items):
        n = t.size(0)
        assert t.size(1) == n, f"expect square [N,N,...], got {tuple(t.shape)}"
        assert tuple(t.shape[2:]) == tail, "inconsistent tail dims across graphs"
        out[b, :n, :n, ...] = t
    return out

def collate_with_side(
    data_list: List[Data],
    side_fields: Iterable[SideFieldSpec],
) -> Tuple[Batch, SideBatch]:
    """
    通用 collate：
      - 把 side_fields 指定的**单图稠密**字段从每个 Data 中取出（然后从 Data 删除）
      - 按 Nmax pad 成 batch 级稠密张量，返回 SideBatch
      - 其它字段走 PyG 的 Batch.from_data_list，返回 Batch
    """
    assert len(data_list) > 0
    Ns = [int(d.num_nodes) for d in data_list]
    Nmax = max(Ns)

    # 先收集 per-graph tensors，再从 Data 上删除它们，避免被 PyG collate
    per_field_values: Dict[str, List[Tensor]] = {}
    for spec in side_fields:
        per_field_values[spec.name] = []

    stripped: List[Data] = []
    for d in data_list:
        d = d.clone()  # 避免原对象被就地修改（可按需去掉以节省时间/内存）
        for spec in side_fields:
            val = getattr(d, spec.name, None)
            if val is None:
                raise KeyError(f"Data missing side field '{spec.name}'")
            if val.dim() < 2 or val.size(0) != val.size(1):
                raise ValueError(f"Side field '{spec.name}' must be (N,N,...) tensor, got {tuple(val.shape)}")
            per_field_values[spec.name].append(val)
            delattr(d, spec.name)
        stripped.append(d)

    # 对每个字段 pad → (B,Nmax,Nmax,...)
    side_tensors: Dict[str, Tensor] = {}
    for spec in side_fields:
        side_tensors[spec.name] = _pad_stack_nd(per_field_values[spec.name], Nmax, spec.pad_value)

    side = SideBatch(side_tensors, Ns, Nmax)
    batch = Batch.from_data_list(stripped)
    return batch, side

def attach_side(batch: Batch, side: SideBatch) -> Batch:
    """把 SideBatch 掛到 Batch 上，供模块读取。"""
    batch._side = side
    return batch
