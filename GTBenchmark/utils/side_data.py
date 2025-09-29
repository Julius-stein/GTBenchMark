# sidedata_global.py
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import torch
from torch_geometric.data import Data, Batch
from GTBenchmark.graphgym.config import cfg

_DENSE_KEYS: Tuple[str, ...] = ("attn_bias", "spatial_pos", "attn_edge_type", "edge_input")

# -------- Global singleton --------
_GLOBAL_SIDE: Optional["SideData"] = None

def set_global_side(side: "SideData") -> None:
    global _GLOBAL_SIDE
    _GLOBAL_SIDE = side

def get_global_side() -> "SideData":
    if _GLOBAL_SIDE is None:
        raise RuntimeError("Global SideData is not set. Call build_and_attach_global(...) first.")
    return _GLOBAL_SIDE

def clear_global_side() -> None:
    global _GLOBAL_SIDE
    _GLOBAL_SIDE = None

# 供 torch.utils.data.DataLoader 使用的 collate_fn
def global_collate(batch_list: List[Data]):
    return get_global_side().collate(batch_list)

def build_and_attach_global(dataset, data_list: List[Data]) -> "SideData":
    side = SideData.from_data_list(data_list)
    side.attach_to(dataset, data_list=data_list)  # 这里会调用 dataset.collate(data_list)
    set_global_side(side)
    return side


def _channel_from_numel(t: torch.Tensor, N: int, name: str) -> int:
    """
    从元素个数推回通道维：C = numel // (N*N)
    - 允许 t 是 [N,N,C] / [N*N,C] / [N*N*C] / 扁平
    - 若 numel 不是 N*N 的整数倍，直接抛错（早失败，定位快）
    """
    NN = N * N
    numel = int(t.numel())
    if NN == 0:
        raise ValueError(f"{name}: N is 0")
    if numel % NN != 0:
        raise ValueError(f"{name}: numel({numel}) not divisible by N*N({NN})")
    C = numel // NN
    if C <= 0:
        raise ValueError(f"{name}: inferred channel {C} invalid")
    return C


# -------- SideData --------
class SideData:
    """管理 Graphormer 方阵副数据（连续缓冲 + offsets）。与 InMemoryDataset 解耦，训练时基于 torch DataLoader 自定义 collate 使用。"""

    @classmethod
    def from_data_list(cls, data_list: List[Data]) -> "SideData":
        ab_list, sp_list, aet_list, ei_list = [], [], [], []
        Ns, Fes, Ds = [], [], []

        for i, d in enumerate(data_list):
            if not hasattr(d, "sample_idx"):
                d.sample_idx = torch.tensor([i], dtype=torch.long)

            N  = int(d.num_nodes)
            ab = getattr(d, "attn_bias", None)        # [N+1, N+1] float
            sp = getattr(d, "spatial_pos", None)      # [N, N]     long
            ae = getattr(d, "attn_edge_type", None)   # [N, N, Fe] long
            ei = getattr(d, "edge_input", None)       # [N, N, D]  long

            # 在 SideData.from_data_list(...) 里，原来推断 Fe/D 的几行替换为：
            if ae is None:
                Fe = 1
            else:
                Fe = _channel_from_numel(ae, N, "attn_edge_type")

            if ei is None:
                D = 1
            else:
                D  = _channel_from_numel(ei, N, "edge_input")


            Ns.append(N); Fes.append(Fe); Ds.append(D)
            ab_list.append(_flatten_or_empty(ab, torch.float32))
            sp_list.append(_flatten_or_empty(sp, torch.long))
            aet_list.append(_flatten_or_empty(ae, torch.long))
            ei_list.append(_flatten_or_empty(ei, torch.long))

            # 移除方阵键，避免 InMemoryDataset.collate 阶段参与拼接
            for k in _DENSE_KEYS:
                if hasattr(d, k):
                    delattr(d, k)

        ab_buf,  ab_off  = _concat_with_offsets(ab_list,  torch.float32)
        sp_buf,  sp_off  = _concat_with_offsets(sp_list,  torch.long)
        aet_buf, aet_off = _concat_with_offsets(aet_list, torch.long)
        ei_buf,  ei_off  = _concat_with_offsets(ei_list,  torch.long)

        return cls(
            ab_buf=ab_buf,  ab_off=ab_off,
            sp_buf=sp_buf,  sp_off=sp_off,
            aet_buf=aet_buf, aet_off=aet_off,
            ei_buf=ei_buf,  ei_off=ei_off,
            Ns=torch.tensor(Ns,  dtype=torch.long),
            Fes=torch.tensor(Fes, dtype=torch.long),
            Ds=torch.tensor(Ds,  dtype=torch.long),
        )

    def __init__(
        self,
        *,
        ab_buf: torch.Tensor,  ab_off: torch.Tensor,
        sp_buf: torch.Tensor,  sp_off: torch.Tensor,
        aet_buf: torch.Tensor, aet_off: torch.Tensor,
        ei_buf: torch.Tensor,  ei_off: torch.Tensor,
        Ns: torch.Tensor, Fes: torch.Tensor, Ds: torch.Tensor,
    ):
        self.ab_buf,  self.ab_off  = ab_buf,  ab_off
        self.sp_buf,  self.sp_off  = sp_buf,  sp_off
        self.aet_buf, self.aet_off = aet_buf, aet_off
        self.ei_buf,  self.ei_off  = ei_buf,  ei_off
        self.Ns, self.Fes, self.Ds = Ns, Fes, Ds

    def attach_to(self, dataset, data_list: Optional[List[Data]] = None) -> None:
        """挂到 dataset；若提供 data_list，则调用 dataset.collate(data_list) 构建 InMemoryDataset。"""
        dataset._side = self
        dataset._dense_keys = _DENSE_KEYS
        if data_list is not None:
            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

    # ------- 训练时用的 collate（给 torch.utils.data.DataLoader） -------
    def collate(self, batch_list: List[Data]) -> Batch:
        # 注意：这里不用 PyG 的 DataLoader 内部 Collater，直接用 Batch.from_data_list
        batch = Batch.from_data_list(batch_list)  # 大方阵键已在预处理剥离，不会冲突

        idxs = [int(d.sample_idx.item()) for d in batch_list]
        Ns_b  = self.Ns[idxs]
        Fes_b = self.Fes[idxs]
        Ds_b  = self.Ds[idxs]

        B     = len(batch_list)
        if cfg.share.targetsize==-1:
            Nmax  = int(Ns_b.max().item()) if B > 0 else 1
        else:
            Nmax  = cfg.share.targetsize
        Fe    = int(Fes_b.max().item()) if B > 0 else 1
        D     = int(Ds_b.max().item())  if B > 0 else 1

        attn_bias      = torch.zeros(B, Nmax+1, Nmax+1, dtype=torch.float32)
        spatial_pos    = torch.zeros(B, Nmax,   Nmax,   dtype=torch.long)
        attn_edge_type = torch.zeros(B, Nmax,   Nmax,   Fe, dtype=torch.long)
        # edge_input     = torch.zeros(B, Nmax,   Nmax,   D,  dtype=torch.long)
        edge_input     = torch.zeros(B, Nmax, Nmax, D, Fe, dtype=torch.long)
        node_mask           = torch.zeros(B, Nmax,   dtype=torch.bool)
        node_with_tok_mask  = torch.zeros(B, Nmax+1, dtype=torch.bool)

        for i, idx in enumerate(idxs):
            N   = int(self.Ns[idx])
            Fei = int(self.Fes[idx])
            Di  = int(self.Ds[idx])

            ab = _slice_flat(self.ab_buf,  self.ab_off,  idx)
            sp = _slice_flat(self.sp_buf,  self.sp_off,  idx)
            ae = _slice_flat(self.aet_buf, self.aet_off, idx)
            ei = _slice_flat(self.ei_buf,  self.ei_off,  idx)

            # 在 SideData.collate(...) 的 for 循环里，还原四个张量的部分替换为严格版：

            # 1) attn_bias: 必须等于 (N+1)*(N+1)
            if ab.numel() != (N + 1) * (N + 1):
                raise RuntimeError(
                    f"attn_bias size mismatch for idx={idx}: got {ab.numel()}, "
                    f"expected {(N+1)*(N+1)} (N={N})"
                )
            attn_bias[i, :N+1, :N+1] = ab.view(N+1, N+1)

            # 2) spatial_pos: 必须等于 N*N
            if sp.numel() != N * N:
                raise RuntimeError(
                    f"spatial_pos size mismatch for idx={idx}: got {sp.numel()}, "
                    f"expected {N*N} (N={N})"
                )
            spatial_pos[i, :N, :N] = sp.view(N, N)

            # 3) attn_edge_type: 必须等于 N*N*Fei
            if ae.numel() != N * N * Fei:
                raise RuntimeError(
                    f"attn_edge_type size mismatch for idx={idx}: got {ae.numel()}, "
                    f"expected {N*N*Fei} (N={N}, Fe={Fei})"
                )
            attn_edge_type[i, :N, :N, :Fei] = ae.view(N, N, Fei)

            # 4) edge_input: 必须等于 N*N*Di
            if ei.numel() != N * N * Di:
                raise RuntimeError(
                    f"edge_input size mismatch for idx={idx}: got {ei.numel()}, "
                    f"expected {N*N*Di} (N={N}, D={Di})"
                )
            edge_input[i, :N, :N, :Di, :Fei] = ei.view(N, N, Di, Fei)


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


# ---- helpers ----
def _flatten_or_empty(t: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    if t is None:
        return torch.empty(0, dtype=dtype)
    return t.reshape(-1).to(dtype).contiguous()

def _slice_flat(buf: torch.Tensor, off: torch.Tensor, idx: int) -> torch.Tensor:
    lo = int(off[idx]); hi = int(off[idx + 1])
    return buf[lo:hi] if hi > lo else buf.new_empty(0, dtype=buf.dtype)

def _concat_with_offsets(lst: List[torch.Tensor], dtype: torch.dtype):
    sizes = [int(x.numel()) for x in lst]
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
    if sizes:
        offsets[1:] = torch.cumsum(torch.tensor(sizes, dtype=torch.long), dim=0)
    total = int(offsets[-1])
    if total == 0:
        return torch.empty(0, dtype=dtype), offsets
    out = torch.empty(total, dtype=dtype)
    pos = 0
    for x in lst:
        n = int(x.numel())
        if n:
            out[pos:pos+n] = x.to(dtype)
            pos += n
    return out, offsets
