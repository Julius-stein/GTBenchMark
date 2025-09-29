import torch, numpy as np
from typing import Iterator, Dict

class SidecarCache:
    def __init__(self, sidecar_dir: str, max_items: int = 256):
        import os
        self.dir, self.max_items = sidecar_dir, max_items
        self.buf, self.order, self.os = {}, [], os

    def load(self, sid: int) -> torch.Tensor:
        if sid in self.buf:
            self.order.remove(sid); self.order.append(sid)
            return self.buf[sid]
        arr = np.load(self.os.path.join(self.dir, f"{sid:08d}.npz"))["values"]
        t = torch.from_numpy(arr).long()   # [T_g, Fe]
        self.buf[sid] = t; self.order.append(sid)
        if len(self.order) > self.max_items:
            old = self.order.pop(0); self.buf.pop(old, None)
        return t

@torch.no_grad()
def iter_graph_views(batch, sidecar: SidecarCache) -> Iterator[Dict]:
    """
    适用于 pair_index 形状 [2, M_all]。
    【关键区别】不使用 num_pairs 计算切段，
    直接用全局 pair_ptr 中的“零值位置”作为每图边界，避免错位。
    """
    dev = batch.pair_index.device
    PI = batch.pair_index.to(dev)          # [2, M_all]
    PP = batch.pair_ptr.to(dev)            # [M_all + B]
    sids = batch.sid.view(-1).tolist()     # [B]
    N_per = batch.num_nodes_graph.view(-1).tolist()
    B = len(sids)

    # 1) 找所有图的边界：pair_ptr==0 的位置（每图开头必为 0）
    zeros = (PP == 0).nonzero(as_tuple=False).view(-1)
    if zeros.numel() < B:
        raise RuntimeError(f"Expect at least {B} zeros in pair_ptr, got {int(zeros.numel())}")

    # 2) 运行指针：在 pair_index 的第二维上前进
    m_off = 0
    for g in range(B):
        ptr_start = int(zeros[g].item())
        ptr_end   = int(zeros[g+1].item()) if g+1 < zeros.numel() else int(PP.numel())

        ptr_g = PP[ptr_start:ptr_end].clone()   # [M_g+1]
        # 基本合法性检查
        if ptr_g.numel() == 0:
            # 理论上不会发生；保底处理
            yield dict(N=int(N_per[g]), i=PI.new_empty(0), j=PI.new_empty(0),
                       pair_ptr=torch.tensor([0], device=dev), values=sidecar.load(int(sids[g])))
            continue
        if not bool(torch.all(ptr_g[1:] >= ptr_g[:-1])):
            raise RuntimeError(f"pair_ptr not non-decreasing for graph {g}: {ptr_g[:10]} ...")

        # 局部 0 起点
        base = ptr_g[0].item()
        ptr_g -= ptr_g[0].clone()
        if ptr_g[0].item() != 0 or bool((ptr_g < 0).any()):
            raise RuntimeError(f"pair_ptr normalize error for graph {g}: base={base}, head={ptr_g[:5]}")

        M_g = ptr_g.numel() - 1
        # 用运行指针在 pair_index 的“第二维”上取出本图的 M_g 个 (i,j)
        i = PI[0, m_off:m_off + M_g].contiguous()
        j = PI[1, m_off:m_off + M_g].contiguous()
        m_off += M_g

        # 可选一致性检查：如果你有 num_pairs，就校验下
        if hasattr(batch, "num_pairs"):
            npairs = int(batch.num_pairs.view(-1)[g].item())
            if npairs != M_g:
                print(f"[warn] graph {g}: num_pairs={npairs} != M_g(from ptr)={M_g}")

        yield dict(N=int(N_per[g]), i=i, j=j, pair_ptr=ptr_g, values=sidecar.load(int(sids[g])))