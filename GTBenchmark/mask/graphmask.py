import torch
import torch.nn as nn
from typing import Optional
import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_mask
from GTBenchmark.transform.graph2dense import to_dense_batch, to_dense_adj
# 若你用自己的路径, 改 import

@register_mask('graph')
class GraphSDPAMask(nn.Module):
    """
    将 PyG Batch 转为 dense 并生成可直接传给 scaled_dot_product_attention 的 Bool attn_mask:
      - Bool attn_mask 语义: True = 允许注意力, False = 屏蔽
      - 自动结合 padding (无效节点) & 图结构 (edge_index)
      - 支持缓存 (结构不变时不重复构造)
      - 支持多头广播
    """
    def __init__(self,
                 add_self_loop: bool = True,
                 cache_adj: bool = True,
                 pad_fill: float = 0.0):
        super().__init__()
        self.add_self_loop = add_self_loop
        self.cache_adj = cache_adj
        self.pad_fill = pad_fill

        # 缓存
        self._sig = None
        self.node_valid_mask = None    # [B,Nmax] True=有效节点
        self.Gindex = None             # 展平 -> dense 索引
        self.inv_index = None          # dense -> 原节点位置 (可选)
        self.adj_allow_cache = None    # [B,Nmax,Nmax] Bool True=允许
        self.Nmax = None
        self.B = None

    # ---------- 辅助：生成 batch 结构签名 ----------
    @staticmethod
    def _make_signature(batch):
        # ptr + num_graphs + sum_nodes  足够判定结构是否变化
        ptr_cpu = batch.ptr.detach().cpu()
        return (tuple(ptr_cpu.tolist()),
                int(batch.num_graphs),
                int(ptr_cpu[-1]))

    # ---------- 检测并重建 dense 化 ----------
    def _rebuild_dense_if_needed(self, batch):
        sig = self._make_signature(batch)
        if sig == self._sig:
            return  # 结构未变

        # 结构变化，重建
        self._sig = sig
        self.B = batch.num_graphs
        self.Nmax = int((batch.ptr[1:] - batch.ptr[:-1]).max())

        dense_x, node_valid_mask, Gindex = to_dense_batch(
            batch.x, batch.batch,
            max_num_nodes=self.Nmax,
            batch_size=self.B,
            fill_value=self.pad_fill
        )
        batch.x = dense_x                         # [B,Nmax,F]
        self.node_valid_mask = node_valid_mask    # [B,Nmax]
        self.Gindex = Gindex                      # [总节点数] (把原展平索引映射到 dense 有效位置)
        # 逆映射：dense -> 原 flatten 索引
        inv = torch.full((self.B * self.Nmax,), -1,
                         dtype=torch.long,
                         device=Gindex.device)
        inv[Gindex] = torch.arange(Gindex.numel(), device=Gindex.device)
        self.inv_index = inv.view(self.B, self.Nmax)

        # 邻接缓存失效
        self.adj_allow_cache = None

    # ---------- 构建 (或复用) 图结构允许矩阵 ----------
    def _get_adj_allow(self, batch):
        if self.cache_adj and self.adj_allow_cache is not None:
            return self.adj_allow_cache

        adj = to_dense_adj(
            batch.edge_index,
            batch=batch.batch,
            max_num_nodes=self.Nmax
        ).bool()  # [B,Nmax,Nmax]  True=存在边

        if self.add_self_loop:
            eye = torch.eye(self.Nmax, device=adj.device, dtype=torch.bool)
            adj |= eye.unsqueeze(0)   # 允许 self-attention

        # 去掉 padding：形成 pair 有效 mask
        valid = self.node_valid_mask  # [B,N]
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)  # [B,N,N]
        adj &= pair_valid             # padding 节点相关的 pair 全 False

        if self.cache_adj:
            self.adj_allow_cache = adj
        return adj

    # ---------- 前向：返回 batch 与 attn_mask ----------
    def forward(self,
                batch,
                want_heads: Optional[int] = None,
                extra_pair_allow: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_float_bias: bool = False,
                pair_bias: Optional[torch.Tensor] = None):
        """
        Args:
            batch: PyG Batch
            want_heads: 需要广播的头数 H；若为 None -> 不扩展 (返回 [B,1,N,N])
            extra_pair_allow: 额外的 pair 允许矩阵 (Bool, True=允许) shape 可广播到 [B,N,N]
            key_padding_mask: 旧语义 True=屏蔽 的 [B,N]；会转换成 allow
            return_float_bias: 若 True, 返回 (attn_mask_bool, float_bias)；否则只返回 bool
            pair_bias: 额外加性 bias (float)，与 logits 同加。shape 可是 [B,1,N,N] / [B,H,N,N] / [1,1,N,N]

        Returns:
            batch: 修改后含 dense x
            attn_mask: Bool [B,H,N,N] (True=允许)
            (可选) float_bias: 若 return_float_bias=True，则返回 broadcast 后的加性偏置 (float)
        """
        if not hasattr(batch, "num_graphs"):
            # 单图 / 已经 dense 的场景：假设 batch.x 是 [N,F]
            raise ValueError("需要 PyG Batch (含 .num_graphs / .ptr)")

        # 1. Dense 化（必要时重建）
        self._rebuild_dense_if_needed(batch)

        # 2. 基础结构允许矩阵
        allow = self._get_adj_allow(batch)  # [B,N,N], Bool True=允许

        # 3. 处理 key padding mask (旧语义 True=屏蔽)
        if key_padding_mask is not None:
            # 旧 -> 允许
            key_allow = ~key_padding_mask.bool()  # [B,N] True=允许
            allow &= key_allow.unsqueeze(1) & key_allow.unsqueeze(2)

        # 4. 额外 pair 允许（例如你的 Exphormer 扩展、全局 token、虚连）
        if extra_pair_allow is not None:
            # extra_pair_allow True=允许；我们与当前 allow 做与 (更严格)
            extra_pair_allow = extra_pair_allow.to(dtype=torch.bool, device=allow.device)
            allow &= extra_pair_allow

        # 5. 广播为 [B,H,N,N]
        if want_heads is None:
            attn_mask = allow.unsqueeze(1)  # [B,1,N,N]
        else:
            attn_mask = allow.unsqueeze(1).expand(self.B, want_heads, self.Nmax, self.Nmax)

        if not return_float_bias and pair_bias is None:
            return batch, attn_mask  # Bool True=允许

        # 6. 构造 float bias（如果需要）
        float_bias = None
        if pair_bias is not None:
            # pair_bias 直接加在 logits 上，所以不对“屏蔽”做 -inf，这里只处理允许的地方。
            float_bias = pair_bias
            # 广播
            target_shape = attn_mask.shape  # [B,H,N,N]
            while float_bias.dim() < 4:
                float_bias = float_bias.unsqueeze(0)
            # 让它广播匹配 [B,H,N,N]
            if float_bias.shape[0] == 1 and target_shape[0] > 1:
                float_bias = float_bias.expand(target_shape[0], *float_bias.shape[1:])
            if float_bias.shape[1] == 1 and target_shape[1] > 1:
                float_bias = float_bias.expand(float_bias.shape[0], target_shape[1], *float_bias.shape[2:])
            # 若需要对被屏蔽位置填零/不影响 softmax，可在外部再 masked_fill 处理

        return batch, attn_mask, float_bias

    # ---------- 从 dense 回退 ----------
    def from_dense_batch(self, batch):
        """
        将 batch.x (B,Nmax,F) 恢复为 (N_total, F) 与原顺序一致
        """
        if self.node_valid_mask is None:
            raise RuntimeError("from_dense() 在 forward 之后调用")
        dense_x = batch.x  # [B,N,F]
        batch.x = dense_x[self.node_valid_mask]
        return batch

    # ---------- 手动清理缓存 ----------
    def clear_cache(self):
        self._sig = None
        self.node_valid_mask = None
        self.Gindex = None
        self.inv_index = None
        self.adj_allow_cache = None
        self.Nmax = None
        self.B = None


@register_mask('noneed')
class NoNeedMask(nn.Module):
    def __init__(self,batch):
        super().__init__()

    def forward(self,batch):
        return batch,None
    
    def from_dense_batch(self, batch):
        return batch