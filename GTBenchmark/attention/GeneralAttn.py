import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Optional, Tuple
from contextlib import nullcontext

# 支持的后端映射
_BACKEND_MAP = {
    "auto":      None,
    "cudnn":     SDPBackend.CUDNN_ATTENTION,
    "flash":     SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math":      SDPBackend.MATH,
}

class SDPAttn(nn.Module):
    """
    基于 scaled_dot_product_attention 的通用 Attention 模块，
    可切换多种后端，并可选返回 Attention 权重。
    """
    def __init__(
        self,
        embed_dim: int,
        dropout_p: float = 0.0,
        backend: str = "auto",
    ) -> None:
        """
        Args:
            embed_dim:   QKV 合并后维度（H * D_head）
            dropout_p:   Attention 权重上的 dropout 概率
            backend:     one of ["auto","cudnn","flash","efficient","math"]
        """
        super().__init__()
        if backend not in _BACKEND_MAP:
            raise ValueError(f"Unknown backend '{backend}', valid keys: {list(_BACKEND_MAP)}")
        self.backend = backend
        self.dropout_p = dropout_p

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
        E: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Args:
            Q, K, V:             shape (B, H, N, D_head)
            mask:                optional bool/float tensor of shape
                                 (N,N), (B,N,N), (B,1,N,N) or (B,H,N,N)
            E:                   optional additive bias tensor of shape
                                 (N,N) or (B,H,N,N)
            is_causal:           whether to apply causal masking
            return_attn_weights: whether to return (output, attn_weights)

        Returns:
            attn_out:            (B, H, N, D_head)
            attn_weights:        (B, H, N, N) if return_attn_weights=True
        """
        # —— 1. 检查维度一致性 —— 
        B, H, N, D = Q.shape
        if K.shape != Q.shape or V.shape != Q.shape:
            raise ValueError("Q, K, V must have the same shape")
        if mask is not None:
            if mask.dim() not in (2, 3, 4):
                raise ValueError("mask.dim must be 2, 3, or 4")
            if mask.shape[-2:] != (N, N):
                raise ValueError(f"mask's last two dims must be (N,N), got {mask.shape[-2:]}")
            if mask.dim() == 4 and mask.shape[1] not in (1, H):
                raise ValueError(f"When mask.dim==4, mask.shape[1] must be 1 or H={H}")

        # —— 2. 使用 fused SDP（不需要权重时） —— 
        if not return_attn_weights:
            backend_enum = _BACKEND_MAP[self.backend]
            ctx = sdpa_kernel([backend_enum]) if backend_enum is not None else nullcontext()
            with ctx:
                out = scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask,
                    dropout_p=self.dropout_p,
                    is_causal=False,
                )

            return out

        # —— 3. 手动计算（需要返回权重时） —— 
        # 3.1 raw scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B,H,N,N)
        # 3.2 加性偏置 E
        if E is not None:
            if E.dim() == 2:
                scores = scores + E
            elif E.dim() == 4:
                scores = scores + E.unsqueeze(0)
            else:
                raise ValueError("E must have dim 2 or 4")
        # 3.3 应用 mask
        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(1)    # -> (1,1,N,N)
            elif m.dim() == 3:
                m = m.unsqueeze(1)                # -> (B,1,N,N)
            # bool mask: False->-inf；float mask: 直接相加
            scores = torch.where(m if m.dtype==torch.bool else torch.ones_like(scores),
                                 float("-inf") if m.dtype==torch.bool else scores, scores) \
                     if m.dtype==torch.bool else scores + m

        # 3.4 softmax + dropout 得到 attn_weights
        attn_weights = F.softmax(scores, dim=-1)                         # (B,H,N,N)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        # 3.5 加权 V
        out = torch.matmul(attn_weights, V)                              # (B,H,N,D)

        return out, attn_weights


import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.register import register_layer
@register_layer('GeneralAttention')
class GeneralAttn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        x_name='x', b_name='attn_bias',
        backend: str = "auto",
    ):
        """
        embed_dim:   输入/输出的维度 H * D_head
        num_heads:   头数 H
        默认为batch first情况
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.x_name = x_name
        self.b_name = b_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 用于生成 Q, K, V 的线性层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 调用前面定义的通用注意力
        self.inner_attn = SDPAttn(
            embed_dim=embed_dim,
            dropout_p=dropout_p,
            backend=backend
        )
        # Optional: 最终再投影一次
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        batch: Tensor,                           # (B, N, embed_dim)
        mask: Optional[Tensor] = None,       # 可为 (N,N),(B,N,N),(B,1,N,N),(B,H,N,N)
        return_attn_weights: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        x = getattr(batch,self.x_name)
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # 1. 线性映射
        Q = self.q_proj(x)  # (B, N, H*D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 拆分 heads
        #   -> (B, N, H, D) -> (B, H, N, D)
        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        # 3. 调用 GeneralAttn
        if return_attn_weights:
            attn_out, attn_w = self.inner_attn(
                Q, K, V,
                mask=mask,
                return_attn_weights=True,
            )
        else:
            attn_out = self.inner_attn(
                Q, K, V,
                mask=mask,
                return_attn_weights=False,
            )

        # 4. 合并 heads
        #   (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D)
        out = attn_out.transpose(1, 2).contiguous().view(B, N, H * D)

        # 5. 最终投影
        out = self.out_proj(out)  # (B, N, embed_dim)

        if return_attn_weights:
            return out, attn_w
        setattr(batch, self.x_name, out)
        return batch
