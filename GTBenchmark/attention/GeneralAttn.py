import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from contextlib import nullcontext
from typing import Optional

_BACKEND_MAP = {
    "auto":      None,
    "cudnn":     SDPBackend.CUDNN_ATTENTION,
    "flash":     SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math":      SDPBackend.MATH,
}

def _to_additive_from_full_mask(full_mask: Tensor, Np: int) -> Tensor:
    """
    full_mask: [B,1,N,N]，bool(True=keep) 或 float(0/-inf)
    返回：     [B,1,Np,Np] 的 additive 浮点掩码（与 graph token 对齐）
    """
    assert full_mask.dim() == 4 and full_mask.size(1) == 1
    B, _, N, _ = full_mask.shape
    if full_mask.dtype == torch.bool:
        add = torch.zeros_like(full_mask, dtype=torch.float)
        add = add.masked_fill(~full_mask, float('-inf'))
    else:
        add = full_mask

    if Np == N + 1:
        add = F.pad(add, (1, 0, 1, 0), value=0.0)  # 给 graph token 左上补 0 行列
    elif Np != N:
        raise RuntimeError(f"Padding mask size mismatch: full N={N}, attn N'={Np}")
    return add  # [B,1,Np,Np]


import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.register import register_layer

@register_layer('GeneralAttention')
class GeneralAttn(nn.Module):
    """
    概念上“分离”：结构偏置 E 放 batch.attn_bias，padding mask 从 Fullpair 进来；
    实现上当前 SDPA 仍需 additive mask -> 在层内临时融合（E + mask），未来换 Flex-Attn 直接传分离量即可。
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        x_name: str = 'x',
        bias_name: str = 'attn_bias',   # BiasEncoder 写入到 batch.attn_bias（[B*H,Np,Np] 或 [B,H,Np,Np]）
        backend: str = "auto",
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.x_name = x_name
        self.bias_name = bias_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p
        if backend not in _BACKEND_MAP:
            raise ValueError(f"Unknown backend '{backend}', valid keys: {list(_BACKEND_MAP)}")
        self.backend = backend

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _reshape_bias(self, attn_bias: Tensor, B: int, H: int, Np: int, device, dtype):
        """
        支持 [B*H,Np,Np] 或 [B,H,Np,Np]，统一成 [B,H,Np,Np]（float）。
        """
        if attn_bias.dim() == 3:
            if attn_bias.size(0) != B * H:
                raise RuntimeError(f"attn_bias size[0]={attn_bias.size(0)} != B*H={B*H}")
            E = attn_bias.view(B, H, Np, Np)
        elif attn_bias.dim() == 4:
            if attn_bias.size(0) != B or attn_bias.size(1) != H:
                raise RuntimeError(f"attn_bias shape={tuple(attn_bias.shape)} expect (B,H,Np,Np)")
            E = attn_bias
        else:
            raise RuntimeError(f"attn_bias dim={attn_bias.dim()} not in {{3,4}}")
        return E.to(device=device, dtype=dtype)  # [B,H,Np,Np]

    def _fuse_for_sdpa(self, B: int, Np: int, H: int,
                       pad_mask: Optional[Tensor], attn_bias: Optional[Tensor],
                       device, dtype) -> Optional[Tensor]:
        """
        SDPA 需要一个 additive mask：把 padding 的 full mask（[B,1,N,N]）与结构偏置 E（[B*H或B,H,Np,Np]）
        在这里**临时相加**成 [B,H,Np,Np]。将来切 Flex-Attn 就把这一段改成“原样传 E 与 mask”即可。
        """
        fused = None
        if attn_bias is not None:
            fused = attn_bias  # 已经是 [B,H,Np,Np]

        if pad_mask is not None:
            add = _to_additive_from_full_mask(pad_mask.to(device), Np)   # [B,1,Np,Np]
            add = add.expand(B, H, Np, Np)
            fused = add if fused is None else fused + add

        if fused is not None:
            fused = fused.to(device=device, dtype=dtype)
        return fused  # [B,H,Np,Np] or None

    def forward(self, batch, mask: Optional[Tensor] = None, return_attn_weights: bool = False):
        """
        mask: Fullpair 产生的 padding full mask（[B,1,N,N]，bool 或 0/-inf）
        结构偏置：从 batch.attn_bias 读取（[B*H,Np,Np] or [B,H,Np,Np]），概念上与 mask 分离；
        目前为了 SDPA 在层内融合成 additive mask，未来切 Flex-Attn 可直接传分离量。
        """
        x: Tensor = getattr(batch, self.x_name)  # [B,Np,D]
        B, Np, _ = x.shape
        H, D = self.num_heads, self.head_dim
        device, dtype = x.device, x.dtype
        _USE_MATH = True

        # QKV + 分头
        Q = self.q_proj(x).view(B, Np, H, D).transpose(1, 2).contiguous()
        K = self.k_proj(x).view(B, Np, H, D).transpose(1, 2).contiguous()
        V = self.v_proj(x).view(B, Np, H, D).transpose(1, 2).contiguous()

        # 读取结构偏置（不在外部相加）
        raw_bias = getattr(batch, self.bias_name, None)
        E = None
        if raw_bias is not None:
            E = self._reshape_bias(raw_bias, B, H, Np, device, dtype)   # [B,H,Np,Np]

        # —— 目前：为 SDPA 临时融合成一个 additive mask —— #
        additive_mask = self._fuse_for_sdpa(B, Np, H, mask, E, device, dtype)

        # 有 mask 时，禁用 flash/cudnn，更稳
        backend = "efficient" if additive_mask is not None else self.backend
        enum = _BACKEND_MAP.get(backend, None)
        ctx = sdpa_kernel([enum]) if enum is not None else nullcontext()

        if not return_attn_weights and not _USE_MATH:
            with ctx:
                out = scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=additive_mask,   # ← 临时融合（E + mask）
                    dropout_p=self.dropout_p,
                    is_causal=False,
                )
        else:
            # 手动算权重（同样先融合）
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [B,H,Np,Np]
            if additive_mask is not None:
                scores = scores + additive_mask
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            out = torch.matmul(attn, V)

        # 合头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, Np, H * D)
        out = self.out_proj(out)

        if return_attn_weights:
            return out, attn
        setattr(batch, self.x_name, out)
        return batch
