import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from contextlib import nullcontext
from typing import Optional
from GTBenchmark.graphgym.register import register_layer


_BACKEND_MAP = {
    "auto":      None,
    "cudnn":     SDPBackend.CUDNN_ATTENTION,
    "flash":     SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math":      SDPBackend.MATH,
}


@register_layer('GeneralAttention')
class GeneralAttn(nn.Module):
    """
    General multi-head attention with:
    - SDPA / math fallback
    - attn_bias support
    - unified mask handling:
        * [B, L]        -> key padding mask
        * [B,1,N,N]    -> attention mask
        * [B,H,N,N]    -> attention mask
    All masks are converted to additive form [B,H,Np,Np].
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        x_name: str = 'x',
        bias_name: str = 'attn_bias',
        backend: str = "auto",
        return_attn_weights: bool = False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = float(dropout_p)
        self._scale = 1.0 / math.sqrt(self.head_dim)

        self.x_name = x_name
        self.bias_name = bias_name
        self.return_attn_weights = bool(return_attn_weights)

        if backend not in _BACKEND_MAP:
            raise ValueError(f"Unknown backend '{backend}'")
        self.backend = backend
        chosen = backend if backend != "auto" else "efficient"
        self._sdpa_enum = _BACKEND_MAP.get(chosen, SDPBackend.EFFICIENT_ATTENTION)

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

        if self.return_attn_weights:
            self.forward = self._forward_with_weights  # type: ignore
        else:
            self.forward = self._forward_no_weights    # type: ignore

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reshape_attn_bias(attn_bias: Tensor, B: int, H: int, Np: int) -> Tensor:
        if attn_bias.dim() == 3:
            if attn_bias.size(0) != B * H:
                raise RuntimeError("attn_bias shape mismatch")
            return attn_bias.view(B, H, Np, Np)
        elif attn_bias.dim() == 4:
            if attn_bias.size(1) == attn_bias.size(2):
                attn_bias = attn_bias.permute(0, 3, 1, 2)
            if attn_bias.size(0) != B or attn_bias.size(1) != H:
                raise RuntimeError("attn_bias shape mismatch")
            return attn_bias
        else:
            raise RuntimeError("attn_bias must be 3D or 4D")

    @staticmethod
    def _mask_to_additive(mask: Tensor, B: int, H: int, Np: int) -> Tensor:
        """
        Unified mask parser:
        - [B, L]      -> key padding mask
        - [B,1,N,N]  -> attention mask
        - [B,H,N,N]  -> attention mask
        Return: [B,H,Np,Np] additive mask
        """
        if mask.dtype != torch.bool and not torch.is_floating_point(mask):
            mask = mask.bool()

        # key padding mask
        if mask.dim() == 2:
            if mask.size(1) != Np:
                raise RuntimeError(f"padding mask length {mask.size(1)} != Np {Np}")
            add = torch.zeros((B, 1, 1, Np),
                              device=mask.device,
                              dtype=torch.float)
            add = add.masked_fill(~mask[:, None, None, :], float('-inf'))
            return add.expand(B, H, Np, Np)

        # attention mask
        if mask.dim() == 4:
            if mask.dtype == torch.bool:
                add = torch.zeros_like(mask, dtype=torch.float)
                add = add.masked_fill(~mask, float('-inf'))
            else:
                add = mask

            if add.size(1) == 1:
                add = add.expand(B, H, Np, Np)
            elif add.size(1) != H:
                raise RuntimeError("attn mask head dim mismatch")

            return add.contiguous()

        raise RuntimeError(f"Unsupported mask dim={mask.dim()}")

    @staticmethod
    def _build_scores(Q: Tensor, K: Tensor, bias: Optional[Tensor]) -> Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if bias is not None:
            scores = scores + bias
        return scores

    # ------------------------------------------------------------------
    # core
    # ------------------------------------------------------------------
    def _prep_qkv(self, batch):
        x: Tensor = getattr(batch, self.x_name)
        B, Np, _ = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.in_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        Q = q.view(B, Np, H, D).transpose(1, 2).contiguous()
        K = k.view(B, Np, H, D).transpose(1, 2).contiguous()
        V = v.view(B, Np, H, D).transpose(1, 2).contiguous()

        return Q, K, V, B, Np, H, D

    # ------------------------------------------------------------------
    # forward paths
    # ------------------------------------------------------------------
    def _forward_no_weights(self, batch, mask: Optional[Tensor] = None):
        Q, K, V, B, Np, H, D = self._prep_qkv(batch)

        bias_list = []

        raw_bias = getattr(batch, self.bias_name, None)
        if raw_bias is not None:
            bias_list.append(self._reshape_attn_bias(raw_bias, B, H, Np))

        if mask is not None:
            bias_list.append(self._mask_to_additive(mask, B, H, Np))

        fused_bias = sum(bias_list).contiguous() if len(bias_list) > 0 else None

        if self.backend != "math":
            try:
                ctx = sdpa_kernel([self._sdpa_enum]) if self._sdpa_enum is not None else nullcontext()
                with ctx:
                    out = scaled_dot_product_attention(
                        Q, K, V,
                        attn_mask=fused_bias,
                        dropout_p=self.dropout_p,
                        is_causal=False,
                        scale=self._scale,
                    )
            except RuntimeError as e:
                if "No available kernel" not in str(e):
                    raise
                scores = self._build_scores(Q, K, fused_bias) * self._scale
                attn = torch.softmax(scores, dim=-1)
                if self._drop is not None and self.training:
                    attn = self._drop(attn)
                out = torch.matmul(attn, V)
        else:
            scores = self._build_scores(Q, K, fused_bias) * self._scale
            attn = torch.softmax(scores, dim=-1)
            if self._drop is not None and self.training:
                attn = self._drop(attn)
            out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, Np, H * D)
        out = self.out_proj(out)
        setattr(batch, self.x_name, out)
        return batch

    def _forward_with_weights(self, batch, mask: Optional[Tensor] = None):
        Q, K, V, B, Np, H, D = self._prep_qkv(batch)

        fused_bias = self._mask_to_additive(mask, B, H, Np) if mask is not None else None
        scores = self._build_scores(Q, K, fused_bias) * self._scale
        attn = torch.softmax(scores, dim=-1)
        attn_used = self._drop(attn) if (self._drop is not None and self.training) else attn
        out = torch.matmul(attn_used, V)

        out = out.transpose(1, 2).contiguous().view(B, Np, H * D)
        out = self.out_proj(out)
        return out, attn
