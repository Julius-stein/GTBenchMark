import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from GTBenchmark.graphgym.config import cfg
from contextlib import nullcontext
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention as  _flex_attention
_BACKEND_MAP = {
    "auto":      None,
    "cudnn":     SDPBackend.CUDNN_ATTENTION,
    "flash":     SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math":      SDPBackend.MATH,
    "flex":      "flex",
}
from GTBenchmark.graphgym.register import register_layer




def _to_additive_from_full_mask(full_mask: Tensor, Np: int) -> Tensor:
    """
    full_mask: [B,1,N,N]，bool(True=keep) 或 float(0/-inf)
    返回：     [B,1,Np,Np] 的 additive 浮点掩码（与 graph token 对齐）
    """
    assert full_mask.dim() == 4
    B, _, N, _ = full_mask.shape
    if full_mask.dtype == torch.bool:
        add = torch.zeros_like(full_mask, dtype=torch.float, device=full_mask.device)
        add = add.masked_fill(~full_mask, float('-inf'))
    else:
        add = full_mask if full_mask.device else full_mask
    if Np == N + 1:
        add = F.pad(add, (1, 0, 1, 0), value=0.0)
    elif Np != N:
        raise RuntimeError(f"Padding mask size mismatch: full N={N}, attn N'={Np}")
    return add  # [B,1,Np,Np]

@register_layer('GeneralAttention')
class GeneralAttn(nn.Module):
    """
    - return_attn_weights=True 时固定走 math 分支并返回 (out, attn)
    - Flex: 用 score_mod 注入结构偏置 E 与 additive（score_mod 内无条件分支）
            同时 **传入 block_mask**（从 batch.flex_block_mask 读取） // NEW
    - SDPA: 融合成 additive mask -> scaled_dot_product_attention
    - Math: 纯手算（scores/softmax/dropout），带权重路径共用
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        x_name: str = 'x',
        bias_name: str = 'attn_bias',     # [B*H,Np,Np] 或 [B,H,Np,Np]
        backend: str = "auto",            # auto / flex / efficient / flash / cudnn / math
        return_attn_weights: bool = False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.x_name = x_name
        self.bias_name = bias_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = float(dropout_p)
        self._scale = 1.0 / math.sqrt(self.head_dim)
        self.return_attn_weights = bool(return_attn_weights)

        if backend not in _BACKEND_MAP:
            raise ValueError(f"Unknown backend '{backend}', valid keys: {list(_BACKEND_MAP)}")
        self.backend = backend

        # ---- 静态确定后端，减少运行期分支 ----
        cfg.share.can_flex = (self.dropout_p == 0.0) and cfg.gt.use_flex
        if self.backend == "auto":
            self._use_flex = cfg.share.can_flex 
        else:
            self._use_flex = False

        if self._use_flex or self.backend == "math":
            self._sdpa_enum = None
        else:
            chosen = self.backend if self.backend != "auto" else "efficient"
            self._sdpa_enum = _BACKEND_MAP.get(chosen, SDPBackend.EFFICIENT_ATTENTION)

        self.in_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

        # 绑定 forward 实现
        if self.return_attn_weights:
            self.forward = self._forward_with_weights  # type: ignore[assignment]
        else:
            self.forward = self._forward_no_weights    # type: ignore[assignment]

    # ------------- helpers -------------
    @staticmethod
    def _reshape_bias(attn_bias: Tensor, B: int, H: int, Np: int) -> Tensor:
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
        return E

    @staticmethod
    def _get_cached_additive(batch, pad_mask: Optional[Tensor], Np: int):
        if pad_mask is None:
            return None
        cache = getattr(batch, "_attn_cache", None)
        if cache is None:
            cache = {}
            setattr(batch, "_attn_cache", cache)
        key = (id(pad_mask), Np)
        add = cache.get(key, None)
        if add is None:
            add = _to_additive_from_full_mask(pad_mask, Np)  # [B,1,Np,Np]
            cache[key] = add
        return add

    @staticmethod
    def _build_scores(Q: Tensor, K: Tensor, fused_bias: Optional[Tensor]) -> Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if fused_bias is not None:
            scores = scores + fused_bias
        return scores

    # ---- NEW: 把 block_mask 也传进来 ----
    def flex_attention(self, Q, K, V, fused_bias: Optional[Tensor], block_mask=None) -> Tensor:

        H = self.num_heads
        B, _, Np, _ = Q.shape

        if fused_bias is None:
            def noop(score, b, h, q_idx, k_idx):
                return score
            return _flex_attention(Q, K, V,
                                   score_mod=noop,
                                   block_mask=block_mask,  # <<< 传入
                                   scale=self._scale,
                                   kernel_options=None)

        fused_flat = fused_bias.reshape(-1)

        def score_mod(score, b, h, q_idx, k_idx):
            lin = (((b * H) + h) * Np + q_idx) * Np + k_idx
            return score + fused_flat[lin]

        return _flex_attention(Q, K, V,
                               score_mod=score_mod,
                               block_mask=block_mask,      # <<< 传入
                               scale=self._scale,
                               kernel_options=None)

    # ------------- 公共准备步骤 -------------
    def _prep_qkv_bias(self, batch, mask: Optional[Tensor]):
        x: Tensor = getattr(batch, self.x_name)     # [B,Np,D_model]
        B, Np, _ = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.in_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        Q = q.view(B, Np, H, D).transpose(1, 2).contiguous()
        K = k.view(B, Np, H, D).transpose(1, 2).contiguous()
        V = v.view(B, Np, H, D).transpose(1, 2).contiguous()

        return Q, K, V, B, Np, H, D

    # ------------- 两条 forward 实现 -------------
    def _forward_no_weights(self, batch, mask: Optional[Tensor] = None):
        Q, K, V, B, Np, H, D = self._prep_qkv_bias(batch, mask)
        
        raw_bias = getattr(batch, self.bias_name, None)
        E = self._reshape_bias(raw_bias, B, H, Np) if raw_bias is not None else None
        #在这个地方就应该分支flex了
        
        if self._use_flex:
            block_mask = getattr(batch, "flex_block_mask", None)  # <<< NEW: 读取 Basemask 的 BlockMask
            out = self.flex_attention(Q, K, V, E, block_mask=block_mask)
        else:
            add = self._get_cached_additive(batch, mask, Np)  # [B,1,Np,Np] or None
            if E is None and add is None:
                fused_bias = None
            else:
                if add is not None:
                    add = add.expand(B, H, Np, Np)
                    fused_bias = add if E is None else (E + add)
                else:
                    fused_bias = E
            if self.backend != "math":
                try:
                    ctx = sdpa_kernel([self._sdpa_enum]) if self._sdpa_enum is not None else nullcontext()
                    with ctx:
                        out = scaled_dot_product_attention(
                            Q, K, V,
                            attn_mask=fused_bias,   # [B,H,Np,Np] or None
                            dropout_p=self.dropout_p,
                            is_causal=False,
                            scale=self._scale,
                        )
                except RuntimeError as e:
                    if "No available kernel" in str(e):
                        scores = self._build_scores(Q, K, fused_bias) * self._scale
                        attn = torch.softmax(scores, dim=-1)
                        if self._drop is not None and self.training:
                            attn = self._drop(attn)
                        out = torch.matmul(attn, V)
                    else:
                        raise
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
        Q, K, V, fused_bias, B, Np, H, D = self._prep_qkv_bias(batch, mask)
        # 固定 math 路径，保证显式返回权重
        scores = self._build_scores(Q, K, fused_bias) * self._scale
        attn = torch.softmax(scores, dim=-1)
        attn_used = self._drop(attn) if (self._drop is not None and self.training) else attn
        out = torch.matmul(attn_used, V)
        out = out.transpose(1, 2).contiguous().view(B, Np, H * D)
        out = self.out_proj(out)
        return out, attn
