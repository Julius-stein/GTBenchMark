# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn.functional import scaled_dot_product_attention
# from torch.nn.attention import sdpa_kernel, SDPBackend
# from GTBenchmark.graphgym.config import cfg
# from contextlib import nullcontext
# from typing import Optional
# from torch.nn.attention.flex_attention import flex_attention as _flex_attention

# _BACKEND_MAP = {
#     "auto":      None,
#     "cudnn":     SDPBackend.CUDNN_ATTENTION,
#     "flash":     SDPBackend.FLASH_ATTENTION,
#     "efficient": SDPBackend.EFFICIENT_ATTENTION,
#     "math":      SDPBackend.MATH,
#     "flex":      "flex",
# }

# def _to_additive_from_full_mask(full_mask: Tensor, N: int) -> Tensor:
#     """
#     full_mask: [1,1,N,N] 或 [1,H,N,N]，bool(True=keep) 或 float(0/-inf)
#     返回：     [1,1,N,N] 的 additive 浮点掩码
#     说明：BatchedAttention 是“扁平文档注意力”，batch 维固定为 1。
#     """
#     assert full_mask.dim() == 4 and full_mask.size(0) == 1
#     _, c, N_in, N_in2 = full_mask.shape
#     assert N_in == N_in2 == N
#     if full_mask.dtype == torch.bool:
#         add = torch.zeros_like(full_mask, dtype=torch.float, device=full_mask.device)
#         add = add.masked_fill(~full_mask, float('-inf'))
#     else:
#         add = full_mask
#     # 统一成 [1,1,N,N]，方便后续 expand 到 H
#     if c != 1:
#         add = add[:, :1]  # 取一个通道；多头偏置请走 attn_bias
#     return add  # [1,1,N,N]

# from GTBenchmark.graphgym.register import register_layer

# @register_layer('BatchedAttention')
# class BatchedAttn(nn.Module):
#     """
#     扁平文档注意力（PyG 的 batch 展平后 N_total×N_total，同图可见）：
#       - Mask 来自 Basemask._forward_flat：
#           * Flex 时：batch.flex_block_mask 为 BlockMask，mask=None
#           * 非 Flex：mask 为 additive/full，shape [1,1,N,N] 或 [1,H,N,N]，batch.flex_block_mask=None
#       - 结构偏置（attn_bias）：
#           * 支持 [H,N,N] 或 [1,H,N,N]
#       - Flex：score_mod 注入（fused_bias 展平后按 (b,h,q,k) 定位）
#       - 非 Flex：合并成 SDPA 的 attn_mask 走 scaled_dot_product_attention
#       - return_attn_weights=True：固定 math 路径并显式返回 (out, attn)
#     """
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout_p: float = 0.0,
#         x_name: str = 'x',
#         bias_name: str = 'attn_bias',     # [H,N,N] 或 [1,H,N,N]
#         backend: str = "auto",            # auto / flex / efficient / flash / cudnn / math
#         return_attn_weights: bool = False
#     ):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         self.x_name = x_name
#         self.bias_name = bias_name
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.dropout_p = float(dropout_p)
#         self._scale = 1.0 / math.sqrt(self.head_dim)
#         self.return_attn_weights = bool(return_attn_weights)

#         if backend not in _BACKEND_MAP:
#             raise ValueError(f"Unknown backend '{backend}', valid keys: {list(_BACKEND_MAP)}")
#         self.backend = backend

#         # ---- 由 cfg 决定是否走 Flex（与 Basemask 同步）----
#         # 你在别处会设置：cfg.share.can_flex = (dropout==0) and cfg.gt.use_flex
#         cfg.share.can_flex = (self.dropout_p == 0.0) and cfg.gt.use_flex
#         if self.backend == "auto":
#             self._use_flex = cfg.share.can_flex 
#         else:
#             self._use_flex = False

#         if self._use_flex or self.backend == "math":
#             self._sdpa_enum = None
#         else:
#             chosen = self.backend if self.backend != "auto" else "efficient"
#             self._sdpa_enum = _BACKEND_MAP.get(chosen, SDPBackend.EFFICIENT_ATTENTION)

#         self.in_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self._drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

#         # 绑定 forward
#         if self.return_attn_weights:
#             self.forward = self._forward_with_weights  # type: ignore[assignment]
#         else:
#             self.forward = self._forward_no_weights    # type: ignore[assignment]

#     # ------------- helpers -------------
#     @staticmethod
#     def _reshape_bias(attn_bias: Tensor, H: int, N: int, device, dtype) -> Tensor:
#         """
#         支持：
#           - [H, N, N]
#           - [1, H, N, N]
#         输出：float，[1,H,N,N]
#         """
#         if attn_bias.dim() == 3:
#             if attn_bias.size(0) != H or attn_bias.size(1) != N or attn_bias.size(2) != N:
#                 raise RuntimeError(f"attn_bias shape {tuple(attn_bias.shape)} expect (H,N,N)")
#             E = attn_bias.unsqueeze(0)  # -> [1,H,N,N]
#         elif attn_bias.dim() == 4:
#             if attn_bias.size(0) != 1 or attn_bias.size(1) != H or attn_bias.size(2) != N or attn_bias.size(3) != N:
#                 raise RuntimeError(f"attn_bias shape {tuple(attn_bias.shape)} expect (1,H,N,N)")
#             E = attn_bias
#         else:
#             raise RuntimeError(f"attn_bias dim={attn_bias.dim()} not in {{3,4}}")
#         return E.to(device=device, dtype=dtype)

#     @staticmethod
#     def _get_cached_additive(batch, pad_mask: Optional[Tensor], N: int, device, dtype):
#         """
#         把 Basemask 给的 full/additive/padding mask 统一变成 [1,H,N,N] 的浮点 additive。
#         """
#         if pad_mask is None:
#             return None
#         cache = getattr(batch, "_attn_cache", None)
#         if cache is None:
#             cache = {}
#             setattr(batch, "_attn_cache", cache)
#         key = (id(pad_mask), N, device, dtype)
#         add = cache.get(key, None)
#         if add is None:
#             add = _to_additive_from_full_mask(pad_mask.to(device), N).to(dtype)  # [1,1,N,N]
#             cache[key] = add
#         return add  # [1,1,N,N]

#     @staticmethod
#     def _build_scores(Q: Tensor, K: Tensor, fused_bias: Optional[Tensor]) -> Tensor:
#         # Q,K: [1,H,N,D]
#         scores = torch.matmul(Q, K.transpose(-2, -1))  # [1,H,N,N]
#         if fused_bias is not None:
#             scores = scores + fused_bias
#         return scores

#     def _prep_qkv(self, x: Tensor):
#         """
#         x: [N, D_model] （扁平）
#         -> Q,K,V: [1,H,N,D]
#         """
#         assert x.dim() == 2, f"x should be [N, D], got {tuple(x.shape)}"
#         N, _ = x.shape
#         H, D = self.num_heads, self.head_dim
#         device, dtype = x.device, x.dtype

#         qkv = self.in_proj(x)                       # [N, 3D_model]
#         q, k, v = qkv.split(self.embed_dim, dim=-1) # [N, D_model] ×3
#         Q = q.view(N, H, D).permute(1, 0, 2).unsqueeze(0).contiguous()  # [1,H,N,D]
#         K = k.view(N, H, D).permute(1, 0, 2).unsqueeze(0).contiguous()  # [1,H,N,D]
#         V = v.view(N, H, D).permute(1, 0, 2).unsqueeze(0).contiguous()  # [1,H,N,D]
#         return Q, K, V, N, H, D, device, dtype

#     # ---- Flex 路径：把 fused_bias 注入到 score_mod，并传入 block_mask ----
#     def _flex_attention(self, Q, K, V, fused_bias: Optional[Tensor], block_mask):
#         H = self.num_heads
#         B = 1
#         _, _, N, _ = Q.shape  # [1,H,N,D]

#         if fused_bias is None:
#             def score_mod(score, b, h, q_idx, k_idx):
#                 return score
#             return _flex_attention(Q, K, V,
#                                    score_mod=score_mod,
#                                    block_mask=block_mask,
#                                    scale=self._scale,
#                                    kernel_options=None)

#         fused_flat = fused_bias.contiguous().reshape(-1)  # [1*H*N*N]
#         def score_mod(score, b, h, q_idx, k_idx):
#             # 线性下标：(b*H + h)*N + q  -> *N + k
#             lin = (((b * H) + h) * N + q_idx) * N + k_idx
#             return score + fused_flat[lin.long()]
#         return _flex_attention(Q, K, V,
#                                score_mod=score_mod,
#                                block_mask=block_mask,
#                                scale=self._scale,
#                                kernel_options=None)

#     # ------------- forward（不返回权重）-------------
#     def _forward_no_weights(self, batch, mask: Optional[Tensor] = None):
#         x: Tensor = getattr(batch, self.x_name)   # [N, D_model]
#         Q, K, V, N, H, D, device, dtype = self._prep_qkv(x)

#         # 结构偏置（可选）
#         raw_bias = getattr(batch, self.bias_name, None)
#         E = None
#         if raw_bias is not None:
#             E = self._reshape_bias(raw_bias, H, N, device, dtype)  # [1,H,N,N]

#         if self._use_flex:
#             block_mask = getattr(batch, "flex_block_mask", None)
#             out = self._flex_attention(Q, K, V, fused_bias=E, block_mask=block_mask)
#         else:
#             # additive（来自 Basemask 的 full/padding mask）
#             add = self._get_cached_additive(batch, mask, N, device, dtype)  # [1,1,N,N] or None
#             if E is None and add is None:
#                 fused_bias = None
#             else:
#                 fused_bias = add.expand(1, H, N, N) if add is not None else None
#                 if E is not None:
#                     fused_bias = E if fused_bias is None else (E + fused_bias)

#             if self.backend == "math":
#                 scores = self._build_scores(Q, K, fused_bias) * self._scale  # [1,H,N,N]
#                 attn = torch.softmax(scores, dim=-1)
#                 if self._drop is not None and self.training:
#                     attn = self._drop(attn)
#                 out = torch.matmul(attn, V)  # [1,H,N,D]
#             else:
#                 ctx = sdpa_kernel([self._sdpa_enum]) if self._sdpa_enum is not None else nullcontext()
#                 with ctx:
#                     out = scaled_dot_product_attention(
#                         Q, K, V,
#                         attn_mask=fused_bias,   # [1,H,N,N] or None
#                         dropout_p=self.dropout_p,
#                         is_causal=False,
#                         scale=self._scale,
#                     )  # [1,H,N,D]

#         out = out.squeeze(0).permute(1, 0, 2).contiguous().view(N, H * D)  # -> [N, D_model]
#         out = self.out_proj(out)
#         setattr(batch, self.x_name, out)
#         return batch

#     # ------------- forward（返回权重）-------------
#     def _forward_with_weights(self, batch, mask: Optional[Tensor] = None):
#         # 固定 math 路径，显式返回权重
#         x: Tensor = getattr(batch, self.x_name)   # [N, D_model]
#         Q, K, V, N, H, D, device, dtype = self._prep_qkv(x)

#         raw_bias = getattr(batch, self.bias_name, None)
#         E = None
#         if raw_bias is not None:
#             E = self._reshape_bias(raw_bias, H, N, device, dtype)  # [1,H,N,N]

#         add = self._get_cached_additive(batch, mask, N, device, dtype)  # [1,1,N,N] or None
#         fused_bias = add.expand(1, H, N, N) if add is not None else None
#         if E is not None:
#             fused_bias = E if fused_bias is None else (E + fused_bias)

#         scores = self._build_scores(Q, K, fused_bias) * self._scale  # [1,H,N,N]
#         attn = torch.softmax(scores, dim=-1)
#         attn_used = self._drop(attn) if (self._drop is not None and self.training) else attn
#         out = torch.matmul(attn_used, V)  # [1,H,N,D]

#         out = out.squeeze(0).permute(1, 0, 2).contiguous().view(N, H * D)
#         out = self.out_proj(out)
#         return out, attn.squeeze(0)   # [N, D_model], [H,N,N]
