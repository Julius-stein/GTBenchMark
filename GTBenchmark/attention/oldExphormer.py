# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
# from torch_scatter import scatter
# from GTBenchmark.graphgym.register import register_layer
# from GTBenchmark.graphgym.config import cfg

# # ============================================================
# # MultiHeadExGraphAttention — branch-free Exphormer attention
# # ============================================================

# class MultiHeadExGraphAttention(MessagePassing):
#     """
#     Exphormer-style attention (branch-free, JIT-safe).
#     α_ij = exp(clamp((Q_i*K_j*E_ij).sum/sqrt(d))) / Σ_j exp(...)
#     """
#     def __init__(self, dim_h: int, n_heads: int, dropout: float = 0.0,
#                  dim_edge: int = None, clamp: float = 5.0,
#                  use_edge_attr: bool = True, return_attn_weights: bool = False):
#         super().__init__(node_dim=0)
#         assert dim_h % n_heads == 0, "dim_h must be divisible by n_heads"
#         self.dim_h = dim_h
#         self.n_heads = n_heads
#         self.dropout = dropout
#         self.head_dim = dim_h // n_heads
#         self.clamp = clamp
#         self.use_edge_attr = use_edge_attr
#         self.dim_edge = dim_edge
#         self.return_attn_weights = return_attn_weights
#         self.use_bias = False

#         # Linear projections
#         self.WQ = nn.Linear(dim_h, dim_h, bias=self.use_bias)
#         self.WK = nn.Linear(dim_h, dim_h, bias=self.use_bias)
#         self.WV = nn.Linear(dim_h, dim_h, bias=self.use_bias)

#         # Fixed edge projection layer
#         if self.use_edge_attr:
#             in_dim_edge = dim_edge if dim_edge is not None else dim_h
#             self.WE = nn.Linear(in_dim_edge, dim_h, bias=self.use_bias)
#         else:
#             self.register_parameter("WE", None)

#         self.alpha_cache = None
#         self._num_nodes = None

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
#         N = x.size(0)
#         self._num_nodes = N

#         # Linear projections
#         Q = self.WQ(x).view(N, self.n_heads, self.head_dim)
#         K = self.WK(x).view(N, self.n_heads, self.head_dim)
#         V = self.WV(x).view(N, self.n_heads, self.head_dim)

#         if self.use_edge_attr:
#             E = self.WE(edge_attr).view(-1, self.n_heads, self.head_dim)
#         else:
#             E = torch.ones(edge_index.size(1), self.n_heads, self.head_dim,
#                            device=x.device, dtype=x.dtype)

#         out = self.propagate(edge_index=edge_index, Q=Q, K=K, V=V, E=E)
#         out = out.reshape(N, self.dim_h)
#         if self.return_attn_weights:
#             return out, self.alpha_cache
#         return out, None

#     def message(self, Q_i, K_j, V_j, index, E):
#         score = (Q_i * K_j * E).sum(-1, keepdim=True) / math.sqrt(self.head_dim)
#         score = torch.exp(torch.clamp(score, -self.clamp, self.clamp))

#         denom = scatter(score, index, dim=0, dim_size=self._num_nodes, reduce='sum')
#         denom_e = denom.index_select(0, index)
#         alpha = score / (denom_e + 1e-9)
#         self.alpha_cache = alpha

#         if self.training and self.dropout > 0:
#             alpha = F.dropout(alpha, p=self.dropout, training=True)

#         return alpha * V_j

#     def aggregate(self, inputs, index, dim_size=None):
#         return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')


# # ============================================================
# # Framework wrapper (MPAttn-compatible)
# # ============================================================

# @register_layer('ExphormerAttention')
# class ExphormerAttn(nn.Module):
#     """
#     Framework-compatible Exphormer attention layer.
#     - identical interface to MPAttn (forward(batch, mask))
#     - no runtime branches
#     - can optionally return attention weights
#     """
#     def __init__(self, dim_h, num_heads, attn_drop=0.0,
#                  x_name='x', e_name='edge_index', eattr_name='edge_attr',
#                  clamp=5.0,return_attn_weights: bool = False):
#         super().__init__()
#         self.x_name = x_name
#         self.e_name = e_name
#         self.eattr_name = eattr_name

#         # 固定开关
#         self.use_edge_attr = getattr(cfg.gt, 'use_edge_attr', True)
#         self.dim_edge = getattr(cfg.gt, 'dim_edge', dim_h)
#         self.return_attn_weights = return_attn_weights

#         self.attention = MultiHeadExGraphAttention(
#             dim_h=dim_h,
#             n_heads=num_heads,
#             dropout=attn_drop,
#             dim_edge=self.dim_edge,
#             clamp=clamp,
#             use_edge_attr=self.use_edge_attr,
#             return_attn_weights=self.return_attn_weights
#         )

#     def forward(self, batch, mask):
#         x = getattr(batch, self.x_name)
#         edge_index = getattr(batch, self.e_name)
#         edge_attr = getattr(batch, self.eattr_name)

#         if not self.use_edge_attr:
#             edge_attr = torch.ones(edge_index.size(1), self.dim_edge,
#                                    device=x.device, dtype=x.dtype)

#         h, attn = self.attention(x, edge_index, edge_attr)
#         setattr(batch, self.x_name, h)
#         if self.return_attn_weights:
#             batch.attn_weights = attn
#         return batch
