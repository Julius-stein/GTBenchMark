# import networkx as nx
# import torch
# import torch.nn.functional as F
# from GTBenchmark.graphgym.config import cfg
# from GTBenchmark.graphgym.register import register_node_encoder
# from torch_geometric.utils import to_dense_adj, to_networkx

# # (batch, node, node, head) -> (batch, head, node, node)
# BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)
# # pad graph token: 前面插 1 行 1 列
# INSERT_GRAPH_TOKEN = (1, 0, 1, 0)


# class BiasEncoder(torch.nn.Module):
#     """
#     Dense（非扁平）版本，保留以兼容非 BatchedAttention 的路径。
#     产生 data.attn_bias: [B, H, N, N]
#     """
#     def __init__(self, num_heads: int, num_spatial_types: int,
#                  num_edge_types: int, use_graph_token: bool = True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.spatial_encoder = torch.nn.Embedding(num_spatial_types + 1, num_heads)
#         self.edge_dis_encoder = torch.nn.Embedding(num_spatial_types * num_heads * num_heads, 1)
#         self.edge_encoder = torch.nn.Embedding(num_edge_types, num_heads)

#         self.use_graph_token = use_graph_token
#         if self.use_graph_token:
#             self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.spatial_encoder.weight.data.normal_(std=0.02)
#         self.edge_encoder.weight.data.normal_(std=0.02)
#         self.edge_dis_encoder.weight.data.normal_(std=0.02)
#         if self.use_graph_token:
#             self.graph_token.data.normal_(std=0.02)

#     def forward(self, data):
#         spatial_types = self.spatial_encoder(data.spatial_types)
#         spatial_encodings = to_dense_adj(data.graph_index, data.batch, spatial_types)
#         bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE)  # [B,H,N,N]

#         if hasattr(data, "shortest_path_types"):
#             edge_types = self.edge_encoder(data.shortest_path_types)
#             edge_encodings = to_dense_adj(data.graph_index, data.batch, edge_types)
#             spatial_distances = to_dense_adj(data.graph_index, data.batch, data.spatial_types)
#             spatial_distances = spatial_distances.float().clamp(min=1.0).unsqueeze(1)

#             B, N, _, max_dist, H = edge_encodings.shape
#             edge_encodings = edge_encodings.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
#             abc = self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)
#             edge_encodings = torch.bmm(edge_encodings, abc)
#             edge_encodings = edge_encodings.reshape(max_dist, B, N, N, self.num_heads).permute(1, 2, 3, 0, 4)
#             edge_encodings = edge_encodings.sum(-2).permute(BATCH_HEAD_NODE_NODE) / spatial_distances
#             bias += edge_encodings

#         if self.use_graph_token:
#             bias = F.pad(bias, INSERT_GRAPH_TOKEN)
#             bias[:, :, 1:, 0] = self.graph_token
#             bias[:, :, 0, :] = self.graph_token

#         data.attn_bias = bias.contiguous()  # [B,H,N,N]
#         return data


# def add_graph_token(data, token):
#     """
#     扁平拼接 graph token，并记录每个图的 token 全局下标到 data.token_idx
#     """
#     B = int(data.batch.max().item()) + 1
#     tokens = token.repeat(B, 1)  # [B, D]

#     data.x = torch.cat([tokens, data.x], dim=0)

#     cat_batch = torch.cat([
#         torch.arange(0, B, device=data.x.device, dtype=torch.long),
#         data.batch
#     ], dim=0)
#     cat_batch, sort_idx = torch.sort(cat_batch)

#     data.x     = data.x[sort_idx]
#     data.batch = cat_batch

#     inv = torch.empty_like(sort_idx)
#     inv[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
#     data.token_idx = inv[:B]  # [B]
#     return data


# class BiasEncoderFlat(torch.nn.Module):
#     """
#     扁平版本：直接在 N_total×N_total 上构造偏置，输出 data.attn_bias: [H, N, N]
#     依赖：
#       - data.edge_index: [2, E]（PyG 合批后的全局下标）
#       - data.batch: [N]
#       - data.spatial_types: [E]
#       - data.shortest_path_types: [E, D]（可选；0 表示无效）
#       - 若使用 graph token，请确保 NodeEncoder 已先运行（并写入 data.token_idx）
#     """
#     def __init__(self, num_heads: int, num_spatial_types: int,
#                  num_edge_types: int, use_graph_token: bool = True):
#         super().__init__()
#         self.num_heads = int(num_heads)
#         self.use_graph_token = bool(use_graph_token)

#         self.spatial_encoder = torch.nn.Embedding(num_spatial_types + 1, num_heads)
#         self.edge_encoder = torch.nn.Embedding(num_edge_types, num_heads)

#         # 默认按配置里“距离桶上限”建，若遇到不同 D，会按需重建
#         self.edge_dis_kernel = torch.nn.Parameter(
#             torch.empty(cfg.posenc_GraphormerBias.num_spatial_types, self.num_heads, self.num_heads)
#         )

#         if self.use_graph_token:
#             self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.spatial_encoder.weight.data.normal_(std=0.02)
#         self.edge_encoder.weight.data.normal_(std=0.02)
#         torch.nn.init.normal_(self.edge_dis_kernel, std=0.02)
#         if self.use_graph_token:
#             self.graph_token.data.normal_(std=0.02)

#     @torch.no_grad()
#     def _maybe_init_edge_dis_kernel(self, D: int, H: int):
#         if self.edge_dis_kernel is None or tuple(self.edge_dis_kernel.shape) != (D, H, H):
#             new = torch.empty(D, H, H, device=self.edge_dis_kernel.device if self.edge_dis_kernel is not None else None,
#                               dtype=self.edge_dis_kernel.dtype if self.edge_dis_kernel is not None else None)
#             torch.nn.init.normal_(new, std=0.02)
#             self.edge_dis_kernel = torch.nn.Parameter(new)

#     @staticmethod
#     def _scatter_add_bias(bias_hnn: torch.Tensor,
#                           rows: torch.Tensor, cols: torch.Tensor,
#                           val_eh: torch.Tensor):
#         """
#         bias_hnn: [H, N, N]
#         rows, cols: [E]
#         val_eh: [E, H]  -> 按 head 分别 scatter_add 到 (rows, cols)
#         """
#         H, N, _ = bias_hnn.shape
#         lin = rows * N + cols  # [E]
#         bias_flat = bias_hnn.view(H, N * N)     # [H, N*N]
#         val_he = val_eh.t().contiguous()        # [H, E]
#         for h in range(H):
#             bias_flat[h].scatter_add_(0, lin, val_he[h])

#     def forward(self, data):
#         H = self.num_heads
#         N = data.x.size(0)

#         bias = data.x.new_zeros(H, N, N)  # [H,N,N]

#         edge_index = data.graph_index if hasattr(data, "graph_index") else data.edge_index
#         row, col = edge_index[0], edge_index[1]
#         E = row.numel()

#         # 1) spatial types
#         if hasattr(data, "spatial_types") and data.spatial_types.numel() == E:
#             val_eh = self.spatial_encoder(data.spatial_types.long())  # [E,H]
#             self._scatter_add_bias(bias, row, col, val_eh)

#         # 2) shortest_path_types
#         if hasattr(data, "shortest_path_types") and data.shortest_path_types.dim() == 2:
#             E2, D = data.shortest_path_types.shape
#             assert E2 == E, "shortest_path_types size mismatch with edge_index"
#             self._maybe_init_edge_dis_kernel(D, H)

#             edge_feat = self.edge_encoder(data.shortest_path_types.long())  # [E,D,H]
#             with torch.no_grad():
#                 # 把 0 桶当作无效
#                 self.edge_encoder.weight.data[0].zero_()

#             # per-distance 的 [H,H] 混合: out[e,d,h] = sum_{h'} edge_feat[e,d,h'] * K[d,h',h]
#             mixed = torch.einsum('dxy,edx->edy', self.edge_dis_kernel, edge_feat)  # [E,D,H]

#             active = (data.shortest_path_types != 0).to(bias.dtype)  # [E,D]
#             cnt = active.sum(dim=1).clamp_min_(1.0)                  # [E]
#             val_eh = mixed.sum(dim=1) / cnt.unsqueeze(-1)            # [E,H]

#             self._scatter_add_bias(bias, row, col, val_eh)

#         # 3) graph token 偏置（要求 add_graph_token 已写入 data.token_idx）
#         if self.use_graph_token and hasattr(data, "batch") and hasattr(data, "token_idx"):
#             # 每个节点所属图的 token 下标
#             t_per_node = data.token_idx[data.batch]           # [N]
#             gtok = self.graph_token.view(H, 1)                # [H,1]

#             h_idx = torch.arange(H, device=bias.device)[:, None]  # [H,1]
#             n_idx = torch.arange(N, device=bias.device)[None, :]  # [1,N]

#             # token 行/列
#             bias[h_idx, t_per_node[None, :], n_idx] = gtok    # bias[:, t_i, j] = gtok
#             bias[h_idx, n_idx, t_per_node[None, :]] = gtok    # bias[:, i, t_i] = gtok

#         data.attn_bias = bias.contiguous()  # [H,N,N]
#         return data


# class NodeEncoder(torch.nn.Module):
#     def __init__(self, embed_dim, num_in_degree, num_out_degree,
#                  input_dropout=0.0, use_graph_token: bool = True):
#         super().__init__()
#         self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim)
#         self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim)
#         self.use_graph_token = use_graph_token
#         if self.use_graph_token:
#             self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
#         self.input_dropout = torch.nn.Dropout(input_dropout)
#         self.reset_parameters()

#     def forward(self, data):
#         in_degree_encoding = self.in_degree_encoder(data.in_degrees)
#         out_degree_encoding = self.out_degree_encoder(data.out_degrees)
#         data.x = (data.x if data.x.size(1) > 0 else 0) + in_degree_encoding + out_degree_encoding
#         if self.use_graph_token:
#             data = add_graph_token(data, self.graph_token)
#         data.x = self.input_dropout(data.x)
#         return data

#     def reset_parameters(self):
#         self.in_degree_encoder.weight.data.normal_(std=0.02)
#         self.out_degree_encoder.weight.data.normal_(std=0.02)
#         if self.use_graph_token:
#             self.graph_token.data.normal_(std=0.02)


# # @register_node_encoder("GraphormerBias")
# class GraphormerEncoder(torch.nn.Sequential):
#     """
#     - BatchedAttention：NodeEncoder -> BiasEncoderFlat（扁平偏置，给 BatchedAttn / Flex）
#     - 其他注意力：NodeEncoder -> BiasEncoder（dense 偏置，给原始 SDPA 路径）
#     """
#     def __init__(self, dim_emb, *args, **kwargs):
#         if cfg.gt.attn_type == "BatchedAttention":
#             encoders = [
#                 NodeEncoder(
#                     dim_emb,
#                     cfg.posenc_GraphormerBias.num_in_degrees,
#                     cfg.posenc_GraphormerBias.num_out_degrees,
#                     cfg.gt.input_dropout,
#                     cfg.posenc_GraphormerBias.use_graph_token
#                 ),
#                 BiasEncoderFlat(
#                     cfg.gt.attn_heads,
#                     cfg.posenc_GraphormerBias.num_spatial_types,
#                     cfg.dataset.edge_encoder_num_types,
#                     cfg.posenc_GraphormerBias.use_graph_token
#                 ),
#             ]
#         else:
#             encoders = [
#                 BiasEncoder(
#                     cfg.gt.attn_heads,
#                     cfg.posenc_GraphormerBias.num_spatial_types,
#                     cfg.dataset.edge_encoder_num_types,
#                     cfg.posenc_GraphormerBias.use_graph_token
#                 ),                
#                 NodeEncoder(
#                     dim_emb,
#                     cfg.posenc_GraphormerBias.num_in_degrees,
#                     cfg.posenc_GraphormerBias.num_out_degrees,
#                     cfg.gt.input_dropout,
#                     cfg.posenc_GraphormerBias.use_graph_token
#                 ),
#             ]
#         if cfg.posenc_GraphormerBias.node_degrees_only:
#             encoders = encoders[1:]
#         super().__init__(*encoders)
