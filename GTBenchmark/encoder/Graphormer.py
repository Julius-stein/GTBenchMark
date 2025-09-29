import torch
import torch.nn.functional as F
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj, to_networkx

# (batch, node, node, head) -> (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)
# pad graph token: 前面插 1 行 1 列
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)
class NodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_in_degree, num_out_degree,
                 input_dropout=0.0, use_graph_token: bool = True):
        super().__init__()
        num_in_degree = cfg.share.max_indegree
        num_out_degree = cfg.share.max_outdegree
        self.in_degree_encoder = torch.nn.Embedding(num_in_degree+1, embed_dim)
        self.out_degree_encoder = torch.nn.Embedding(num_out_degree+1, embed_dim)
        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.reset_parameters()

    def forward(self, data):
        # data.x: [sumN, F]（外部的 atom embedding 已经准备好）
        in_degree_encoding = self.in_degree_encoder(data.in_degree)
        out_degree_encoding = self.out_degree_encoder(data.out_degree)
        data.x = data.x + in_degree_encoding + out_degree_encoding
        if self.use_graph_token:
            data = add_graph_token(data, self.graph_token)
        data.x = self.input_dropout(data.x)
        return data

    def reset_parameters(self):
        self.in_degree_encoder.weight.data.normal_(std=0.02)
        self.out_degree_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

def add_graph_token(data, token):
    """Helper function to augment a batch of PyG graphs
    with a graph token each. Note that the token is
    automatically replicated to fit the batch.

    Args:
        data: A PyG data object holding a single graph
        token: A tensor containing the graph token values

    Returns:
        The augmented data object.
    """
    B = len(data.batch.unique())
    tokens = torch.repeat_interleave(token, B, 0)
    data.x = torch.cat([tokens, data.x], 0)
    data.batch = torch.cat(
        [torch.arange(0, B, device=data.x.device, dtype=torch.long), data.batch]
    )
    data.batch, sort_idx = torch.sort(data.batch)
    data.x = data.x[sort_idx]
    return data

class BiasEncoderDense(torch.nn.Module):
    """
    Graphormer 官方风格的 dense attention bias:
      - 输入:
          data.attn_bias: [B, N+1, N+1]
          data.spatial_pos: [B, N, N]
          data.attn_edge_type: [B, N, N, Fe]
          (可选) data.edge_input: [B, N, N, D, Fe]  (multi-hop 时)
      - 输出:
          data.attn_bias: [B, H, N+1, N+1]
    """
    def __init__(self, num_heads, num_spatial, num_edges,
                 edge_type="single_hop", multi_hop_max_dist=0,
                 use_graph_token=True):
        super().__init__()
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.use_graph_token = use_graph_token

        self.spatial_encoder = torch.nn.Embedding(num_spatial+1, num_heads, padding_idx=0)
        self.edge_encoder = torch.nn.Embedding(num_edges + 2, num_heads, padding_idx=0)
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = torch.nn.Embedding(num_spatial * num_heads * num_heads, 1)
        if self.use_graph_token:
            self.graph_token_virtual_distance = torch.nn.Embedding(1, num_heads)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.spatial_encoder.weight, std=0.02)
        torch.nn.init.normal_(self.edge_encoder.weight, std=0.02)
        if self.edge_type == "multi_hop":
            torch.nn.init.normal_(self.edge_dis_encoder.weight, std=0.02)
        if self.use_graph_token:
            torch.nn.init.normal_(self.graph_token_virtual_distance.weight, std=0.02)

    def forward(self, data):
        attn_bias = data.attn_bias         # [B,N+1,N+1]
        spatial_pos = data.spatial_pos     # [B,N,N]
        attn_edge_type = getattr(data, "attn_edge_type", None)
        edge_input = getattr(data, "edge_input", None)

        B, N = spatial_pos.size(0), spatial_pos.size(1)

        # base bias
        graph_attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # spatial bias
        spatial_pos_bias = self.spatial_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] += spatial_pos_bias

        # graph token虚拟距离
        if self.use_graph_token:
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] += t
            graph_attn_bias[:, :, 0, :] += t

        # edge bias
        if self.edge_type == "multi_hop":
            assert edge_input is not None
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist,:]

            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3,0,1,2,4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist]
            )
            edge_input = edge_input_flat.reshape(max_dist,B,N,N,self.num_heads).permute(1,2,3,0,4)
            edge_input = (edge_input.sum(-2) / spatial_pos_.float().unsqueeze(-1)).permute(0,3,1,2)
        else:
            assert attn_edge_type is not None
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0,3,1,2)

        graph_attn_bias[:, :, 1:, 1:] += edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        data.attn_bias = graph_attn_bias.contiguous()   # [B,H,N+1,N+1]
        return data

@register_node_encoder("GraphormerBias")
class GraphormerEncoder(torch.nn.Sequential):
    """
    Hybrid 版本：
      - NodeEncoder: 稀疏输入（[sumN,F] + batch）
      - BiasEncoderDense: dense 输入（[B,N,N]）
    """
    def __init__(self, dim_emb):
        encoders = [
            NodeEncoder(
                dim_emb,
                cfg.posenc_GraphormerBias.num_in_degrees,
                cfg.posenc_GraphormerBias.num_out_degrees,
                cfg.gt.input_dropout,
                cfg.posenc_GraphormerBias.use_graph_token
            ),
            BiasEncoderDense(
                cfg.gt.attn_heads,
                cfg.posenc_GraphormerBias.num_spatial_types,
                cfg.dataset.edge_encoder_num_types,
                edge_type="multi_hop",
                multi_hop_max_dist=cfg.posenc_GraphormerBias.multi_hop_max_dist,
                use_graph_token=cfg.posenc_GraphormerBias.use_graph_token
            )
        ]
        if cfg.posenc_GraphormerBias.node_degrees_only:
            encoders = encoders[:1]
        super().__init__(*encoders)
