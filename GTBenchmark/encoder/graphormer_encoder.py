import networkx as nx
import torch
import torch.nn.functional as F
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_node_encoder
from GTBenchmark.transform.graph2dense import to_dense_adj

# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)

# Inserts a leading 0 row and a leading 0 column with F.pad
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)





def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset.to(x.device)
    return x



class BiasEncoder(torch.nn.Module):
    def __init__(self,
                 num_heads: int,
                 num_spatial_types: int,
                 num_edge_types: int,
                 use_graph_token: bool = True,
                 # 新增：保持与 Graphormer 一致的两个开关
                 edge_type: str = "multi_hop",           # "multi_hop" | "single_hop"
                 multi_hop_max_dist: int = 0,            # 0 表示不裁剪
                 num_edge_dis: int = None):              # 可不传，默认= num_spatial_types
        super().__init__()
        self.num_heads = num_heads
        self.use_graph_token = use_graph_token

        # —— 新增属性 —— #
        self.edge_type = edge_type
        self.multi_hop_max_dist = int(multi_hop_max_dist)
        self.num_edge_dis = int(num_edge_dis) if num_edge_dis is not None else int(num_spatial_types)

        # 距离偏置：0 作为 pad（不连通）
        self.spatial_encoder = torch.nn.Embedding(num_spatial_types,
                                                  num_heads,
                                                  padding_idx=0)
        # 边类型偏置：0 作为 pad（我们在预处理里把“无边”与“-1”都映射到 0）
        self.edge_encoder = torch.nn.Embedding(num_edge_types+ 1,
                                               num_heads,
                                               padding_idx=0)

        # multi-hop 时才用
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = torch.nn.Embedding(
                self.num_edge_dis * num_heads * num_heads, 1
            )

        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        self.edge_encoder.weight.data.normal_(std=0.02)
        if hasattr(self, "edge_dis_encoder"):
            self.edge_dis_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

    def forward(self, data):
        """
        旁路读取：优先从 batch._side 取三件套（spatial_pos / attn_edge_type / edge_input）
        产出 data.attn_bias：形状 [B*H, N(+1), N(+1)]（含 graph token 时 N+1）
        """
        side = getattr(data, "_side", None)
        if side is None:
            # 没有旁路就走你原来的实现（如果还保留了 to_dense_adj 的分支）
            # 或者直接报错提示 DataLoader 要换成附着旁路版
            raise RuntimeError("BiasEncoder: missing side inputs (batch._side). "
                               "请使用附着旁路的 collate_fn 或在前向里构造 _side。")

        device = data.x.device
        spatial_pos = side['spatial_pos'].to(device)          # [B,N,N] long
        B, N, _ = spatial_pos.shape
        H = self.num_heads

        # 1) 距离偏置
        bias = self.spatial_encoder(spatial_pos).permute(0, 3, 1, 2)    # [B,H,N,N]

        # 2) 边偏置
        if self.edge_type == "multi_hop":
            # Graphormer 的 “平均后按距离衰减再映射” 公式
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            ein = side['edge_input'].to(device)                          # [B,N,N,D,Fe]
            # 统一 D：必要时裁剪
            if self.multi_hop_max_dist > 0 and ein.size(-2) > self.multi_hop_max_dist:
                ein = ein[:, :, :, :self.multi_hop_max_dist, :]
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)

            # -1（pad）→ 0（embedding 的 padding_idx）
            ein = ein.clamp(min=0)
            # [B,N,N,D,Fe] --edge_encoder(mean over Fe)--> [B,N,N,D,H]
            e = self.edge_encoder(ein).mean(-2)
            # e = self.edge_encoder(ein)
            D = e.size(-2)
            # [B,N,N,D,H] -> [D, B*N*N, H]
            e_flat = e.permute(3, 0, 1, 2, 4).reshape(D, -1, H)
            # W: [D,H,H]
            W = self.edge_dis_encoder.weight.reshape(-1, H, H)[:D]
            # [D, B*N*N, H] @ [D,H,H] -> [D, B*N*N, H]
            mixed = torch.bmm(e_flat, W).reshape(D, B, N, N, H).permute(1, 2, 3, 0, 4)
            # sum over D, 再 / 距离（避免除 0）
            edge_bias = (mixed.sum(-2) / (spatial_pos_.float().unsqueeze(-1) + 1e-9))\
                        .permute(0, 3, 1, 2)                                    # [B,H,N,N]
        else:
            # single_hop：直接对 attn_edge_type 做嵌入并按 Fe 平均
            aetype = side['attn_edge_type'].to(device)                          # [B,N,N,Fe]
            edge_bias = self.edge_encoder(aetype).mean(-2).permute(0, 3, 1, 2)  # [B,H,N,N]

        bias = bias + edge_bias                                                # [B,H,N,N]

        # 3) graph token（如果使用）
        if self.use_graph_token:
            bias = F.pad(bias, (1, 0, 1, 0))                                   # [B,H,N+1,N+1]
            t = self.graph_token.to(bias.device).view(1, H, 1)                 # [1,H,1]
            bias[:, :, 1:, 0] += t
            bias[:, :, 0, 1:] += t

        data.attn_bias = bias  # [B, H, Np, Np]
        return data



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


class NodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_in_degree, num_out_degree,
                 input_dropout=0.0, use_graph_token: bool = True):
        super().__init__()
        self.atom_encoder = torch.nn.Embedding(
            cfg.dataset.node_encoder_num_types + 1,  # +1 for padding
            embed_dim,
            padding_idx=0
        )
        self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim, padding_idx=0)
        self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim, padding_idx=0)
        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.reset_parameters()

    def forward(self, data):
        # 逻辑等同TypeDictNode
        # x = data.x
        # if x.dim() == 1:
        #     x = x.unsqueeze(1)
        # x = convert_to_single_emb(x, offset=512)  # long ids, 0 reserved for pad
        # # —— 关键：按列求和（与官方相同）——
        # node_feature = self.atom_encoder(x).sum(dim=-2)  # [N, C]
        if data.x.size(1) > 0:
            data.x = data.x \
                + self.in_degree_encoder(data.in_degrees) \
                + self.out_degree_encoder(data.out_degrees)
        else:
             data.x =  self.in_degree_encoder(data.in_degrees) \
                + self.out_degree_encoder(data.out_degrees)

        if self.use_graph_token:
            data = add_graph_token(data, self.graph_token)

        data.x = self.input_dropout(data.x)
        return data

    def reset_parameters(self):
        self.atom_encoder.weight.data.normal_(std=0.02)
        self.in_degree_encoder.weight.data.normal_(std=0.02)
        self.out_degree_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)



@register_node_encoder("GraphormerBias")
class GraphormerEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb, *args, **kwargs):
        encoders = [
            BiasEncoder(
                cfg.gt.attn_heads,
                cfg.posenc_GraphormerBias.num_spatial_types,
                cfg.dataset.edge_encoder_num_types,
                cfg.posenc_GraphormerBias.use_graph_token
            ),
            NodeEncoder(
                dim_emb,
                cfg.posenc_GraphormerBias.num_in_degrees,
                cfg.posenc_GraphormerBias.num_out_degrees,
                cfg.gt.input_dropout,
                cfg.posenc_GraphormerBias.use_graph_token
            ),
        ]
        if cfg.posenc_GraphormerBias.node_degrees_only:  # No attn. bias encoder
            encoders = encoders[1:]
        super().__init__(*encoders)
