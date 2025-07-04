import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from models.FFN import FeedForwardNetwork

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim):
        super(EncoderLayer, self).__init__()

        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, dropout=attention_dropout_rate, num_heads=num_heads, batch_first=True)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None):
        attn_bias = self.linear_bias(attn_bias).permute(0, 3, 1, 2)
        qlen = attn_bias.shape[-1]
        attn_bias = attn_bias.reshape(-1, qlen, qlen)


        y = self.self_attention_norm(x)
        y, _ = self.self_attention(y, y, y, attn_mask=attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class GT(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        num_global_node,
        attention_dropout_rate,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.n_layers = n_layers
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_global_node = num_global_node
        self.graph_token = nn.Embedding(self.num_global_node, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(self.num_global_node, attn_bias_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        attn_bias, x = batched_data.attn_bias, batched_data.x
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        node_feature = self.node_encoder(x)         # [n_graph, n_node, n_hidden]
        if perturb is not None:
            node_feature += perturb

        global_node_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        node_feature = torch.cat([node_feature, global_node_feature], dim=1)

        graph_attn_bias = torch.cat([graph_attn_bias, self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(2).
                                     repeat(n_graph, 1, n_node, 1)], dim=1)
        graph_attn_bias = torch.cat(
            [graph_attn_bias, self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(0).
            repeat(n_graph, n_node+self.num_global_node, 1, 1)], dim=2)

        # transfomrer encoder
        output = self.input_dropout(node_feature)

        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        # output part
        output = self.downstream_out_proj(output[:, 0, :])
        return F.log_softmax(output, dim=1)

