import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.register import register_layer
from GTBenchmark.graphgym.config import cfg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class MultiHeadGraphAttentionLayer(MessagePassing):
    def __init__(self, dim_h:int, n_heads:int, dropout):
        super().__init__(node_dim=0)
        self.dim_h = dim_h
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = dim_h // n_heads
        self.eij = None

        self.WQ = nn.Linear(dim_h, dim_h)
        self.WK = nn.Linear(dim_h, dim_h)
        self.WV = nn.Linear(dim_h, dim_h)


    def forward(self, x, edge_index, edge_bias):
        Q_h, K_h, V_h = self.WQ(x), self.WK(x), self.WV(x)

        Q = Q_h.view(-1, self.n_heads, self.head_dim)
        K = K_h.view(-1, self.n_heads, self.head_dim)
        V = V_h.view(-1, self.n_heads, self.head_dim)

        h = self.propagate(edge_index=edge_index, Q=Q, K=K, V=V, edge_bias=edge_bias).view(-1, self.dim_h)
        return h, self.eij
    
    def message(self, Q_i, K_j, V_j, index, edge_bias):
        d_k = Q_i.size(-1)
        qijk = (Q_i * K_j).sum(dim=-1) / np.sqrt(d_k)
        if edge_bias is not None:
            qijk += edge_bias

        alpha = softmax(qijk, index)  # Log-Sum-Exp trick used. No need for clipping (-5,5)
        self.eij = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        return alpha.view(-1, self.n_heads, 1) * V_j

@register_layer('MessagePassingAttention')
class MPAttn(nn.Module):
    def __init__(self, dim_h, num_heads, x_name='x', b_name='attn_bias', e_name='edge_index'):
        super(MPAttn, self).__init__()
        self.x_name = x_name
        self.b_name = b_name
        self.e_name = e_name
        self.attention = MultiHeadGraphAttentionLayer(dim_h, num_heads, dropout=cfg.gt.attn_dropout)

    def forward(self, batch):
        x, edge_index, edge_bias = getattr(batch, self.x_name), getattr(batch, self.e_name), getattr(batch, self.b_name, None)
        h, _ = self.attention(x, edge_index, edge_bias)
        setattr(batch, self.x_name, h)
        return batch
