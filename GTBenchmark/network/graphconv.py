import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_network


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):

        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x

@register_network("GraphConv")
class GraphConv(nn.Module):
    def __init__(self):
        super(GraphConv, self).__init__()
        self.in_channels = cfg.share.dim_in
        self.hidden_channels = cfg.gnn.dim_inner
        self.out_channels = cfg.share.dim_out
        self.num_layers = cfg.gnn.layers
        self.dropout = cfg.gnn.dropout
        self.use_bn = cfg.gnn.batch_norm
        self.use_residual=True
        self.use_weight=True
        self.use_init=False
        self.use_act=True

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(self.in_channels, self.hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers):
            self.convs.append(
                GraphConvLayer(self.hidden_channels, self.hidden_channels, self.use_weight, self.use_init))
            self.bns.append(nn.BatchNorm1d(self.hidden_channels))
        self.classifier = nn.Linear(self.hidden_channels, self.out_channels)
        self.activation = F.relu


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self,batch):
        x = batch.x
        edge_index = batch.edge_index 
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
            # layer_.append(x)
        batch.x = self.classifier(x)

        return batch
        # return x
