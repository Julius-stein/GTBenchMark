import torch
import torch.nn as nn
import torch.nn.functional as F

import GTBenchmark.graphgym.models.head  # noqa, register module
import GTBenchmark.graphgym.register as register
import torch_geometric.nn as pyg_nn
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_network
from GTBenchmark.network.utils import FeatureEncoder
from torch_geometric.nn import (Sequential, Linear, HeteroConv, GraphConv, SAGEConv, HGTConv, GATConv)



@register_network('ANS-GT')
class ANS_GT(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.dim_h      = cfg.gt.dim_hidden
        self.input_drop = nn.Dropout(cfg.gt.input_dropout)
        self.activation = register.act_dict[cfg.gt.act]
        self.batch_norm = cfg.gt.batch_norm
        self.layer_norm = cfg.gt.layer_norm
        self.l2_norm    = cfg.gt.l2_norm
        GNNHead         = register.head_dict[cfg.gt.head]

        self.encoder = FeatureEncoder()
        
        self.num_virtual_nodes = cfg.gt.virtual_nodes
        self.dim_in = dim_in

        try:
            self.postfixed_local_model = False
            layer_type = cfg.gt.layer_type
            if '--' in layer_type:
                self.postfixed_local_model = True
                layer_type, postfixed_gnn_type = layer_type.split('--')
            # if '-' in layer_type:
            #     self.prefixed_local_model = True
            #     prefixed_gnn_type, layer_type = layer_type.split('-')
            if '+' in layer_type:
                local_gnn_type, global_model_type = layer_type.split('+')
            else:
                if layer_type in ['Transformer']:
                    local_gnn_type, global_model_type = 'None', layer_type
                else:
                    local_gnn_type, global_model_type = layer_type, 'None'
        except:
            raise ValueError(f"Unexpected layer type format: {layer_type}")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(cfg.gt.layers):
            conv = register.layer_dict['GraphTransformerLayer'](dim_h=self.dim_h)
            self.convs.append(conv)

            if self.layer_norm:
                self.norms.append(nn.LayerNorm(self.dim_h))
            elif self.batch_norm:
                self.norms.append(nn.BatchNorm1d(self.dim_h))
            

        self.post_gt = GNNHead(self.dim_h, dim_out)
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_virtual_nodes > 0:
            for node_type in self.virtual_nodes:
                torch.nn.init.normal_(self.virtual_nodes[node_type])

    def forward(self, batch):
        batch = self.encoder(batch)
        
        batch.x = self.input_drop(batch.x)

        num_nodes_dict = None
        
        for i in range(cfg.gt.layers):
            batch = self.convs[i](batch)

        if self.num_virtual_nodes > 0:
            # Remove the virtual nodes
            for node_type in batch.node_types:
                batch[node_type].x = batch[node_type].x[:num_nodes_dict[node_type], :]
                batch[node_type].num_nodes -= self.num_virtual_nodes


        # Output L2 norm
        if cfg.gt.l2_norm:
            for node_type in batch.node_types:
                batch[node_type].x = F.normalize(batch[node_type].x, p=2, dim=-1) 

        return self.post_gt(batch)
