import torch
import torch.nn as nn
import torch.nn.functional as F

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.register import register_layer
from GTBenchmark.graphgym.config import cfg

@register_layer('GraphTransformerLayer')
class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, dim_h):
        super().__init__()
        #assert dim_h == cfg.gt.dim_hidden
        self.out_channels = dim_h
        self.num_heads = cfg.gt.attn_heads
        self.dropout = cfg.gt.dropout
        self.residual = cfg.gt.residual
        self.layer_norm = cfg.gt.layer_norm        
        self.batch_norm = cfg.gt.batch_norm
        self.act = register.act_dict[cfg.gt.act]

        
        self.ffn_dim = cfg.gt.ffn_dim if cfg.gt.ffn_dim!=0 else dim_h*2
        
        
        self.attention = register.layer_dict[cfg.gt.attn_type](dim_h, self.num_heads,cfg.gt.attn_dropout)
        
        # self.preattnLayernorm = nn.LayerNorm(dim_h)
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(dim_h)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(dim_h)
        
        # FFN
        self.FFN_layer1 = nn.Linear(dim_h, self.ffn_dim)
        self.FFN_layer2 = nn.Linear(self.ffn_dim, dim_h)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(dim_h)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(dim_h)
        
    def forward(self, batch,mask):
        
        h_in1 = batch.x
        
        # batch.x = self.preattnLayernorm(batch.x) 

        batch = self.attention(batch,mask)
        # torch.save(batch,"./tmp1.pt")
        #h = attn_out.view(-1, self.out_channels)
        
        h = batch.x 

        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = self.act(h)
        h = self.FFN_layer2(h)

        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        batch.x = h
        return batch
        
    def __repr__(self):
        return '{}(out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.out_channels, self.num_heads, self.residual)
