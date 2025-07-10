
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from GTBenchmark.graphgym.register import register_layer
from models.FFN import FeedForwardNetwork
from GTBenchmark.graphgym.config import cfg
import GTBenchmark.graphgym.register as register
from models.MHA import MultiHeadAttention
@register_layer('NAGphormer_Layer')
class NAGphormerLayer(nn.Module):
    def __init__(self, dim_h):
        super(NAGphormerLayer, self).__init__()

        self.num_heads = cfg.gt.attn_heads
        self.dropout = cfg.gt.dropout
        self.attn_dropout = cfg.gt.attn_dropout  
        self.ffn_dim = cfg.gt.ffn_dim
        
        
        self.attention_norm = nn.LayerNorm(dim_h)
        self.attention = MultiHeadAttention(dim_h,self.attn_dropout,self.num_heads,1)
        self.attention_post_dropout = nn.Dropout(self.dropout)
        
        
        self.ffn_norm = nn.LayerNorm(dim_h)
        self.ffn = FeedForwardNetwork(dim_h, self.ffn_dim, self.dropout)
        self.ffn_dropout = nn.Dropout(self.dropout)

    def forward(self, batch):
        h_in1 = batch.x

        batch.x = self.attention_norm(batch.x)

        
        batch.x = self.attention(batch.x, batch.x, batch.x)
        batch.x = self.attention_post_dropout(batch.x)
        batch.x = batch.x + h_in1

        h_in1 = batch.x
        y = self.ffn_norm(batch.x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        batch.x = h_in1 + y
        return batch














