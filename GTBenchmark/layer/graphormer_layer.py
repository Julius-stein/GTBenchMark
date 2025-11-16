import torch
import torch.nn as nn
import torch.nn.functional as F

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.register import register_layer
from GTBenchmark.graphgym.config import cfg
# from models.MHA import MultiHeadAttention
# from models.FFN import FeedForwardNetwork
# @register_layer('GraphormerLayer')
# class GraphormerLayer(torch.nn.Module):
#     def __init__(self, dim_h):
        
#         super().__init__()
#         self.num_heads = cfg.gt.n_heads
#         self.dropout = cfg.gt.dropout
#         self.attn_dropout_rate = cfg.gt.attn_dropout
#         self.ffn_dim = cfg.gt.ffn_dim
        
        
        
#         #self.attention = MultiHeadAttention(dim_h,self.attn_dropout_rate,self.num_heads,155)
#         self.attention = register.layer_dict[cfg.gt.attn_type](dim_h, self.num_heads)
#         self.input_norm = torch.nn.LayerNorm(dim_h)
#         self.attn_dropout = torch.nn.Dropout(self.dropout)
#         self.ffn_norm = nn.LayerNorm(dim_h)
#         self.ffn = FeedForwardNetwork(dim_h, self.ffn_dim, self.dropout)
#         self.ffn_dropout = nn.Dropout(self.dropout)
        

#     def forward(self, batch):
#         h_in1 = batch.x
#         batch.x = self.input_norm(batch.x)
#         batch = self.attention(batch)
#         batch.x = self.attn_dropout(batch.x) + h_in1
        
#         h_in1 = batch.x
#         y = self.ffn_norm(batch.x)
#         y = self.ffn(y)
#         y = self.ffn_dropout(y) 
#         batch.x = h_in1 + y
#         return batch
