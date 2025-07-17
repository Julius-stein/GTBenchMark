
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from GTBenchmark.graphgym.register import register_layer
from GTBenchmark.graphgym.register import register_network
from GTBenchmark.graphgym.config import cfg
import GTBenchmark.graphgym.register as register
from GTBenchmark.encoder.hop_to_token_encoder import HopToTokenEncoder
@register_network('NAGphormer')
class NAGphormer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.seq_len = cfg.dataset.hop + 1  
        pecfg = cfg.posenc_LapPE
        self.pe_dim = pecfg.dim_pe  
        
        self.hidden_dim = cfg.gt.dim_hidden
        self.ffn_dim = cfg.gt.ffn_dim
        self.num_heads = cfg.gt.attn_heads
        self.n_layers = cfg.gt.layers
        self.dropout_rate = cfg.gt.dropout
        self.attention_dropout_rate = cfg.gt.attn_dropout

        self.encoder = HopToTokenEncoder(cfg.dataset.hop)  
        GNNHead = register.head_dict[cfg.gt.head]

        self.Linear1 = nn.Linear(dim_in, self.hidden_dim)  
        

        self.convs = nn.ModuleList()
        for i in range(cfg.gt.layers):
            conv = register.layer_dict[cfg.gt.layer_type](dim_h=self.hidden_dim)
            self.convs.append(conv)
            

        self.final_ln = nn.LayerNorm(self.hidden_dim)  
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))  
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)  
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)  
        self.post_gt = GNNHead(self.hidden_dim//2, dim_out)  

    def forward(self, batch):
        
        
        # hopTotoken
        batch = self.encoder(batch)
        
       

        batch.x = self.Linear1(batch.x)

        
        for enc_layer in self.convs:
            batch = enc_layer(batch)
        
     
        ### ReadOut ###
        x = batch.x 
        output = self.final_ln(x)
        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len-1, 1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        output = (node_tensor + neighbor_tensor).squeeze(1)
        output = self.out_proj(output)
        output = torch.relu(output)
        batch.x = output

        return self.post_gt(batch)
        
