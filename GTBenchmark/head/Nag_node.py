# file: nag_heads.py
import torch
import torch.nn as nn
from GTBenchmark.graphgym.register import register_head

@register_head('NAGReadout')
class NAGReadout(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.final_ln = nn.LayerNorm(dim_in)
        self.attn_fc = nn.Linear(2*dim_in, 1)
        self.proj    = nn.Linear(dim_in, dim_in//2)
        self.classif = nn.Linear(dim_in//2, dim_out)

    def forward(self, batch):
        # x: (N, K+1, d)
        x = batch.x
        x = self.final_ln(x) 
        node = x[:, 0:1, :]          # (N,1,d)
        neigh = x[:, 1:, :]          # (N,K,d)
        tgt_rep = node.repeat(1, neigh.size(1), 1)  # broadcast

        # learn α_hop
        alpha = self.attn_fc(torch.cat([tgt_rep, neigh], dim=-1))   # (N,K,1)
        alpha = torch.softmax(alpha, dim=1)
        neigh_sum = (alpha * neigh).sum(dim=1, keepdim=False)       # (N,d)

        out = node.squeeze(1) + neigh_sum                           # (N,d)
        out = torch.relu(self.proj(out))
        out = self.classif(out)                                     # (N,num_class)

        batch.out = torch.log_softmax(out, dim=-1)
        return batch.out, batch.y
