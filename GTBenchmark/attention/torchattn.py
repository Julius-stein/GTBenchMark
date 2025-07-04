import torch
import torch.nn as nn
import torch.nn.functional as F
import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.register import register_layer
from GTBenchmark.graphgym.config import cfg
from torch_geometric.utils import to_dense_adj, to_dense_batch

class TorchAttn(nn.Module):
    is_sparse = None
    def __init__(self, dim_h, num_heads, x_name='x', b_name='attn_bias'):
        super(TorchAttn, self).__init__()
        if self.is_sparse is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'is_sparse' class"
                             f"variable before calling the constructor.")
        self.x_name = x_name
        self.b_name = b_name
        self.attention = nn.MultiheadAttention(dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)

    def forward(self, batch):
        h, mask = to_dense_batch(getattr(batch, self.x_name), batch.batch)
        # multi-head attention out
        if self.is_sparse:
            attn_mask = ~(to_dense_adj(batch.edge_index, batch.batch).bool())
            #对角线置1否则softmax会出nan
            attn_mask[:,torch.arange(attn_mask.shape[-1]),torch.arange(attn_mask.shape[-1])] = False
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            new_attn_mask = torch.zeros_like(attn_mask, dtype=h.dtype)
            new_attn_mask.masked_fill(attn_mask, float('-inf'))
            attn_mask = new_attn_mask
        else:
            attn_mask = None
        
        if attn_mask is None:
            attn_mask = getattr(batch, self.b_name, None)
        else:
            attn_mask = attn_mask + getattr(batch, self.b_name, None)
        attn_out, _ = self.attention(h, h, h, need_weights=False, key_padding_mask=~mask, attn_mask=attn_mask)
        h = attn_out[mask]
        setattr(batch, self.x_name, h)
        return batch

@register_layer('TorchFullAttention')
class TorchFullAttn(TorchAttn):
    is_sparse = False
    
@register_layer('TorchSparseAttention')
class TorchSparseAttn(TorchAttn):
    is_sparse = True