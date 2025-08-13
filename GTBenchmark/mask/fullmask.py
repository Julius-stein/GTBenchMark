import torch
import torch.nn as nn
import torch.nn.functional as F

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_mask
from GTBenchmark.transform.graph2dense import to_dense_batch


@register_mask('full')
class Fullpair(nn.Module):
    def __init__(self,batch):
        super().__init__()
        self.Dmask = None
        if hasattr(batch,"num_graphs"):
            self.batch_size = batch.num_graphs
            self.max_num_nodes = int((batch.ptr[1:] - batch.ptr[:-1]).max())
            self.flat_size = self.batch_size * self.max_num_nodes
            self.F = cfg.gt.dim_hidden
                   
                

    def forward(self, batch):
        if hasattr(batch, "num_graphs"):
            x = batch.x
            if self.Dmask is None:
                batch.x, self.Dmask, self.Gindex = to_dense_batch(
                    x, batch.batch, batch_size=self.batch_size
                )
            else:
                dense_flat = torch.full((self.flat_size, self.F), 0.0,
                                        dtype=x.dtype, device=x.device)
                dense_flat[self.Gindex] = x
                batch.x = dense_flat.view(self.batch_size, self.max_num_nodes, self.F)

            # Dmask: [B, N]，True=有效节点
            key_pad = (~self.Dmask).unsqueeze(1).unsqueeze(2).float()  # [B,1,1,N]
            attn_mask = key_pad * (-1e9)                               # [B,1,1,N]（additive）
            attn_mask = attn_mask.expand(-1, 1, self.Dmask.size(1), -1)  # -> [B,1,N,N]
        else:
            attn_mask = None

        return batch, attn_mask
    
    def from_dense_batch(self,batch):
        # x: (B,N,d) -- > (N, d)
        if hasattr(batch,"num_graphs"):
            x = batch.x
            if self.Dmask == None:
                raise RuntimeError("from_dense_batch() called before dense batch was built in forward().")
            else:
                batch.x = x[self.Dmask]
        else:
            pass

        return batch

