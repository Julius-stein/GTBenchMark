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
        if cfg.dataset.task == "graph":
            self.batch_size = batch.num_graphs
            self.max_num_nodes = int((batch.ptr[1:] - batch.ptr[:-1]).max())
            self.flat_size = self.batch_size * self.max_num_nodes
            self.F = cfg.share.dim_in
                   
                

    def forward(self, batch):
        # x: (N, d) -- >(B,N,d)
        if cfg.dataset.task == "graph":
            x = batch.x
            if self.Dmask == None:
                batch.x,self.Dmask,self.Gindex = to_dense_batch(x,batch.batch,max_num_nodes=self.max_num_nodes,batch_size=self.batch_size)
                
            else:
                # 已有 x, idx, 并已知 batch_size, max_num_nodes, fill_value, device,省去计算过程
                # 1) 先构造一个全 fill_value 的扁平张量
                dense_flat = torch.full((self.flat_size, self.F), 0.0,
                                        dtype=x.dtype, device=x.device)
                dense_flat[self.Gindex] = x
                batch.x = dense_flat.view(self.batch_size, self.max_num_nodes, self.F)

            attn_mask = self.Dmask.unsqueeze(1).unsqueeze(2)       # [B, 1, 1, N]
            attn_mask = attn_mask.expand(-1,1,self.max_num_nodes, -1)  
        else:
            attn_mask = None

        return batch, attn_mask
    
    def from_dense_batch(self,batch):
        # x: (B,N,d) -- > (N, d)
        if cfg.dataset.task == "graph":
            x = batch.x
            if self.Dmask == None:
                raise RuntimeError("from_dense_batch() called before dense batch was built in forward().")
            else:
                batch.x = x[self.Dmask]
        else:
            pass

        return batch

