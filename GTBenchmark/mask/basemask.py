import torch
import torch.nn as nn

from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_mask
from GTBenchmark.transform.graph2dense import to_dense_batch, get_graph_sizes, to_dense_adj
import warnings
warnings.filterwarnings(
    "ignore",
    message="It is not recommended to directly access the internal storage format `data`",
    category=UserWarning,
)

# @register_mask("full")
class Basemask(nn.Module):
    """
    PyG -> attn_mask 的通用构建器。提供两种通路，由flatattn确定

    - 扁平(BatchedAttention)：对 N_total×N_total，只保留“同图”的注意力（Document型）。
    - 稠密(General)：对 [B, Nmax] 有效位做 Padding 掩码（无效位全 -inf）。
    主要承担1.数据转化 2.无效位mask
    """

    def __init__(self,):
        super().__init__()
        self.num_heads = int(cfg.gt.n_heads)
        self.F = int(cfg.gt.dim_hidden)

        # 路径选择
        self.flatattn_type = (cfg.gt.attn_type == "BatchedAttention")
        self.batch_size = 0
        self.node_counts = None
        self.max_num_nodes = 0
        self.flat_size = 0

        self.transadj = cfg.mask.transadj


    def forward(self,batch):
        # Batch 维信息：统一基于 batch.batch（兼容 GraphToken 改写）
        if hasattr(batch, "num_graphs") and batch.num_graphs > 1:
            self.batch_size = int(batch.num_graphs)
            # print(batch.batch.max().item())
            # 统计每张图节点数（依赖 batch.batch）
            self.node_counts = torch.bincount(batch.batch, minlength=self.batch_size)
            # 无需额外padding，做batch的最小形状
            self.max_num_nodes = int(self.node_counts.max().item())
            self.flat_size = int(self.node_counts.sum().item())
        
        # 绑定 forward / 回填方法
        if self.flatattn_type:
            self.from_dense_batch = self._noop   # 扁平路径无需回填
            return self._forward_flat(batch)    # type: ignore[assignment]
        elif self.batch_size==0: #full-batch方法
            self.from_dense_batch = self.nofakebatch
            return self.fakebatch(batch)
        else:
            self.from_dense_batch = self._from_dense_batch
            return self._forward_general(batch)

    # -------------------------
    # General：to_dense_batch + Padding
    # -------------------------
    def _forward_general(self, batch):
        if self.batch_size  == 0:
            return batch, None

        x = batch.x
        device = x.device

        # to_dense_batch 仅在第一次或 size 变化时做
        if not hasattr(batch, "dense_cache"):
            dense_x, Dmask, Gindex = to_dense_batch(
                x, batch.batch, max_num_nodes=self.max_num_nodes,batch_size=self.batch_size
            )  # dense_x:[B,Nmax,F], Dmask:[B,Nmax](True=有效), Gindex:[N_total] -> 还原用
            if self.transadj:
                adj,Aindex = to_dense_adj(batch.edge_index,batch.batch,edge_attr=batch.edge_attr,max_num_nodes=self.max_num_nodes,batch_size=self.batch_size)
                setattr(batch,"Aindex",Aindex)
                batch.adj = adj
            setattr(batch, "dense_cache", (Dmask, Gindex))
            batch.x = dense_x
        else:
            Dmask, Gindex = getattr(batch,"dense_cache") 
            # 复用掩码，仅把 x 写入既有稠密缓冲
            dense_flat = torch.zeros((self.batch_size * self.max_num_nodes, self.F),
                                     dtype=x.dtype, device=device)
            dense_flat[Gindex] = x
            batch.x = dense_flat.view(self.batch_size, self.max_num_nodes, self.F)
            # if self.transadj:
            #     Aindex = getattr(batch,"Aindex")
            #     adjflat= torch.zeros(self.batch_size *self.max_num_nodes*self.max_num_nodes*batch.edge_attr.size(-1))
            #     adjflat[Aindex] = batch.edge_attr
            #     batch.adj = adjflat.view(self.batch_size, self.max_num_nodes, self.max_num_nodes, batch.edge_attr.size(-1))



        # 生成 Padding 掩码

        # key_pad: [B,1,1,N]，True位置 -inf
        key_pad = (~Dmask).unsqueeze(1).unsqueeze(2).float()
        attn_mask = key_pad * (-1e9)
        attn_mask = attn_mask.expand(self.batch_size, self.num_heads,
                                        self.max_num_nodes, self.max_num_nodes)
        return batch, attn_mask

    # -------------------------
    # Batched：扁平 Document 掩码（同图可见）
    # -------------------------
    def _forward_flat(self, batch):
        N_total = batch.x.size(0)  # 扁平总节点数（已包含 GraphToken 的情况）
        # additive：同图可见（Document）
        same_graph = (batch.batch[:, None] == batch.batch[None, :])
        key_pad = (~same_graph).unsqueeze(0).unsqueeze(0).float()  # [1,1,N,N]
        attn_mask = key_pad * (-1e9)
        attn_mask = attn_mask.expand(1, self.num_heads, N_total, N_total)
        return batch, attn_mask

    # -------------------------
    # 稠密 -> 扁平 回填
    # -------------------------
    def _from_dense_batch(self, batch):
        """
        把 [B,Nmax,F] 的 x 回填为 [N_total,F]。
        需要先在 _forward_general() 里构建过 Dmask/Gindex。
        """
        if not hasattr(batch, "num_graphs"):
            return batch
        if not hasattr(batch, "dense_cache"):
            raise RuntimeError("from_dense_batch() 在构建稠密前被调用。")
        Dmask, _ = getattr(batch,"dense_cache") 
        # 直接基于 Dmask 提取有效位
        x = batch.x  # [B,Nmax,F]
        batch.x = x[Dmask]  # [N_total, F]
        return batch

    def _noop(self, batch):
        return batch
    def fakebatch(self, batch):
        batch.x = batch.x.unsqueeze(0)
        return batch,None
    
    def nofakebatch(self, batch):
        batch.x.squeeze_(0)
        return batch
