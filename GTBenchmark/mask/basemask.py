import torch
import torch.nn as nn

from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_mask
from GTBenchmark.transform.graph2dense import to_dense_batch, get_graph_sizes

# @register_mask("full")
class Basemask(nn.Module):
    """
    PyG -> (attn_mask 或 Flex BlockMask) 的通用构建器。提供两种通路，由flatattn确定

    - 扁平(BatchedAttention)：对 N_total×N_total，只保留“同图”的注意力（Document型）。
    - 稠密(General)：对 [B, Nmax] 有效位做 Padding 掩码（无效位全 -inf）。
    主要承担1.数据转化 2.无效位mask

    支持FlexAttention:
      - 模型定义时已确定能否使用flexAttn：
        * Batched：用 create_block_mask 构造“同图” block 稀疏掩码（B=1, H=heads, Q=KV=N_total）
        * General：用 create_block_mask 构造“有效位内”掩码（B=batch, H=heads, Q=KV=Nmax）每次调用forward都会执行一次把PyGBathedData转化为一般数据。
      - block_mask 会放在 batch.flex_block_mask；attn_mask 置为 None
    """

    def __init__(self,):
        super().__init__()
        # ---- 检测 Flex ----
        self.Trans = False
        if cfg.dataset.task == "graph":
            self.Trans = True
        self.use_flex = cfg.share.can_flex

        self.num_heads = int(cfg.gt.attn_heads)
        self.F = int(cfg.gt.dim_hidden)

        # 路径选择
        self.flatattn_type = (cfg.gt.attn_type == "BatchedAttention")
        self.batch_size = 0
        self.node_counts = None
        self.max_num_nodes = 0
        self.flat_size = 0

        # 稠密路径缓存
        self.Dmask = None          # [B, Nmax] bool
        self.Gindex = None         # 扁平回填索引

    def forward(self,batch):
        # Batch 维信息：统一基于 batch.batch（兼容 GraphToken 改写）
        if hasattr(batch, "num_graphs") and batch.num_graphs > 1:
            self.batch_size = int(batch.num_graphs)
            # print(batch.batch.max().item())
            # 统计每张图节点数（依赖 batch.batch）
            self.node_counts = torch.bincount(batch.batch, minlength=self.batch_size)
            if cfg.share.targetsize == -1:
            # 无需额外padding，做batch的最小形状
                self.max_num_nodes = int(self.node_counts.max().item())
            else:
                self.max_num_nodes = cfg.share.targetsize
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
        """
        输出：
          - attn_mask: [B, H, Nmax, Nmax] additive（若不用 Flex）
          - batch.flex_block_mask: BlockMask（若用 Flex）
        """
        if self.batch_size  == 0:
            return batch, None

        x = batch.x
        device = x.device

        # to_dense_batch 仅在第一次或 size 变化时做
        need_build_dense = (
            self.Dmask is None
            or self.Dmask.shape[0] != self.batch_size
            or self.Dmask.shape[1] != self.max_num_nodes
        )

        if need_build_dense:
            dense_x, Dmask, Gindex = to_dense_batch(
                x, batch.batch, max_num_nodes=self.max_num_nodes,batch_size=self.batch_size
            )  # dense_x:[B,Nmax,F], Dmask:[B,Nmax](True=有效), Gindex:[N_total] -> 还原用
            batch.x = dense_x
            self.Dmask = Dmask
            self.Gindex = Gindex
        else:
            # 复用掩码，仅把 x 写入既有稠密缓冲
            dense_flat = torch.zeros((self.batch_size * self.max_num_nodes, self.F),
                                     dtype=x.dtype, device=device)
            dense_flat[self.Gindex] = x
            batch.x = dense_flat.view(self.batch_size, self.max_num_nodes, self.F)

        # 生成 Padding 掩码（non-Flex 回退）
        if not self.use_flex:
            # key_pad: [B,1,1,N]，True位置 -inf
            key_pad = (~self.Dmask).unsqueeze(1).unsqueeze(2).float()
            attn_mask = key_pad * (-1e9)
            attn_mask = attn_mask.expand(self.batch_size, self.num_heads,
                                         self.max_num_nodes, self.max_num_nodes)
            # 清理 Flex 产物
            batch.flex_block_mask = None
            return batch, attn_mask

        # Flex：按有效位构造 BlockMask
        lengths = self.Dmask.sum(dim=1).to(torch.int32)  # [B]

        def pad_mask_mod(b, h, q_idx, kv_idx):
            lb = lengths[b]
            # 与原 additive padding mask 等价：只限制 KV 侧
            return kv_idx < lb

        from torch.nn.attention.flex_attention import create_block_mask
        block_mask = create_block_mask(
            pad_mask_mod,
            B=self.batch_size,
            H=self.num_heads,
            Q_LEN=self.max_num_nodes,
            KV_LEN=self.max_num_nodes,
            device=device,
            BLOCK_SIZE = 64,#Block_size>64且为2^*
        )
        batch.flex_block_mask = block_mask
        return batch, None

    # -------------------------
    # Batched：扁平 Document 掩码（同图可见）
    # -------------------------
    def _forward_flat(self, batch):
        """
        输出：
          - attn_mask: [1, H, N, N] additive（若不用 Flex）
          - batch.flex_block_mask: BlockMask（若用 Flex；B=1, Q=KV=N_total）
        """

        device = batch.x.device
        N_total = batch.x.size(0)  # 扁平总节点数（已包含 GraphToken 的情况）

        if not self.use_flex:
            # additive：同图可见（Document）
            same_graph = (batch.batch[:, None] == batch.batch[None, :])
            key_pad = (~same_graph).unsqueeze(0).unsqueeze(0).float()  # [1,1,N,N]
            attn_mask = key_pad * (-1e9)
            attn_mask = attn_mask.expand(1, self.num_heads, N_total, N_total)
            batch.flex_block_mask = None
            return batch, attn_mask

        # Flex：构造 “同图可见” 的 mask_mod
        # graph_id: [N_total]，每个节点所属图 id
        graph_id = batch.batch
        # 注意：这里完全张量索引，兼容 vmap
        def doc_mask_mod(b, h, q_idx, kv_idx):
            # 扁平场景 B=1，因此忽略 b，直接根据全局索引判断是否同图
            return graph_id[q_idx] == graph_id[kv_idx]

        from torch.nn.attention.flex_attention import create_block_mask
        block_mask = create_block_mask(
            doc_mask_mod,
            B=1,
            H=self.num_heads,
            Q_LEN=N_total,
            KV_LEN=N_total,
            device=device,
            BLOCK_SIZE=getattr(cfg.gt, "flex_block_size", 64)
        )
        batch.flex_block_mask = block_mask
        return batch, None

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
        if self.Dmask is None or self.Gindex is None:
            raise RuntimeError("from_dense_batch() 在构建稠密前被调用。")

        # 直接基于 Dmask 提取有效位
        x = batch.x  # [B,Nmax,F]
        batch.x = x[self.Dmask]  # [N_total, F]
        return batch

    def _noop(self, batch):
        return batch
    def fakebatch(self, batch):
        batch.x = batch.x.unsqueeze(0)
        return batch,None
    
    def nofakebatch(self, batch):
        batch.x.squeeze_(0)
        return batch
