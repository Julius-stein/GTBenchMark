import torch
import torch.nn as nn
import torch.nn.functional as F

# import GTBenchmark.graphgym.models.head  # noqa, register module
import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_network
from GTBenchmark.network.utils import FeatureEncoder
from GTBenchmark.graphgym.models.gnn import GNNPreMP, GeneralLayer



@register_network('GTModel')
class GTModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        # ---------------- 基础 ----------------
        self.dim_h      = cfg.gt.dim_hidden
        #@!
        # self.input_drop = nn.Dropout(cfg.gt.input_dropout)
        # self.activation = register.act_dict[cfg.gt.act]
        self.layer_norm = cfg.gt.layer_norm
        self.l2_norm    = getattr(cfg.gt, "l2_norm", False)  # 可能不存在就给缺省
        GNNHead         = register.head_dict[cfg.gt.head]

        self.encoder = FeatureEncoder()
        if cfg.gnn.layers_pre_mp > 0:
            if len(cfg.gt.node_encoder_list)==0:
                if cfg.gt.layer_type == "NodeFormerConv":
                    self.pre_mp = GeneralLayer('linear', dim_in = dim_in, dim_out = cfg.gt.dim_hidden, num_layers = 1, has_act = True, has_bias = True, cfg = cfg)
                else:
                    self.pre_mp = GNNPreMP(
                    cfg.share.dim_in, self.dim_h, cfg.gnn.layers_pre_mp)
            
            else:
                self.pre_mp = GNNPreMP(
                    self.dim_h, self.dim_h, cfg.gnn.layers_pre_mp)
                
        else:
            self.pre_mp = None
        

        # ---------------- 选择 GT 层类型 ----------------
        if cfg.gnn.mode.lower() == "gps":
            local_model =  cfg.gnn.layer_type
            global_model = cfg.gt.layer_type
            GTLayer = register.layer_dict["GPSLayer"]
            self.gt_layers = nn.ModuleList([GTLayer(self.dim_h,local_model,global_model) for _ in range(cfg.gt.layers)])
        elif cfg.gt.layer_type in ["Transformer","GraphTransformerLayer","TransformerLayer"]:
            gt_layer_key = "GraphTransformerLayer"
            GTLayer = register.layer_dict[gt_layer_key]
            self.gt_layers = nn.ModuleList([GTLayer(dim_h=self.dim_h) for _ in range(cfg.gt.layers)])
        else:
            gt_layer_key = cfg.gt.layer_type
            
            GTLayer = register.layer_dict[gt_layer_key]
            self.gt_layers = nn.ModuleList([GTLayer(dim_h=self.dim_h) for _ in range(cfg.gt.layers)])

        # ---------------- GNN 栈--------------------------------------
        self.gnn_mode = cfg.gnn.mode.lower()  # off | post | cascade | parallel | gps
        if self.gnn_mode in ("post", "cascade", "parallel"):
            GNNLayer = register.layer_dict[cfg.gnn.layertype]
            gnn_layers = cfg.gnn.layers if hasattr(cfg.gnn, "layers") else cfg.gt.layers
            self.gnn_layers = nn.ModuleList([GNNLayer(self.dim_h) for _ in range(gnn_layers)])
            self.has_gnn = True
        else:
            self.gnn_layers = nn.ModuleList()
            self.has_gnn = False

        if self.gnn_mode == "parallel":
            self.parallel_fuse = nn.Sequential(
                nn.Linear(self.dim_h * 2, cfg.gnn.dim_inner),
                register.act_dict[cfg.gnn.act],
                nn.Dropout(cfg.gnn.dropout),
                nn.Linear(cfg.gnn.dim_inner, self.dim_h),
            )

        # ---------------- 绑定执行路径（forward 零判断） ----------------
        if cfg.gt.layer_type == "NodeFormerConv":
            self.run = self._run_attn_only
        elif self.gnn_mode == "off":
            self.run = self._run_gt_only
        elif self.gnn_mode == "gps":
            self.run = self._run_gps
        elif self.gnn_mode == "post":
            self.run = self._run_post
        elif self.gnn_mode == "cascade":
            self.run = self._run_cascade
        elif self.gnn_mode == "parallel":
            self.run = self._run_parallel
        else:
            raise ValueError(f"Unknown gnn.mode: {self.gnn_mode}")

        self.post_head = GNNHead(self.dim_h, dim_out)
        # self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    # ---------------- 内部小工具 ----------------
    def _gt_stack(self, dense, mask):
        for layer in self.gt_layers:
            dense = layer(dense, mask)
        return dense

    def _gnn_stack(self, graph):
        for layer in self.gnn_layers:
            graph = layer(graph)
        return graph

    # ---------------- 执行路径 ----------------
    def _run_attn_only(self,batch,maskGen):
        return self._gt_stack(batch,None)
    def _run_gt_only(self, batch, maskGen):
        # off：GT 栈内部已决定是否融合（hybrid 由层内部完成）
        dense, mask = maskGen(batch)
        dense = self._gt_stack(dense, mask)
        return maskGen.from_dense_batch(dense)
    
    def _run_gps(self, batch, maskGen):
        # gps：GT 栈内部已决定是否融合（hybrid 由层内部完成）
        for layer in self.gt_layers:
            batch = layer(batch, maskGen)
        return batch

    def _run_post(self, batch, maskGen):
        # 先 GT 全栈，再 GNN 全栈
        dense, mask = maskGen(batch)
        dense = self._gt_stack(dense, mask)
        graph = maskGen.from_dense_batch(dense)
        graph = self._gnn_stack(graph)
        return graph

    def _run_cascade(self, batch, maskGen):
        # 交替：GT(i) -> GNN(i) -> ...
        dense, mask = maskGen(batch)
        i = j = 0
        while i < len(self.gt_layers) or j < len(self.gnn_layers):
            if i < len(self.gt_layers):
                dense = self.gt_layers[i](dense, mask)
                i += 1
            if j < len(self.gnn_layers):
                graph = maskGen.from_dense_batch(dense)
                graph = self.gnn_layers[j](graph)
                j += 1
                dense, mask = maskGen.to_dense_batch(graph)
        return maskGen.from_dense_batch(dense)

    def _run_parallel(self, batch, maskGen):
        # GT 全栈 + GNN 全栈，然后 concat→MLP
        dense, mask = maskGen(batch)
        gt_out = self._gt_stack(dense, mask)

        graph = maskGen.from_dense_batch(dense)
        gnn_out_graph = self._gnn_stack(graph)
        gnn_out_dense, _ = maskGen.to_dense_batch(gnn_out_graph)

        fused = torch.cat([gt_out.x, gnn_out_dense.x], dim=-1)
        gt_out.x = self.parallel_fuse(fused)
        return maskGen.from_dense_batch(gt_out)

    # ---------------- 标准 forward（零判断） ----------------
    def forward(self, batch):
        batch = self.encoder(batch)
        if self.pre_mp is not None:
            batch = self.pre_mp(batch)
        maskGen = register.mask_dict[cfg.mask.name](batch)
        graph_batch = self.run(batch, maskGen)

        if self.l2_norm and hasattr(graph_batch, 'x'):
            graph_batch.x = F.normalize(graph_batch.x, p=2, dim=-1)

        return self.post_head(graph_batch)
    
