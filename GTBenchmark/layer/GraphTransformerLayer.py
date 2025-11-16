import torch
import torch.nn as nn
import torch.nn.functional as F
import GTBenchmark.graphgym.register as register 
from GTBenchmark.graphgym.register import register_layer 
from GTBenchmark.layer.FFN import FFN_block
from GTBenchmark.graphgym.config import cfg 

@register_layer('GraphTransformerLayer')
class GraphTransformerLayer(nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        cfg_gt = cfg.gt  # 缓存，少一次全局查找
        self.out_channels = dim_h
        self.num_heads = cfg_gt.n_heads
        self.residual = cfg_gt.residual
        self.prepend_norm = cfg_gt.prepend_norm

        # —— 缓存超参数 —— #
        self.p_res = float(cfg_gt.dropout)

        self.use_ln = bool(cfg_gt.layer_norm)
        self.use_bn = bool(cfg_gt.batch_norm)
        self.ffn_dim = (cfg_gt.ffn_dim if cfg_gt.ffn_dim != 0 else dim_h * 2)

        # —— Attention —— #
        # 你的 GeneralAttn / MHA 注册保持不变
        self.attention = register.layer_dict[cfg_gt.attn_type](
            dim_h, self.num_heads, cfg_gt.attn_dropout
        )

        # —— Norm 组合成模块，避免前向分支 —— #
        def build_norm():
            layers = []
            if self.use_ln: layers.append(nn.LayerNorm(dim_h, eps=1e-5))
            if self.use_bn: layers.append(nn.BatchNorm1d(dim_h))
            return nn.Sequential(*layers) if layers else nn.Identity()

        self.norm1 = build_norm()
        self.norm2 = build_norm()

        # —— MLP —— #
        self.ffn = FFN_block(dim_h)

        # —— 残差 Dropout —— #
        self.drop_res = nn.Dropout(self.p_res) if self.p_res > 0 else nn.Identity()

        # —— 绑定前向以移除运行时分支 —— #
        self._forward = self._forward_preln if self.prepend_norm else self._forward_postln

    # ====== Pre-LN 变体 ====== #
    def _forward_preln(self, batch, mask):
        x = batch.x

        # 1) Attn block (Pre-Norm)
        x_norm = self.norm1(x)
        batch.x = x_norm               
        batch = self.attention(batch, mask)
        h = self.drop_res(batch.x)
        x = x + h                      # 非原地，利于 torch.compile

        # 2) FFN block (Pre-Norm)
        h = self.norm2(x)
        h = self.ffn(x)
        h = self.drop_res(h)
        x = x + h

        batch.x = x
        return batch

    # ====== Post-LN 变体 ====== #
    def _forward_postln(self, batch, mask):
        x = batch.x

        # 1) Attn block
        batch.x = x
        batch = self.attention(batch, mask)
        h = self.drop_res(batch.x)
        x = self.norm1(x + h)

        # 2) FFN block
        h = self.ffn(x)
        h = self.drop_res(h)
        x = self.norm2(x + h)

        batch.x = x
        return batch

    def forward(self, batch, mask):
        return self._forward(batch, mask)
    


