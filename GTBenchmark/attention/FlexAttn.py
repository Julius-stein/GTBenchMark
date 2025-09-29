
# 入口处，尽量早（在 import torch 前最稳）：
# import os
# os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
# os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN"  # 先只留 ATEN，必有兜底

import torch
# import torch._inductor.config as indcfg
# indcfg.max_autotune = False
# indcfg.max_autotune_gemm_backends = "ATEN"
# print("max_autotune =", indcfg.max_autotune, " backends =", indcfg.max_autotune_gemm_backends)


# import torch
# import torch._inductor.config as indcfg
# indcfg.max_autotune_gemm_backends = "TRITON,ATEN"  # 运行时方式（同效，只要足够早）
import torch
import torch.nn as nn
from torch._dynamo import disable as dynamo_disable
from torch.nn.attention.flex_attention import flex_attention as _flex_attention
from GTBenchmark.graphgym.register import register_layer
from GTBenchmark.graphgym.config import cfg
# print("backends =", indcfg.max_autotune_gemm_backends)  # 应输出 'TRITON,ATEN'


class FlexAttnCore(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, scale: float):
        super().__init__()
        self.H, self.D, self.scale = num_heads, head_dim, scale

    @torch.compile(fullgraph=True, dynamic=False,mode="max-autotune")
    def forward(self, Q, K, V, E_bias_or_none, block_mask):
        B, H, N, D = Q.shape
        if E_bias_or_none is None:
            def score_mod(score, b, h, qi, ki): return score
        else:
            Eflat = E_bias_or_none.reshape(-1)
            HH, NN = self.H, N
            def score_mod(score, b, h, qi, ki):
                lin = (((b * HH) + h) * NN + qi) * NN + ki
                return score + Eflat[lin]

        return _flex_attention(
            Q, K, V,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.scale,
            kernel_options={
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,                
            }  # 避免超过shared Memory大小
        )

@register_layer('FlexAttention')
class FlexAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.0,
                 x_name='x', bias_name='attn_bias', use_flex=True):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.x_name, self.bias_name = x_name, bias_name
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.use_flex = use_flex and (dropout_p == 0.0)
        cfg.share.can_flex = True
        self.in_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.core = FlexAttnCore(num_heads, self.head_dim, self.scale)  # 单独的 Flex 核

    @staticmethod
    def _qkv(x, H, D):
        # x: [B,N,D_model] -> Q,K,V: [B,H,N,D]
        B, N, _ = x.shape
        qkv = x @ torch.empty((x.shape[-1], 3 * H * D), device=x.device, dtype=x.dtype)  # 占位。实际用 self.in_proj
        raise NotImplementedError  # 仅示意

    # @dynamo_disable(recursive=True)  # 整个外层 forward 不进图（含 batch 属性读写、dict 等）
    def forward(self, batch, pad_mask=None):
        x: torch.Tensor = getattr(batch, self.x_name)        # [B,Nm,F]
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # 1) 线性投影 + 形状变换（eager）
        qkv = self.in_proj(x)                              # [B,N,3F]
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        Q = q.view(B, N, H, D).transpose(1, 2).contiguous()  # [B,H,N,D]
        K = k.view(B, N, H, D).transpose(1, 2).contiguous()
        V = v.view(B, N, H, D).transpose(1, 2).contiguous()

        # 2) 准备 E_bias 与 block_mask（eager；不要在编译里构造）
        raw_bias = getattr(batch, self.bias_name, None)      # 允许为 None
        if raw_bias is None:
            E_bias = None
        else:
            if raw_bias.dim() == 3:      # [B*H,N,N]
                E_bias = raw_bias.view(B, H, N, N).contiguous()
            elif raw_bias.dim() == 4:    # [B,H,N,N]
                E_bias = raw_bias.contiguous()
            else:
                raise RuntimeError("attn_bias dim must be 3 or 4")

        block_mask = getattr(batch, "flex_block_mask", None) # 建议在 DataLoader/transform 里预构造

        # 3) Flex 内核（仅此处进编译图）
        out = self.core(Q, K, V, E_bias, block_mask)     # [B,H,N,D]
        out = out.transpose(1, 2).contiguous().view(B, N, H * D)


        out = self.out_proj(out)
        setattr(batch, self.x_name, out)                     # eager 的 Python 副作用
        return batch
