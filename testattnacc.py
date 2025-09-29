import datetime, os, logging
import torch

import GTBenchmark  # noqa, register custom modules
from GTBenchmark.agg_runs import agg_runs

from GTBenchmark.graphgym.cmd_args import parse_args
from GTBenchmark.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from GTBenchmark.graphgym.loader import create_loader
from GTBenchmark.graphgym.logger import setup_printing
from GTBenchmark.graphgym.optimizer import create_optimizer, \
    create_scheduler, OptimizerConfig
from GTBenchmark.graphgym.model_builder import create_model
from GTBenchmark.graphgym.train import train
from GTBenchmark.graphgym.utils.comp_budget import params_count
from GTBenchmark.graphgym.utils.device import auto_select_device
from GTBenchmark.graphgym.register import train_dict
from torch_geometric import seed_everything

from GTBenchmark.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from GTBenchmark.logger import create_logger
from GTBenchmark.utils.utils import (new_optimizer_config, new_scheduler_config, \
                             custom_set_out_dir, custom_set_run_dir)

# test_equiv_generalattn_vs_mha.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# ---------- 你项目里的 GeneralAttn ----------
# 假设你已把 GeneralAttn 放在可导入路径；否则把类粘贴进本文件。
from GTBenchmark.attention.GeneralAttn import GeneralAttn

# —— 这里写一个极简“Batch”以适配 GeneralAttn —— #
class Batch:
    def __init__(self, x, attn_bias=None):
        self.x = x
        if attn_bias is not None:
            self.attn_bias = attn_bias

# ---------- 配置 ----------
B = 2
H = 8
D = 10                # head_dim
E = H * D             # embed_dim
N_base = 12           # 不含 CLS 的节点数
use_cls = True
Np = N_base + 1 if use_cls else N_base

# ---------- 构造输入 ----------
x = torch.randn(B, Np, E, device=device, dtype=dtype)          # [B, Np, E]
QK_scale = 1.0 / math.sqrt(D)

# 结构偏置 E_bias: [B,H,Np,Np]
# 注意：Graphormer中 CLS 与任何节点的 bias 通常为 0；这里也设为 0 以贴近语义
E_bias = torch.randn(B, H, Np, Np, device=device, dtype=dtype) * 0.1
if use_cls:
    E_bias[:, :, 0, :] = 0.0
    E_bias[:, :, :, 0] = 0.0

# base mask（不含 CLS）：构造一个随机可达矩阵，然后转为 additive（0/-inf）
keep_prob = 0.85
base_keep = torch.rand(B, 1, N_base, N_base, device=device) < keep_prob
# 保证对角线 keep
eye = torch.eye(N_base, device=device, dtype=torch.bool).view(1,1,N_base,N_base)
base_keep = base_keep | eye
# 转加性：keep=0, mask=-inf
base_add = torch.zeros_like(base_keep, dtype=dtype)
base_add = base_add.masked_fill(~base_keep, float("-inf"))
# pad 到含 CLS 的大小：[B,1,Np,Np]，CLS 行列为 0
if use_cls:
    add_mask = F.pad(base_add, (1,0,1,0), value=0.0)
else:
    add_mask = base_add
# 扩到 [B,H,Np,Np]
add_mask_H = add_mask.expand(B, H, Np, Np).contiguous()


ga = GeneralAttn(embed_dim=E, num_heads=H, dropout_p=0.0, backend="math").to(device).eval()

# ---------- 构造 PyTorch MHA，并把权重对齐到 ga ----------
mha = nn.MultiheadAttention(embed_dim=E, num_heads=H, dropout=0.0, batch_first=True).to(device).eval()

# 同步权重： in_proj(q/k/v) 与 out_proj
with torch.no_grad():
    # in_proj_weight: [3E, E], in_proj_bias: [3E]
    mha.in_proj_weight[:E].copy_(ga.q_proj.weight)
    mha.in_proj_bias[:E].copy_(ga.q_proj.bias)

    mha.in_proj_weight[E:2*E].copy_(ga.k_proj.weight)
    mha.in_proj_bias[E:2*E].copy_(ga.k_proj.bias)

    mha.in_proj_weight[2*E:].copy_(ga.v_proj.weight)
    mha.in_proj_bias[2*E:].copy_(ga.v_proj.bias)

    mha.out_proj.weight.copy_(ga.out_proj.weight)
    mha.out_proj.bias.copy_(ga.out_proj.bias)

# ---------- 参考：SDPA 路径 ----------
# 先用 ga 的 q/k/v 线性层得到 Q,K,V，和 ga 完全一致
with torch.no_grad():
    Q = ga.q_proj(x).view(B, Np, H, D).transpose(1, 2).contiguous()
    K = ga.k_proj(x).view(B, Np, H, D).transpose(1, 2).contiguous()
    V = ga.v_proj(x).view(B, Np, H, D).transpose(1, 2).contiguous()

# 融合 mask 与偏置，得到 [B,H,Np,Np] 的 additive
fused_add = (E_bias + add_mask_H).contiguous()

# ---------- 1) GeneralAttn（math） ----------
batch = Batch(x.clone(), attn_bias=E_bias.clone())
ga_out, ga_attn = ga(batch, mask=add_mask_H, return_attn_weights=True)
# ga_attn: [B,H,Np,Np]

# ---------- 2) SDPA ----------
with torch.no_grad():
    sdpa_out = scaled_dot_product_attention(
        Q, K, V,
        attn_mask=fused_add,      # [B,H,Np,Np] additive
        dropout_p=0.0,
        is_causal=False,
        scale=QK_scale
    )  # -> [B,H,Np,D]
    sdpa_out = sdpa_out.transpose(1, 2).contiguous().view(B, Np, E)
    # 注意：SDPA 没有 out_proj，这里只对比 GeneralAttn 在 out_proj 之前的语义；
    # 若想对比到最终 out，需要再乘同一个 out_proj
    sdpa_out_proj = ga.out_proj(sdpa_out)

# ---------- 3) MHA ----------
# MHA 需要 (B*H, Np, Np) 的 3D additive attn_mask
mha_mask = (E_bias + add_mask_H).reshape(B*H, Np, Np).contiguous()
with torch.no_grad():
    mha_out, mha_attn = mha(
        x, x, x,
        need_weights=True,
        average_attn_weights=False,  # 返回每个 head 的注意力
        attn_mask=mha_mask
    )
    # mha_attn: [B*H, Np, Np] -> [B,H,Np,Np]
    mha_attn = mha_attn.view(B, H, Np, Np)

# ---------- 误差对比 ----------
def maxdiff(a, b):
    return (a - b).abs().max().item()

print("== Max |Δ| vs SDPA (out_proj):")
print("GeneralAttn vs SDPA_out_proj:", maxdiff(ga_out, sdpa_out_proj))

print("\n== Max |Δ| on attention weights:")
print("GeneralAttn.attn vs MHA.attn :", maxdiff(ga_attn, mha_attn))

print("\n== Max |Δ| on outputs:")
print("GeneralAttn.out vs MHA.out   :", maxdiff(ga_out, mha_out))
