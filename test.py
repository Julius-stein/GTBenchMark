#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续图 vs 随机图：PyTorch 基础算子访存性能比较（仅输出时间）
节点数 = 169,343
边数   ≈ 116,243 (无向)
"""

import numpy as np
import torch
import time

N = 169_343
E = 116_243  # 无向边数量
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------
# 1. 构造连续 block 图
# --------------------------------------------
block_size = 128
num_blocks = N // block_size
edges = []

for b in range(num_blocks):
    start = b * block_size
    end = min(N, start + block_size)
    pairs = np.random.randint(start, end, size=(E // num_blocks, 2))
    edges.append(pairs)

edge_index = np.concatenate(edges, axis=0)
edge_index = np.unique(edge_index, axis=0)[:E]  # 控制边数
rows, cols = edge_index[:, 0], edge_index[:, 1]
print(f"Synthetic graph: {N} nodes, {len(rows)} directed edges (~{len(rows)//2} undirected)")

# --------------------------------------------
# 2. 数据准备
# --------------------------------------------
x = torch.randn(N, 128, device=device)
src = torch.tensor(rows, dtype=torch.long, device=device)
dst = torch.tensor(cols, dtype=torch.long, device=device)

# --------------------------------------------
# 3. 三种算子 benchmark
# --------------------------------------------
def bench_index_select(x, src):
    torch.cuda.synchronize()
    t0 = time.time()
    _ = torch.index_select(x, 0, src)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000

def bench_scatter_add(x, src, dst):
    out = torch.zeros_like(x)
    torch.cuda.synchronize()
    t0 = time.time()
    out.index_add_(0, dst, x[src])
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000

def bench_elementwise_mul(x):
    torch.cuda.synchronize()
    t0 = time.time()
    _ = x * x
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000

# 顺序访问（连续图）
t_index = bench_index_select(x, src)
t_scatter = bench_scatter_add(x, src, dst)
t_elem = bench_elementwise_mul(x)

# 随机访问（随机图）
perm = torch.randperm(N, device=device)
src_shuf = perm[src]
dst_shuf = perm[dst]
t_index_rand = bench_index_select(x, src_shuf)
t_scatter_rand = bench_scatter_add(x, src_shuf, dst_shuf)
t_elem_rand = bench_elementwise_mul(x)  # 不依赖索引

# --------------------------------------------
# 4. 打印结果
# --------------------------------------------
print("\n========= 实验结果 =========")
print(f"节点数: {N:,} | 有向边数: {len(rows):,}")
print("\n算子平均时间 (ms)")
print(f"{'算子':<20}{'连续图':>15}{'随机图':>15}{'顺序/随机':>15}")
print("-" * 65)
print(f"{'index_select':<20}{t_index:>15.3f}{t_index_rand:>15.3f}{t_index_rand/t_index:>15.2f}")
print(f"{'scatter_add':<20}{t_scatter:>15.3f}{t_scatter_rand:>15.3f}{t_scatter_rand/t_scatter:>15.2f}")
print(f"{'elementwise_mul':<20}{t_elem:>15.3f}{t_elem_rand:>15.3f}{'—':>15}")
print("-" * 65)
