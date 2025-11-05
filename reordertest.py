#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time, warnings
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

import metis
import torch_scatter
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 环境检查
# --------------------------------------------------
if "METIS_DLL" not in os.environ:
    print("⚠️ 请先设置 METIS_DLL，例如:")
    print("  export METIS_DLL=/usr/lib/x86_64-linux-gnu/libmetis.so")
    raise SystemExit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
import math

# =====================================================
# ExphormerAttention — 独立版
# =====================================================
class ExphormerAttention(nn.Module):
    """
    Standalone Exphormer Attention
    """
    def __init__(self, dim_h, num_heads, dim_edge=None,
                 attn_drop=0.0, use_virt_nodes=False, return_attn_weights=False):
        super().__init__()
        assert dim_h % num_heads == 0, "hidden dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.out_dim = dim_h // num_heads
        self.dim_h = dim_h
        self.use_virt_nodes = use_virt_nodes
        self.attn_drop = attn_drop
        self.return_attn_weights = return_attn_weights

        # 如果没传dim_edge，默认等于dim_h
        if dim_edge is None:
            dim_edge = dim_h

        # Linear projections
        self.Q = nn.Linear(dim_h, self.out_dim * num_heads, bias=False)
        self.K = nn.Linear(dim_h, self.out_dim * num_heads, bias=False)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=False)
        self.V = nn.Linear(dim_h, self.out_dim * num_heads, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.K.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.V.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.E.weight, gain=1 / math.sqrt(2))

    def propagate_attention(self, h_Q, h_K, h_V, E, edge_index):
        src, dst = edge_index[0], edge_index[1]

        src_K = h_K[src]     # [E, H, D]
        dst_Q = h_Q[dst]     # [E, H, D]
        score = (src_K * dst_Q) / np.sqrt(self.out_dim)
        score = score * E[src] if E is not None else score
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # [E, H, 1]

        msg = h_V[src] * score
        N = h_V.size(0)
        H = h_V.size(1)

        wV = torch.zeros_like(h_V)           # [N, H, D]
        scatter(msg, dst, dim=0, out=wV, reduce='add')

        Z = torch.zeros((N, H, 1), device=h_V.device, dtype=h_V.dtype)
        scatter(score, dst, dim=0, out=Z, reduce='add')

        h_out = wV / (Z + 1e-6)
        return h_out

    def forward(self, h, edge_index, edge_attr=None,
                virt_h=None, virt_edge_index=None, virt_edge_attr=None):
        # 处理虚拟节点
        if self.use_virt_nodes and virt_h is not None:
            h = torch.cat([h, virt_h], dim=0)
            if virt_edge_index is not None and virt_edge_attr is not None:
                edge_index = torch.cat([edge_index, virt_edge_index], dim=1)
                edge_attr = torch.cat([edge_attr, virt_edge_attr], dim=0)

        # Linear projections
        Q_h = self.Q(h).view(-1, self.num_heads, self.out_dim)
        K_h = self.K(h).view(-1, self.num_heads, self.out_dim)
        V_h = self.V(h).view(-1, self.num_heads, self.out_dim)
        E_h = self.E(edge_attr).view(-1, self.num_heads, self.out_dim) if edge_attr is not None else None

        h_out = self.propagate_attention(Q_h, K_h, V_h, E_h, edge_index)
        h_out = h_out.view(-1, self.num_heads * self.out_dim)

        if self.use_virt_nodes and virt_h is not None:
            num_real = virt_h.size(0)
            virt_out = h_out[num_real:]
            h_out = h_out[:num_real]
            return h_out, virt_out
        return h_out


# =====================================================
# ExphormerFullLayer — Attention + FFN
# =====================================================
class ExphormerFullLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, dim_edge=None,
                 layer_norm=True, batch_norm=True,
                 activation='relu',
                 residual=True, use_virt_nodes=False):
        super().__init__()

        self.attn = ExphormerAttention(
            dim_h=in_dim, num_heads=num_heads,
            dim_edge=dim_edge, use_virt_nodes=use_virt_nodes
        )

        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.dropout = dropout

        if layer_norm:
            self.norm1 = nn.LayerNorm(out_dim)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(out_dim)

        # FeedForward
        self.ffn1 = nn.Linear(out_dim, out_dim * 2)
        self.act = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.ffn2 = nn.Linear(out_dim * 2, out_dim)

        if layer_norm:
            self.norm2 = nn.LayerNorm(out_dim)
        if batch_norm:
            self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h_in = x
        h_attn = self.attn(x, edge_index, edge_attr)
        h = F.dropout(h_attn, self.dropout, training=self.training)
        if self.residual:
            h = h + h_in
        if self.layer_norm:
            h = self.norm1(h)
        if self.batch_norm:
            h = self.bn1(h)

        h_in2 = h
        h = self.ffn1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.ffn2(h)
        if self.residual:
            h = h + h_in2
        if self.layer_norm:
            h = self.norm2(h)
        if self.batch_norm:
            h = self.bn2(h)
        return h


# --------------------------------------------------
# SlashBurn 实现
# --------------------------------------------------
def slashburn_order(G, k=500):
    G = G.copy()
    order = []
    while len(G) > 0:
        if len(G) <= k:
            order.extend(list(G.nodes()))
            break
        degs = sorted(G.degree, key=lambda x: x[1], reverse=True)
        hubs = [n for n, _ in degs[:k]]
        order.extend(hubs)
        G.remove_nodes_from(hubs)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        for comp in comps[1:]:
            order.extend(list(comp))
            G.remove_nodes_from(comp)
        if len(G) == 0:
            break
    return np.array(order, dtype=int)

# --------------------------------------------------
# 图重排指标
# --------------------------------------------------
def evaluate_permutation(A, perm, parts=None, block_size=128):
    A = A.tocsr()
    Ap = A[perm][:, perm]
    rows, cols = Ap.nonzero()
    if parts is None:
        block_ids = np.arange(A.shape[0]) // block_size
    else:
        block_ids = np.array(parts)[perm]
    bandwidth = int(np.abs(rows - cols).max()) if len(rows) else 0
    intra = int(np.sum(block_ids[rows] == block_ids[cols]))
    inter = int(len(rows) - intra)
    total_edges = max(len(rows), 1)
    block_density = intra / total_edges
    inter_ratio = inter / total_edges

    num_blocks = math.ceil(A.shape[0] / block_size)
    block_rows = rows // block_size
    block_cols = cols // block_size
    active_blocks = set(zip(block_rows, block_cols))
    total_blocks = num_blocks ** 2
    empty_block_ratio = 1 - len(active_blocks) / total_blocks

    return bandwidth, block_density, inter_ratio, empty_block_ratio

# --------------------------------------------------
# GPU Kernel Benchmarks
# --------------------------------------------------
def bench_index_select(x, src_index, repeat=10):
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    min_time = 1e9
    for _ in range(repeat):
        torch.cuda.synchronize()
        start.record()
        _ = torch.index_select(x, 0, src_index)
        end.record()
        torch.cuda.synchronize()
        min_time = min(min_time, start.elapsed_time(end))
    return min_time

def bench_scatter_add(x, src, dst, repeat=10):
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    min_time = 1e9
    for _ in range(repeat):
        torch.cuda.synchronize()
        start.record()
        out = torch.zeros_like(x)
        torch_scatter.scatter_add(x[src], dst, dim=0, out=out)
        end.record()
        torch.cuda.synchronize()
        min_time = min(min_time, start.elapsed_time(end))
    return min_time

def bench_elementwise_mul(x, repeat=10):
    y = torch.randn_like(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    min_time = 1e9
    for _ in range(repeat):
        torch.cuda.synchronize()
        start.record()
        z = x * y
        end.record()
        torch.cuda.synchronize()
        min_time = min(min_time, start.elapsed_time(end))
    return min_time

# --------------------------------------------------
# 访存连续性指标
# --------------------------------------------------
def memory_access_locality(src_index, window=128):
    """
    更贴近 GPU 真实访存模式的连续性指标：
    看原始访问序列中，连续元素之间的索引差距是否小。
    """
    idx = src_index.detach().cpu().numpy()
    deltas = np.abs(np.diff(idx))

    # 连续访问定义：差值 <= window
    run_flags = (deltas <= 1)
    avg_run_len = 1
    run_lengths = []
    cur_len = 1
    for cont in run_flags:
        if cont:
            cur_len += 1
        else:
            run_lengths.append(cur_len)
            cur_len = 1
    run_lengths.append(cur_len)

    avg_run = np.mean(run_lengths)
    prop_close = np.mean(deltas < window)
    locality_score = avg_run / len(idx)
    return {
        "AvgRunLen": float(avg_run),
        "PropClose": float(prop_close),
        "LocalityScore": float(locality_score),
    }

def warp_locality_score(src_index, warp_size=32, stride_threshold=128):
    idx = src_index.detach().cpu().numpy()
    n = len(idx)
    n_warps = n // warp_size
    scores = []
    for i in range(n_warps):
        warp_slice = idx[i*warp_size : (i+1)*warp_size]
        span = warp_slice.max() - warp_slice.min()
        # 局部访问越紧密，span越小
        scores.append(span <= stride_threshold)
    return np.mean(scores)  # 表示访问“块内”的比例



# --------------------------------------------------
# 主程序
# --------------------------------------------------
def main(save_prefix="reorder_bench_memlocal"):
    print("Loading ogbn-arxiv ...")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="/mnt/data2/duxin/.datasets/")
    data = dataset[0]
    A = to_scipy_sparse_matrix(data.edge_index).tocsr()
    N = A.shape[0]
    rows, cols = A.nonzero()
    print(f"Loaded ogbn-arxiv: {N} nodes, {A.nnz//2} undirected edges")

    results, perms = {}, {}
    block_size = 128

    # === 原始顺序 ===
    perm_orig = np.arange(N)
    results["Original"] = evaluate_permutation(A, perm_orig)
    perms["Original"] = perm_orig

    # === RCM ===
    print("\nRunning RCM ...")
    perm_rcm = reverse_cuthill_mckee(A)
    results["RCM"] = evaluate_permutation(A, perm_rcm)
    perms["RCM"] = perm_rcm

    # === METIS ===
    print("\nRunning METIS partition ...")
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    num_parts = int(math.ceil(N / block_size))
    edgecuts, parts = metis.part_graph(G, nparts=num_parts)
    parts_arr = np.array(parts)
    perm_metis = np.argsort(parts_arr)
    results["METIS"] = evaluate_permutation(A, perm_metis, parts=parts_arr)
    perms["METIS"] = perm_metis

    # === SlashBurn ===
    print("\nRunning SlashBurn ...")
    perm_slash_core = slashburn_order(G, k=300)
    missing = np.setdiff1d(np.arange(N), perm_slash_core)
    perm_slash = np.concatenate([perm_slash_core, missing])
    results["SlashBurn"] = evaluate_permutation(A, perm_slash)
    perms["SlashBurn"] = perm_slash

    # === GPU测试 ===
    print("\nBenchmarking on GPU ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(N, 128, device=device)
    E = len(rows)
    perf_results = {}
    memlocal_results = {}

    for name, perm in perms.items():
        rank = np.empty(N, dtype=np.int64)
        rank[perm] = np.arange(N)
        x_perm = x[torch.tensor(perm.copy(), dtype=torch.long, device=device)]

        # ✅ 同时重排数据与索引
        # x_perm = x[perm]
        src = torch.tensor(rank[rows], dtype=torch.long, device=device)
        dst = torch.tensor(rank[cols], dtype=torch.long, device=device)

        # Benchmark
        t_index = bench_index_select(x_perm, src)
        t_scat = bench_scatter_add(x_perm, src, dst)
        t_elem = bench_elementwise_mul(x_perm)

        # Locality metric
        # memlocal_results[name] = memory_access_locality(src.cpu())
        memlocal_results[name] = {
                "WarpLocality": warp_locality_score(src.cpu()),
            }

        perf_results[name] = (t_index, t_scat, t_elem)

        print(f"{name:10s}: index={t_index:.3f}ms  scatter={t_scat:.3f}ms  elem={t_elem:.3f}ms")

    # === 输出结果表 ===
    print("\nAlgorithm     BlockDensity  EmptyBlockRatio  AvgRunLen  Propclose  index(ms)  scatter(ms)  elem(ms)")
    for name in results:
        r = results[name]
        m = memlocal_results[name]
        t1, t2, t3 = perf_results[name]
        print(f"{name:12s} {r[1]:.4f}     {r[3]:.4f}     {m["WarpLocality"]:7.1f}     {t1:8.3f}   {t2:8.3f}   {t3:8.3f}")
    # --------------------------------------------------
    # GCN 执行性能对比
    # --------------------------------------------------
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    @torch.no_grad()
    def bench_gcn_forward(x, edge_index, repeat=10):
        model = GCNConv(x.size(-1), 128).to(device)
        torch.cuda.synchronize()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        min_time = 1e9
        for _ in range(repeat):
            torch.cuda.synchronize()
            start.record()
            out = model(x, edge_index)
            out = F.relu(out)
            end.record()
            torch.cuda.synchronize()
            min_time = min(min_time, start.elapsed_time(end))
        return min_time


    print("\n=== GCN Forward Comparison ===")
    edge_index_orig = torch.tensor(data.edge_index, dtype=torch.long, device=device)

    # 构造随机顺序 edge_index：仅打乱节点索引，不改变连边结构
    perm_rand = torch.randperm(N, device=device)
    rank_rand = torch.empty_like(perm_rand)
    rank_rand[perm_rand] = torch.arange(N, device=device)
    edge_index_rand = rank_rand[edge_index_orig]

    t_orig = bench_gcn_forward(x, edge_index_orig)
    t_rand = bench_gcn_forward(x, edge_index_rand)

    print(f"Original edge_index: {t_orig:.3f} ms/forward")
    print(f"Random edge_index:   {t_rand:.3f} ms/forward")

    speedup = (t_rand - t_orig) / t_rand * 100
    print(f"Speed difference: {speedup:+.2f}% (positive = original faster)")
    # --------------------------------------------------
    # ExphormerAttention 执行性能对比
    # --------------------------------------------------
    from torch_geometric.data import Data

    @torch.no_grad()
    def bench_exphormer_forward(edge_index, edge_attr, dim_h=128, num_heads=8, repeat=10):
        """
        Benchmark the forward time and peak memory of standalone ExphormerAttention.
        Only uses attention core, no FFN or normalization.
        """
        model = ExphormerAttention(dim_h=dim_h, num_heads=num_heads, dim_edge=dim_h).to(device)
        x = torch.randn(N, dim_h, device=device)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        min_time = 1e9

        for _ in range(repeat):
            torch.cuda.synchronize()
            start.record()
            _ = model(x, edge_index, edge_attr)
            end.record()
            torch.cuda.synchronize()
            min_time = min(min_time, start.elapsed_time(end))

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        return min_time, peak_mem



    print("\n=== ExphormerAttention Forward Comparison ===")
    edge_attr = torch.randn(A.nnz, 128, device=device)

    # 原始顺序
    edge_index_orig = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    t_exph_orig, mem_orig = bench_exphormer_forward(edge_index_orig, edge_attr)

    # 随机顺序
    perm_rand = torch.randperm(N)
    rank_rand = np.empty(N, dtype=np.int64)
    rank_rand[perm_rand.cpu().numpy()] = np.arange(N)
    edge_index_rand = torch.tensor(rank_rand[edge_index_orig.cpu()], dtype=torch.long, device=device)
    t_exph_rand, mem_rand = bench_exphormer_forward(edge_index_rand, edge_attr)

    print(f"Original edge_index: {t_exph_orig:.3f} ms, {mem_orig:.1f} MB")
    print(f"Random   edge_index: {t_exph_rand:.3f} ms, {mem_rand:.1f} MB")
    print(f"Speed diff: {(t_exph_rand - t_exph_orig)/t_exph_rand*100:+.2f}% (positive = reordered faster)")
    print(f"Memory diff: {(mem_rand - mem_orig):+.1f} MB")


if __name__ == "__main__":
    main()
