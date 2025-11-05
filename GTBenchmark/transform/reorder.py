#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Reordering Module — framework interface version (no metric return)
"""

import os, math
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from GTBenchmark.graphgym.config import cfg
import metis # 先export METIS_DLL=/usr/lib/x86_64-linux-gnu/libmetis.so


# ---------------------------------------------------------
# 核心算法
# ---------------------------------------------------------
def _reorder_rcm(edge_index: torch.Tensor, num_nodes: int):
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    adj = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(num_nodes, num_nodes))
    perm = reverse_cuthill_mckee(adj.tocsr(), symmetric_mode=True)
    return np.array(perm, dtype=np.int64)

def _reorder_metis(edge_index: torch.Tensor, num_nodes: int, block_size: int):
    nparts = max(1, math.ceil(num_nodes / block_size))
    nbrs = [set() for _ in range(num_nodes)]
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for u, v in zip(src, dst):
        if u != v:
            nbrs[u].add(int(v))
            nbrs[v].add(int(u))
    adj = [list(n) for n in nbrs]
    _, parts = metis.part_graph(adj, nparts)
    parts = np.array(parts, dtype=np.int64)
    return np.argsort(parts, kind='stable').astype(np.int64)

def _reorder_slashburn(edge_index: torch.Tensor, num_nodes: int, k: int):
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    nbrs = [set() for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if u != v:
            nbrs[u].add(int(v)); nbrs[v].add(int(u))
    deg = np.array([len(n) for n in nbrs])
    removed = np.zeros(num_nodes, dtype=bool)
    perm = np.empty(num_nodes, dtype=np.int64)
    s, e = 0, num_nodes - 1
    while np.count_nonzero(~removed) > k:
        idx = np.argpartition(deg * (~removed), -k)[-k:]
        idx = idx[np.argsort(deg[idx])[::-1]]
        for h in idx:
            if removed[h]: continue
            removed[h] = True; perm[e] = h; e -= 1
            for nb in nbrs[h]:
                if not removed[nb]:
                    deg[nb] -= 1; nbrs[nb].discard(h)
    for i in np.where(~removed)[0]:
        perm[s] = i; s += 1
    return perm

def _reorder_random(num_nodes: int):
    return np.random.permutation(num_nodes).astype(np.int64)

def _reorder_degree(edge_index: torch.Tensor, num_nodes: int, descending: bool = True):
    src, dst = edge_index[0], edge_index[1]
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, src.numpy(), 1)
    np.add.at(deg, dst.numpy(), 1)
    order = np.argsort(deg)
    if descending:
        order = order[::-1]
    return order.astype(np.int64)

def _reorder_bfs(edge_index: torch.Tensor, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for u, v in zip(src, dst):
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    visited = np.zeros(num_nodes, dtype=bool)
    perm = []
    from collections import deque
    for start in range(num_nodes):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        while queue:
            u = queue.popleft()
            perm.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
    return np.array(perm, dtype=np.int64)

def _reorder_gorder(edge_index: torch.Tensor, num_nodes: int, window: int = 32):
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for u, v in zip(src, dst):
        if u != v:
            adj[u].append(v)
            adj[v].append(u)

    ordered = []
    remaining = set(range(num_nodes))
    # 初始选度最大的节点
    deg = np.array([len(adj[i]) for i in range(num_nodes)])
    start = int(np.argmax(deg))
    ordered.append(start)
    remaining.remove(start)

    while remaining:
        scores = np.zeros(num_nodes)
        recent = ordered[-window:] if len(ordered) > window else ordered
        recent_set = set()
        for r in recent:
            recent_set.update(adj[r])
        for v in remaining:
            inter = len(set(adj[v]) & recent_set)
            scores[v] = inter
        nxt = int(max(remaining, key=lambda x: (scores[x], deg[x])))
        ordered.append(nxt)
        remaining.remove(nxt)
    return np.array(ordered, dtype=np.int64)




# ---------------------------------------------------------
# 评估 + 热力图绘制
# ---------------------------------------------------------
def _save_heatmap(edge_index: torch.Tensor, num_nodes: int, block_size: int, save_path: str):
    """
    绘制 (src, dst) 边的散点热力图；
    并计算并注释：
      - 空块率 (%)
      - 所有块的边数方差（包括空块）
    """
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    nblocks = math.ceil(num_nodes / block_size)

    # ---------- 计算块统计 ----------
    bi, bj = src // block_size, dst // block_size
    blk = np.zeros((nblocks, nblocks), dtype=np.int64)
    np.add.at(blk, (bi, bj), 1)

    empty_blocks = int((blk == 0).sum())
    total_blocks = nblocks * nblocks
    empty_ratio = empty_blocks / total_blocks
    var_blocks = float(np.var(blk))   # 所有块，包括空块

    print(f"[{os.path.basename(save_path)}] "
          f"空块率={empty_ratio*100:.2f}%, "
          f"块方差={var_blocks:.4f}")

    # ---------- 绘制边散点热图 ----------
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(src, dst, gridsize=200, cmap='YlGnBu', bins='log')
    plt.colorbar(hb, ax=ax)
    ax.set_xlabel("source node index")
    ax.set_ylabel("destination node index")
    ax.set_title(os.path.basename(save_path))

    # ---------- 在图上标注 ----------
    text = f"空块率: {empty_ratio*100:.2f}%\n块方差: {var_blocks:.4f}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=11, color='white', va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)





def reorder_pyg_dataset(data: Data, methods: list[str], attr_name: str = "edge_index", save_suffix: str = None):
    """
    通用图重排接口，可多次调用。
    - 可指定 attr_name (默认 edge_index)，例如 'edge_index' 或 'expander_edges'
    - 支持方法区分大小写（Metis, Random, RCM 等）
    - 输入边结构既可为 [2, E] 也可为 [E, 2]
    - 输出不会覆盖原属性，生成新字段 <attr_name>_<method>[_suffix]
    """
    assert hasattr(data, attr_name), f"❌ data has no attribute '{attr_name}'"
    edge_data = getattr(data, attr_name)
    n = data.num_nodes

    # --- 兼容输入形状 [E,2] ---
    if edge_data.shape[0] == 2:
        edge_index = edge_data
    elif edge_data.shape[1] == 2:
        edge_index = edge_data.t()
    else:
        raise ValueError(f"Unsupported shape {edge_data.shape}, must be [2,E] or [E,2].")

    for method in methods:
        if method == 'Metis':
            bs = cfg.reorder_Metis.block_size
            perm = _reorder_metis(edge_index, n, bs)
        elif method == 'RCM':
            bs = cfg.reorder_RCM.block_size
            perm = _reorder_rcm(edge_index, n)
        elif method == 'Random':
            bs = cfg.reorder_Random.block_size
            perm = _reorder_random(n)
        elif method == 'Slashburn':
            k = cfg.reorder_Slashburn.k
            bs = cfg.reorder_Slashburn.block_size
            perm = _reorder_slashburn(edge_index, n, k)
        elif method == 'Degree':
            bs = cfg.reorder_Degree.block_size
            perm = _reorder_degree(edge_index, n)
        elif method == 'BFS':
            bs = cfg.reorder_BFS.block_size
            perm = _reorder_bfs(edge_index, n)
        elif method == 'GOrder':
            bs = cfg.reorder_GOrder.block_size
            perm = _reorder_gorder(edge_index, n, window=cfg.reorder_GOrder.window)
        else:
            print(f"⚠️ Unknown method: {method}, skipped.")
            continue

        # ---- 应用节点重排 ----
        old2new = np.empty_like(perm)
        old2new[perm] = np.arange(n)
        idx_map = torch.as_tensor(old2new, dtype=torch.long)
        src_new = idx_map[edge_index[0]]
        dst_new = idx_map[edge_index[1]]
        edge_index_new = torch.stack([src_new, dst_new], dim=0)

        # ---- 输出形状保持与输入一致 ----
        if edge_data.shape[0] == 2:
            out_tensor = edge_index_new
        else:
            out_tensor = edge_index_new.t()

        # ---- 命名与保存 ----
        attr_out = f"{attr_name}_{method}" if save_suffix is None else f"{attr_name}_{method}_{save_suffix}"
        setattr(data, attr_out, out_tensor)
        print(f"✅ Added {attr_out} ({out_tensor.shape})")

        # ---- 可选可视化 ----
        _save_heatmap(edge_index_new, n, bs, f"heatmap_{attr_name}_{method}.png")

    return data


