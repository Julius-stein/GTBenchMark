# check_on_cora.py
import sys
from typing import List, Tuple

import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from GTBenchmark.transform.graph_partitionV2 import partition_graph_to_datalist, compute_size_stats

DEVICE = torch.device("cpu")


def _collect_global_edge_set(edge_index: torch.Tensor) -> set:
    # Use ordered tuple (min(u,v), max(u,v)) to ignore direction for equality
    row, col = edge_index
    return set((int(min(u, v)), int(max(u, v))) for u, v in zip(row.tolist(), col.tolist()) if u != v)


def _edge_set_of_subgraphs(datalist: List[torch_geometric.data.Data], global_orig_n: int) -> set:
    edges = set()
    for d in datalist:
        if d.edge_index.numel() == 0:
            continue
        # map local node ids to global using orig_n_id
        g = d.orig_n_id
        r, c = d.edge_index
        for u, v in zip(r.tolist(), c.tolist()):
            gu, gv = int(g[u]), int(g[v])
            if gu == gv:  # skip self
                continue
            a, b = (min(gu, gv), max(gu, gv))
            edges.add((a, b))
    return edges


def _check_masks(global_data, sg):
    for name in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(global_data, name) and getattr(global_data, name) is not None:
            gmask = getattr(global_data, name)
            if gmask.dim() == 1 and gmask.size(0) == global_data.num_nodes and hasattr(sg, name):
                lmask = getattr(sg, name)
                assert torch.equal(lmask.to(torch.bool), gmask[sg.orig_n_id].to(torch.bool)), f"{name} mismatch."


def _check_features_and_labels(global_data, sg):
    if hasattr(global_data, "x") and global_data.x is not None and sg.x is not None:
        assert torch.equal(sg.x, global_data.x[sg.orig_n_id]), "x mismatch with global"
    if hasattr(global_data, "y") and global_data.y is not None and sg.y is not None and global_data.y.size(0) == global_data.num_nodes:
        assert torch.equal(sg.y, global_data.y[sg.orig_n_id]), "y mismatch with global"


def _check_dataloader_batch(datalist: List[torch_geometric.data.Data], batch_size: int = 2):
    loader = DataLoader(datalist, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    # PyG Batch must have 'batch' vector
    assert hasattr(batch, "batch") and batch.batch is not None, "PyG Batch missing 'batch'"
    # batch.ptr marks graph boundaries; len == num_graphs_in_batch + 1
    assert hasattr(batch, "ptr") and batch.ptr.numel() == (min(batch_size, len(datalist)) + 1), "ptr wrong size"


def run_checks(n_parts: int = 4):
    # Load Cora
    dataset = Planetoid(root="/mnt/data2/duxin/.datasets", name="Cora")
    data = dataset[0].to(DEVICE)  # single big graph

    combos = [
        ("random", "edge-cut"),
        ("random", "vertex-cut"),
        ("metis",  "edge-cut"),
        ("metis",  "vertex-cut"),
    ]

    print(f"Graph: N={data.num_nodes}, E={data.edge_index.size(1)}")
    global_edges = _collect_global_edge_set(data.edge_index)

    for algo, cut in combos:
        print(f"\n=== Check {algo} / {cut} / k={n_parts} ===")
        datalist, meta = partition_graph_to_datalist(
            data, n_parts=n_parts, cut_type=cut, algo=algo, directed=False, random_seed=42
        )

        # 基本统计
        mean_sz, var_sz = compute_size_stats(datalist)
        print(f"Subgraph sizes: mean={mean_sz:.2f}, var={var_sz:.2f}")

        # === 打印每个子图细节 ===
        print(f"\n[Subgraph Details]")
        total_nodes, total_edges = 0, 0
        all_ids = []
        for i, sg in enumerate(datalist):
            n, e = sg.num_nodes, sg.edge_index.size(1)
            total_nodes += n
            total_edges += e
            all_ids.append(sg.orig_n_id.tolist())
            print(f"  Part {i:02d}: nodes={n:5d}, edges={e:6d}, "
                  f"orig_id_range=[{int(sg.orig_n_id.min())}-{int(sg.orig_n_id.max())}]")
        print(f"Total: nodes(sum)={total_nodes}, edges(sum)={total_edges}")

        # === 检查全局节点覆盖与唯一性 ===
        flat_ids = torch.cat([torch.tensor(ids) for ids in all_ids])
        unique_ids = torch.unique(flat_ids)
        print(f"Unique node coverage: {unique_ids.numel()}/{data.num_nodes}")
        dup_count = flat_ids.numel() - unique_ids.numel()
        print(f"Duplicated node entries across subgraphs: {dup_count}")

        # === 检查 mask / attr 一致性汇总 ===
        num_mask_mismatch = 0
        num_feat_mismatch = 0
        for i, sg in enumerate(datalist):
            try:
                _check_features_and_labels(data, sg)
            except AssertionError as e:
                num_feat_mismatch += 1
                print(f"  ⚠️ Feature mismatch in part {i}: {e}")
            try:
                _check_masks(data, sg)
            except AssertionError as e:
                num_mask_mismatch += 1
                print(f"  ⚠️ Mask mismatch in part {i}: {e}")
        if num_feat_mismatch == 0 and num_mask_mismatch == 0:
            print("Feature and mask consistency: ✅ All matched.")
        else:
            print(f"Feature mismatches: {num_feat_mismatch}, Mask mismatches: {num_mask_mismatch}")

        # 1) 划分正确性
        if cut == "edge-cut":
            # 每个节点应只出现一次
            seen = torch.zeros(data.num_nodes, dtype=torch.long)
            for sg in datalist:
                seen[sg.orig_n_id] += 1
            assert torch.all(seen == 1), "edge-cut violates unique-node property"

            # 不应包含跨子图边：联合边集应是“原图内部边”的一个子集
            sub_edges = _edge_set_of_subgraphs(datalist, data.num_nodes)
            assert sub_edges.issubset(global_edges), "edge-cut edges not subset of global edges"

        elif cut == "vertex-cut":
            # 边应被完整覆盖且唯一：联合边集 == 全图边集
            sub_edges = _edge_set_of_subgraphs(datalist, data.num_nodes)
            assert sub_edges == global_edges, "vertex-cut must cover all edges exactly once"

        # 2) 映射一致性 & 属性一致性
        for sg in datalist:
            # orig_n_id 不应有重复
            assert sg.orig_n_id.unique().numel() == sg.orig_n_id.numel(), "orig_n_id duplicated within a subgraph"
            _check_features_and_labels(data, sg)
            _check_masks(data, sg)

        # 3) DataLoader 批能力
        _check_dataloader_batch(datalist, batch_size=2)

        print("All checks passed.")

if __name__ == "__main__":
    run_checks(n_parts=8)
