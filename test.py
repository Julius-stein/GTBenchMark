#!/usr/bin/env python
# compare_batch_v3.py
# -------------------------------------------------------------
# 比较两份 PyG Data/Batch，对共享字段做逐元素一致性检查。
# —— 支持 float/int/bool Tensor 以及普通标量 / 字符串。
# —— 键集合可不完全一致，脚本仍会比对交集。
# 用法：
#   python compare_batch_v3.py a.pt b.pt --atol 1e-6 --rtol 1e-6 --ignore ptr
# -------------------------------------------------------------
import argparse, sys, torch
from torch_geometric.data import Data, Batch, HeteroData

def to_dict(obj):
    d = obj.to_dict() if hasattr(obj, "to_dict") else {k: getattr(obj, k) for k in obj.keys}
    return {k: v for k, v in d.items() if v is not None}

def load(path):
    obj = torch.load(path, map_location="cpu",weights_only=False)
    if isinstance(obj, (Data, Batch, HeteroData)) or hasattr(obj, "keys"):
        return to_dict(obj)
    sys.exit(f"{path} 不是 PyG Data/Batch 对象")

def tensor_diff(a: torch.Tensor, b: torch.Tensor,
                atol: float, rtol: float, key: str):
    if a.dtype == torch.bool:
        mism = (a ^ b).sum().item()
        return mism == 0, {"bool_mismatch": mism}
    else:
        diff = (a - b).abs().float()
        abs_err, rel_err = diff.max().item(), diff.div(b.abs().clamp_min(1e-12)).max().item()
        ok = (abs_err <= atol) or (rel_err <= rtol)
        return ok, {"abs": abs_err, "rel": rel_err}

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("file_a"); pa.add_argument("file_b")
    pa.add_argument("--atol", type=float, default=1e-8)
    pa.add_argument("--rtol", type=float, default=1e-5)
    pa.add_argument("--ignore", nargs="*", default=["num_node_features"], help="fields to skip")
    args = pa.parse_args()

    dict_a, dict_b = load(args.file_a), load(args.file_b)
    for k in args.ignore:
        dict_a.pop(k, None); dict_b.pop(k, None)

    keys_a, keys_b = set(dict_a), set(dict_b)
    shared = sorted(keys_a & keys_b)
    only_a, only_b = sorted(keys_a - keys_b), sorted(keys_b - keys_a)

    if only_a: print("Fields only in A:", ", ".join(only_a))
    if only_b: print("Fields only in B:", ", ".join(only_b))

    if not shared:
        sys.exit("⚠ 没有共同字段可比较")

    all_ok = True
    worst = {"abs": 0.0, "rel": 0.0, "bool_mismatch": 0, "key": None}

    for k in shared:
        va, vb = dict_a[k], dict_b[k]

        # 类型不同
        if type(va) is not type(vb):
            print(f"✗ '{k}': type {type(va).__name__} vs {type(vb).__name__}")
            all_ok = False
            continue

        if isinstance(va, torch.Tensor):
            if va.shape != vb.shape:
                print(f"✗ '{k}': shape {va.shape} vs {vb.shape}")
                all_ok = False
                continue
            ok, stats = tensor_diff(va, vb, args.atol, args.rtol, k)
            if not ok:
                if "bool_mismatch" in stats:
                    print(f"✗ '{k}': {stats['bool_mismatch']} bool elements differ")
                else:
                    print(f"✗ '{k}': abs={stats['abs']:.3e} rel={stats['rel']:.3e}")
                all_ok = False
            # 记录最坏差值
            for s in ("abs", "rel", "bool_mismatch"):
                if s in stats and stats[s] > worst[s]:
                    worst.update({s: stats[s], "key": k})
        else:  # 标量 / str / list …
            if va != vb:
                print(f"✗ '{k}': {va!r} != {vb!r}")
                all_ok = False

    if all_ok:
        print("✔ 所有共享字段完全一致！")
        sys.exit(0)
    else:
        if worst["bool_mismatch"]:
            print(f"最大 bool 差异: {worst['bool_mismatch']} 处 (key='{worst['key']}')")
        else:
            print(f"最大 abs={worst['abs']:.3e}, rel={worst['rel']:.3e} (key='{worst['key']}')")
        sys.exit(1)

if __name__ == "__main__":
    main()
