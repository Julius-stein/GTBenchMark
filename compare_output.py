import torch, json
from typing import Dict, Any, Iterable

# utils/act_diff_norm.py
import re, json, torch

def _sort_key(name: str):
    # 拆分 cname:field
    cname = name.split(":")[0]

    if cname.startswith("ENCODER"):
        return (-1, 0, 0)   # 放最前面
    if cname.startswith("HEAD"):
        # HEAD.LN 在 FC 前
        return (9999, 0 if "LN" in cname else 1, 0)

    m = re.match(r"L(\d+)\.(\w+)", cname)
    if m:
        lid = int(m.group(1))
        sub = m.group(2)
        order = {"prenorm":0, "attn":1, "ffn1":2, "ffn2":3, "out":4}
        return (lid, order.get(sub, 9), 0)

    # fallback:放最后
    return (9998, 9, 0)

def _enumerate_tensors(obj, prefix=""):
    out = {}
    if torch.is_tensor(obj):
        out[prefix] = obj.float().contiguous()
    elif isinstance(obj, (int, float, bool)):
        out[prefix] = torch.tensor(obj, dtype=torch.float32)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_enumerate_tensors(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(_enumerate_tensors(v, key))
    return out

# —— 把不同命名归一化到统一规范 —— #
# 规范示例：ENCODER、L0.out、L0.prenorm、L0.attn、L0.ffn1、L0.ffn2、HEAD.LN、HEAD.FC
def _canon(name: str) -> str:
    n = re.sub(r'^(module\.)?', '', name)
    n = re.sub(r'^(?:model\.)?', '', n)

    # Head
    if re.fullmatch(r'(?:post_mp\.ln|post_gt\.ln)', n): return "HEAD.LN"
    if re.fullmatch(r'(?:post_mp\.layers\.0|post_gt\.fc)', n): return "HEAD.FC"

    # Encoder
    if re.fullmatch(r'encoder', n): return "ENCODER"

    # Layer block output
    m = re.fullmatch(r'(?:layers|convs)\.(\d+)', n)
    if m: return f"L{m.group(1)}.out"

    # prenorm
    m = re.fullmatch(r'(?:layers|convs)\.(\d+)\.(?:input_norm|norm1)', n)
    if m: return f"L{m.group(1)}.prenorm"

    # attention
    m = re.fullmatch(r'(?:layers|convs)\.(\d+)\.attention', n)
    if m: return f"L{m.group(1)}.attn"

    # ffn1 / ffn2
    m = re.fullmatch(r'(?:layers|convs)\.(\d+)\.(?:mlp\.1|fc1)', n)
    if m: return f"L{m.group(1)}.ffn1"
    m = re.fullmatch(r'(?:layers|convs)\.(\d+)\.(?:mlp\.4|fc2)', n)
    if m: return f"L{m.group(1)}.ffn2"

    return n  # 其他保持原样（便于排查遗漏）
    

def diff_runs_normalized(
    pt_a: str, pt_b: str,
    name_include=(), name_exclude=(),
    field_include=('x','batch','ptr','edge_index','real_nodes'),  # 常用字段白名单
    field_exclude=(),
    rtol=1e-5, atol=1e-7, top_k=80
):
    A = torch.load(pt_a, map_location="cpu")["acts"]
    B = torch.load(pt_b, map_location="cpu")["acts"]

    # (层名 -> 规范名) + 展平到 (规范名:字段路径 -> 张量)
    def expand(acts):
        flat = {}
        for raw_name, val in acts.items():
            cname = _canon(raw_name)
            # 名称过滤
            if name_include and not any(s in cname for s in name_include): 
                continue
            if name_exclude and any(s in cname for s in name_exclude): 
                continue
            # 字段展平
            if torch.is_tensor(val):
                kv = _enumerate_tensors(val, "x")
            elif isinstance(val,tuple):
                    kv = {}
                    kv.update(_enumerate_tensors(val[0], "x"))
                    kv.update(_enumerate_tensors(val[1], "attn_weight"))
            else:
                kv = _enumerate_tensors(val, "")
            # 字段过滤（只保留白名单）
            if field_include or field_exclude:
                kv = {k:v for k,v in kv.items()
                      if (not field_include or any(f in k for f in field_include))
                      and (not field_exclude or not any(f in k for f in field_exclude))}
            for k, v in kv.items():
                flat[f"{cname}:{k}"] = v
        return flat

    ta, tb = expand(A), expand(B)
    keys = sorted(set(ta) & set(tb))

    rows, bad = [], []
    for k in keys:
        a, b = ta[k], tb[k]
        if a.shape != b.shape:
            rows.append({"name": k, "shape_a": list(a.shape), "shape_b": list(b.shape),
                         "max_abs": None, "mean_abs": None, "allclose": False})
            continue
        d = (a - b).abs()
        rec = {"name": k, "shape": list(a.shape),
               "max_abs": float(d.max().item()),
               "mean_abs": float(d.mean().item()),
               "allclose": bool(torch.allclose(a, b, rtol=rtol, atol=atol))}
        rows.append(rec)
        if not rec["allclose"]:
            bad.append(rec)

    # bad = sorted(bad, key=lambda r: (-1 if r["max_abs"] is None else r["max_abs"]))[:top_k]
    bad = sorted(bad, key=lambda r: _sort_key(r["name"]))
    print(f"Compared {len(rows)} tensors; not-allclose: {len(bad)}")
    for r in bad:
        if r["max_abs"] is None:
            print(f"[X] {r['name']} | shape mismatch: {r.get('shape_a')} vs {r.get('shape_b')}")
        else:
            print(f"[X] {r['name']} | shape={r['shape']} | max|Δ|={r['max_abs']:.3e} mean|Δ|={r['mean_abs']:.3e}")
    return rows


def _filter_paths(paths: Iterable[str], include: Iterable[str]=None, exclude: Iterable[str]=None):
    def ok(p):
        if include and not any(s in p for s in include): return False
        if exclude and any(s in p for s in exclude): return False
        return True
    return [p for p in paths if ok(p)]

def print_report(rows, top_k=50):
    bad = [r for r in rows if not r["allclose"]]
    bad = sorted(bad, key=lambda r: (-1 if r["max_abs"] is None else r["max_abs"]))[:top_k]
    print(f"Total compared: {len(rows)} | Not allclose: {len(bad)}")
    for r in bad:
        line = f"[X] {r['name']} | "
        if "shape" in r: line += f"shape={r['shape']} | "
        if r["max_abs"] is not None:
            line += f"max|Δ|={r['max_abs']:.3e}, mean|Δ|={r['mean_abs']:.3e}"
        else:
            line += f"shape mismatch: {r.get('shape_a')} vs {r.get('shape_b')}"
        print(line)


rows = diff_runs_normalized(
    "results/debug/acts/run1.pt",
    "results/debug/acts/run2.pt",
    name_include=("L", "HEAD", "ENCODER"),
    field_include=("x","attn_weight","attn_bias"),
    rtol=1e-5, atol=1e-7, top_k=80
)
# print_report(rows, top_k=80)
