# utils/act_capture_pyg.py
import os, json, re, torch
from collections import OrderedDict
from typing import Iterable, Union, Type

from torch_geometric.data import Data, Batch


_PYG_IGNORE_FIELDS = {
    # 这些通常是元数据或重复信息，不参与数值对比；按需增减
    "__num_nodes__", "_num_nodes", "_num_edges", "__slices__", "is_sorted",
    "n_id", "e_id", "face", "ptr_dict", "to_dict", "stores", "node_stores", "edge_stores"
}

def _to_cpu_clone_tensor(t: torch.Tensor, cast_fp32=True):
    t = t.detach()
    if cast_fp32 and t.dtype in (torch.float16, torch.bfloat16, torch.float32):
        t = t.float()
    return t.cpu().clone().contiguous()

def _pyg_to_plain_dict(obj, cast_fp32=True):
    """
    将 PyG 的 Data/Batch 展开成 {field_name: value}，其中 value 都是 CPU 张量或基础标量。
    """
    out = OrderedDict()
    # Data/Batch 通常有 keys()
    keys = list(getattr(obj, "keys", lambda: [])())
    # Batch 额外包含 batch、ptr 等
    if isinstance(obj, Batch) and "batch" not in keys and hasattr(obj, "batch"):
        keys += ["batch"]
    if isinstance(obj, Batch) and "ptr" in dir(obj) and hasattr(obj, "ptr"):
        keys += ["ptr"]
    # 去重并保持顺序
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    for k in sorted(keys):
        if k in _PYG_IGNORE_FIELDS:
            continue
        v = getattr(obj, k, None)
        if v is None:
            continue
        if torch.is_tensor(v):
            out[k] = _to_cpu_clone_tensor(v, cast_fp32)
        elif isinstance(v, (int, float, bool)):
            out[k] = v
        else:
            # 递归处理常见容器
            if isinstance(v, (list, tuple)):
                out[k] = [ _to_cpu_clone_tensor(x, cast_fp32) if torch.is_tensor(x) else x for x in v ]
            elif isinstance(v, dict):
                tmp = {}
                for kk, vv in v.items():
                    tmp[kk] = _to_cpu_clone_tensor(vv, cast_fp32) if torch.is_tensor(vv) else vv
                out[k] = tmp
            else:
                # 其他类型先跳过；如果你需要可补充
                pass
    return out

def _to_cpu_clone_any(x, cast_fp32=True, materialize_pyg=True):
    if torch.is_tensor(x):
        return _to_cpu_clone_tensor(x, cast_fp32)
    if isinstance(x, (Data, Batch)):
        return _pyg_to_plain_dict(x, cast_fp32) if materialize_pyg else x.clone().to("cpu")
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu_clone_any(v, cast_fp32, materialize_pyg) for v in x)
    if isinstance(x, dict):
        return {k: _to_cpu_clone_any(v, cast_fp32, materialize_pyg) for k, v in x.items()}
    if isinstance(x, (int, float, bool)):
        return x
    return x

class ActivationCatcher:
    """
    与之前的 ActivationCatcher 类似，但：
      - Data/Batch 会被“字段化”为字典存储（materialize_pyg=True）
      - 支持 include/exclude/regex 选择模块
    """
    def __init__(self,
                 model: torch.nn.Module,
                 include: Iterable[Union[str, Type[torch.nn.Module]]] = None,
                 exclude: Iterable[Union[str, Type[torch.nn.Module]]] = ("Dropout",),
                 regex: Iterable[str] = None,
                 save_dir: str = "debug/acts",
                 cast_fp32: bool = True,
                 capture_inputs: bool = False,
                 materialize_pyg: bool = True):
        self.model = model
        self.include = tuple(include) if include else None
        self.exclude = tuple(exclude) if exclude else None
        self.regex = [re.compile(p) for p in (regex or [])]
        self.save_dir = save_dir
        self.cast_fp32 = cast_fp32
        self.capture_inputs = capture_inputs
        self.materialize_pyg = materialize_pyg
        self.acts = OrderedDict()
        self.inputs = OrderedDict() if capture_inputs else None
        self._handles = []

    def _match(self, name, module):
        ok = True
        if self.include:
            ok = False
            for f in self.include:
                if isinstance(f, str) and f.lower() in name.lower():
                    ok = True; break
                if isinstance(f, type) and isinstance(module, f):
                    ok = True; break
        if ok and self.regex:
            ok = any(p.search(name) for p in self.regex)
        if ok and self.exclude:
            for f in self.exclude:
                if isinstance(f, str) and f.lower() in name.lower():
                    ok = False; break
                if isinstance(f, type) and isinstance(module, f):
                    ok = False; break
        return ok

    def _hook(self, name):
        def fn(module, inp, out):
            self.acts[name] = _to_cpu_clone_any(out, self.cast_fp32, self.materialize_pyg)
        return fn

    def _pre_hook(self, name):
        def fn(module, inp):
            self.inputs[name] = _to_cpu_clone_any(inp, self.cast_fp32, self.materialize_pyg)
        return fn

    def register(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for name, module in self.model.named_modules():
            if not name:
                continue
            if self._match(name, module):
                self._handles.append(module.register_forward_hook(self._hook(name)))
                if self.capture_inputs:
                    self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
        return self

    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()

    def save(self, tag="forward_dump"):
        path = os.path.join(self.save_dir, f"{tag}.pt")
        pkg = {"acts": self.acts}
        if self.inputs is not None: pkg["inputs"] = self.inputs
        torch.save(pkg, path)

        # 生成简短索引（字段级）
        index = []
        def summarize_val(v):
            if torch.is_tensor(v):
                return {"shape": list(v.shape), "dtype": str(v.dtype), "norm": float(v.float().norm().item())}
            return {"shape": None, "dtype": "n/a", "norm": None}
        for layer, val in self.acts.items():
            if isinstance(val, dict):  # Data/Batch 已被字典化
                for k, vv in val.items():
                    if torch.is_tensor(vv):
                        index.append({"name": f"{layer}:{k}", **summarize_val(vv)})
            else:
                index.append({"name": layer, **summarize_val(val)})
        with open(os.path.join(self.save_dir, f"{tag}.index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        return path
