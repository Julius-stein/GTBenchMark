# cache_utils_dataset.py
import os
import json
import hashlib
from typing import Any, Dict

import torch


def _make_hashable(obj: Any):
    """递归将 Python 对象转换为 JSON 可序列化的普通结构。"""
    if isinstance(obj, (list, tuple)):
        return [_make_hashable(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: _make_hashable(v) for k, v in obj.items()}
    else:
        return obj
    
def _flatten_dict(d, prefix=""):
    """
    把多级 dict 展开成 "a.b.c": value 的形式。
    支持任意深度，不会遗漏 kernel、model、inner 等字段。
    """
    out = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full))
        else:
            out[full] = v
    return out



def generate_cache_key(config_dict: Dict[str, Any],
                       plain_keys=None,
                       hash_len: int = 12) -> str:
    """
    给定一个 dict 生成缓存 key：
      - plain_keys 里的字段明文拼在前面
      - 其他字段统一参与 hash
    """
    plain_keys = plain_keys or []

    plain = {k: config_dict[k] for k in plain_keys if k in config_dict}
    to_hash = {k: v for k, v in config_dict.items() if k not in plain_keys}

    serial = json.dumps(_make_hashable(to_hash), sort_keys=True)
    hashed = hashlib.md5(serial.encode("utf-8")).hexdigest()[:hash_len]

    plain_str = "_".join(f"{v}" for k, v in plain.items()) if plain else "nop"
    return f"{plain_str}p-{hashed}"


def build_cache_config_from_cfg(cfg) -> Dict[str, Any]:
    """
    只抽取你关心的少数字段：
      - metis 全部设置
      - 所有 enable 的 posenc_* 模块的全部设置
      - 单独放一个顶层 metis_patches 用于明文
    其余 cfg 字段一律忽略。
    """
    conf: Dict[str, Any] = {}

    # ===== metis 部分 =====
    metis_conf: Dict[str, Any] = {
        "patches": cfg.metis.patches,
        "drop_rate": cfg.metis.drop_rate,
        "num_hops": cfg.metis.num_hops,
        "patch_rw_dim": cfg.metis.patch_rw_dim,
        "patch_num_diff": cfg.metis.patch_num_diff,
    }

    prep_conf: Dict[str, Any] = {
        "prepro": cfg.dataset.preprocess,
        "add_self_loop": cfg.dataset.add_self_loops,
        "add_edge_index": cfg.dataset.add_edge_index,
        "dist_cutoff":  cfg.dataset.dist_cutoff,
        "exp_algorithm":  cfg.dataset.exp_algorithm,
        "exp_count": cfg.dataset.exp_count,
        "exp_deg" : cfg.dataset.exp_deg,
        "exp_max_num_iters" : cfg.dataset.exp_max_num_iters,
        "num_virt_node": cfg.dataset.num_virt_node,
        "rb_order": cfg.dataset.rb_order,
        "use_exp_edges": cfg.dataset.use_exp_edges
    }
    # 有的话就带上
    if hasattr(cfg.metis, "seed"):
        metis_conf["seed"] = cfg.metis.seed

    conf["metis"] = metis_conf
    conf["prep"] = prep_conf
    # 顶层字段，用来做明文
    conf["metis_patches"] = cfg.metis.patches

    # ===== positional encodings 部分 =====
    posenc_conf: Dict[str, Any] = {}

    for key, pecfg in cfg.items():
        if key.startswith("posenc_") and getattr(pecfg, "enable", False):
            # pecfg 可能有多级子字段，必须 flatten 才能被 hash
            # pecfg.items() 是 GraphGym 的 EasyDict，可转换成普通 dict
            nested_dict = {k: v for k, v in pecfg.items()}
            flat_dict = _flatten_dict(nested_dict, prefix=key)
            posenc_conf.update(flat_dict)

    conf["posenc"] = posenc_conf

    return conf


def get_cache_dir(cfg) -> str:
    """
    缓存目录：
      <cfg.dataset.dir>/partitioned/<cfg.dataset.name>/
    """
    return os.path.join(cfg.dataset.dir, "partitioned", cfg.dataset.name)


def get_cache_path(cfg) -> str:
    """
    完整缓存路径：
      <cache_dir>/<cache_key>.pt
    """
    conf = build_cache_config_from_cfg(cfg)
    key = generate_cache_key(conf, plain_keys=["metis_patches"])
    cache_dir = get_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.pt")


def save_dataset_to_cache(dataset, cfg) -> str:
    """
    将 dataset 和必要信息一起保存到缓存文件中。
    你可以根据需要在字典里多存一些东西（如 cfg 快照）。
    """
    cache_path = get_cache_path(cfg)
    torch.save(dataset, cache_path)
    return cache_path


def load_dataset_from_cache(cfg):
    """
    从缓存文件中加载 dataset。如果不存在则返回 None。
    """
    cache_path = get_cache_path(cfg)
    if not os.path.exists(cache_path):
        return None, cache_path

    dataset = torch.load(cache_path, map_location="cpu",weights_only=False)
    return dataset, cache_path
