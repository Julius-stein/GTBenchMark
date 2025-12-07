"""
searchPatches_main.py
遍历一组 patches，不使用 Optuna。
输出的 CSV / JSON 与 Optuna 版完全兼容。
"""

import argparse
import datetime
import json
import logging
import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple

import torch
import pandas as pd
import time

from GTBenchmark.agg_runs import agg_runs
from GTBenchmark.graphgym.cmd_args import parse_args as gg_parse_args
from GTBenchmark.graphgym.config import (cfg, dump_cfg, set_cfg, load_cfg)
from GTBenchmark.graphgym.loader import create_loader
from GTBenchmark.graphgym.logger import setup_printing
from GTBenchmark.graphgym.optimizer import create_optimizer, create_scheduler
from GTBenchmark.graphgym.model_builder import create_model
from GTBenchmark.graphgym.utils.comp_budget import params_count
from GTBenchmark.graphgym.utils.device import auto_select_device
from GTBenchmark.graphgym.register import train_dict
from torch_geometric import seed_everything

from GTBenchmark.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from GTBenchmark.logger import create_logger
from GTBenchmark.utils.utils import (
    new_optimizer_config, new_scheduler_config,
    custom_set_out_dir, custom_set_run_dir
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



# -------------------- 工具函数 --------------------

def _update_cfg_pathlike(cfg_obj, dotted_key: str, value: Any):
    parts = dotted_key.split(".")
    base = cfg_obj
    for p in parts[:-1]:
        base = getattr(base, p)
    setattr(base, parts[-1], value)


def _maybe_maximize(metric_agg: str) -> str:
    return "maximize" if str(metric_agg).lower().strip() == "argmax" else "minimize"


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_search_args():
    gg_args = gg_parse_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--study-name", type=str, default="GT_pro")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--param-limit", type=int, default=-1)
    args2, _ = parser.parse_known_args()

    for k, v in vars(args2).items():
        setattr(gg_args, k.replace("-", "_"), v)
    return gg_args


def _apply_params_to_cfg(cfg_local, hp: Dict[str, Any]):
    for k, v in hp.items():
        _update_cfg_pathlike(cfg_local, k, v)


def _get_gpu_memory_used(gpu_id: int = 0) -> int:
    try:
        out = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}").read().strip()
        return int(out)
    except:
        return 99999



# ================== 你原来的单次训练（保持不动） ==================

def _single_run_with_cfg(args, trial_tag: str):
    original_bs = cfg.train.batch_size if hasattr(cfg.train, "batch_size") else None
    current_bs = original_bs

    MAX_RETRY = 5
    retry = 0

    while True:
        try:
            custom_set_out_dir(cfg, args.cfg_file, f"{cfg.name_tag}patch_{trial_tag}", args.gpu)
            dump_cfg(cfg)
            torch.set_num_threads(cfg.num_threads)
            seed_everything(cfg.seed)

            if args.gpu == -1:
                auto_select_device(strategy="greedy")
            else:
                if cfg.device == "auto":
                    cfg.device = f"cuda:{args.gpu}"

            if current_bs is not None:
                cfg.train.batch_size = current_bs

            setup_printing()
            loaders, dataset = create_loader(returnDataset=True)
            loggers = create_logger()
            model = create_model(dataset=dataset)

            n_params = params_count(model)
            cfg.params = n_params
            if args.param_limit > 0 and n_params > args.param_limit:
                raise ValueError(f"Too many params: {n_params} > {args.param_limit}")

            optimizer = create_optimizer(model.named_parameters(), new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

            run_id = 0
            cfg.run_id = run_id
            custom_set_run_dir(cfg, run_id)

            best_test, best_val, avg_epoch = train_dict[cfg.train.mode](
                loggers, loaders, model, optimizer, scheduler
            )

            try:
                agg_runs(cfg.out_dir, cfg.metric_best)
            except Exception as e:
                logging.info(f"Failed agg_runs: {e}")

            return best_val, best_test, cfg.run_dir, avg_epoch

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise e

            gpu_used = _get_gpu_memory_used(args.gpu)
            if gpu_used > 2000:
                raise ValueError("CUDA OOM and GPU is not idle.")

            if current_bs is None or retry >= MAX_RETRY:
                raise ValueError("CUDA OOM could not be recovered.")

            new_bs = max(1, current_bs // 2)
            if new_bs == current_bs:
                raise ValueError("CUDA OOM and batch_size cannot shrink.")

            logging.warning(
                f"[OOM] trial={trial_tag}, bs={current_bs}→{new_bs}, retry={retry}/{MAX_RETRY}, gpu={gpu_used}MiB"
            )

            current_bs = new_bs
            retry += 1
            torch.cuda.empty_cache()
            time.sleep(1)
            continue



# ================== Patch Sweep 主逻辑 ==================

PATCH_LIST = [
1325, 331, 133, 27, 15, 7, 3
]


def main():
    args = parse_search_args()
    print("==== Running PATCH SWEEP ====")

    all_records = []
    direction = _maybe_maximize(cfg.metric_agg)

    for idx, p in enumerate(PATCH_LIST):
        print(f"\n===== [{idx}] patches={p} =====")

        set_cfg(cfg)
        load_cfg(cfg, args)

        # ---- 设定参数 ----
        hp = {}
        hp["perf.mode"] = "off"
        hp["optim.max_epoch"] = 400
        hp["out_dir"] = "./results/PatchSweep/"
        hp["metis.patches"] = p

        # ---- 动态 batch ----
        TARGET_TOTAL_NODES = 20000
        avg_nodes = 132534 / p
        bs = min(256, max(1, int(TARGET_TOTAL_NODES / avg_nodes)))

        hp["train.batch_size"] = bs
        hp["train.notes"] = f"patch={p},avg_nodes={avg_nodes},bs={bs}"

        _apply_params_to_cfg(cfg, hp)

        best_val, best_test, run_dir, avg_time = _single_run_with_cfg(args, f"patch{p}")

        rec = {
            "trial": idx,
            "value": best_val,
            "state": "COMPLETE",
            "p.metis.patches": p,
            "a.best_test": best_test,
            "a.params_count": getattr(cfg, "params", None),
            "a.avg_epoch_time": avg_time,
            "a.run_dir": run_dir,
            "a.batch_size": bs,
            "a.avg_nodes": avg_nodes,
        }
        all_records.append(rec)

    # ============ 保存 CSV ============

    out_dir = "./results/PatchSweep/"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(all_records)
    csv_path = osp.join(out_dir, f"optuna_all_{args.study_name}.csv")
    df.to_csv(csv_path, index=False)

    print("\n==== Sweep Done ====")
    print(df.to_markdown(index=False))
    print(f"Saved CSV → {csv_path}")

    # 找 best
    if direction == "maximize":
        best_idx = df["value"].idxmax()
    else:
        best_idx = df["value"].idxmin()

    best_row = df.loc[best_idx].to_dict()
    best_json = osp.join(out_dir, f"optuna_best_{args.study_name}.json")
    with open(best_json, "w") as f:
        json.dump(best_row, f, indent=2)

    print(f"Best saved → {best_json}")


if __name__ == "__main__":
    main()
