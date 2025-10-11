import argparse
import datetime
import json
import logging
import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple

import optuna
import torch
import pandas as pd
import time

# ==== 你的工程依赖（原样复用） ====
import GTBenchmark  # noqa, register custom modules
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

from optuna.exceptions import TrialPruned

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------- 工具函数 ----------------
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


#
# !python searchParam_main.py \
#     --cfg path/to/config.yaml \
#     --gpu 0 \
#     --n-trials 30 \
#     --study-name gt_search \
#     --param-limit 6000000
#

# ---------------- CLI ----------------
def parse_search_args():
    gg_args = gg_parse_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--study-name", type=str, default="gt_optuna_search")
    parser.add_argument("--param-limit", type=int, default=20000000,
                        help="Max number of model params allowed, -1 means no limit")
    args2, _ = parser.parse_known_args()
    for k, v in vars(args2).items():
        setattr(gg_args, k.replace("-", "_"), v)
    return gg_args

def _set_search_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    hp = {}
    hp["optim.base_lr"] = trial.suggest_float("optim.base_lr", 1e-5, 1e-3, log=True)
    hp["optim.weight_decay"] = trial.suggest_float("optim.weight_decay", 1e-6, 1e-3, log=True)
    # hp["gt.dim_hidden"] = trial.suggest_categorical("gt.dim_hidden", [64,80, 128,256])
    # hp["gt.ffn_dim"] = trial.suggest_categorical("gt.ffn_dim", [64, 80,128,256])
    # hp["gt.layers"] = trial.suggest_int("gt.layers", 1, 10)
    # hp["gt.attn_heads"] = trial.suggest_categorical("gt.attn_heads", [1,2, 4, 8])
    hp["gt.dropout"] = trial.suggest_float("gt.dropout", 0.0, 0.6)
    hp["gt.attn_dropout"] = trial.suggest_float("gt.attn_dropout", 0.0, 0.8)
    hp["optim.clip_grad_norm_value"] = trial.suggest_float("optim.clip_grad_norm_value", 0.2, 1.0)
    # hp["gnn.layers_pre_mp"] = trial.suggest_int("gnn.layers_pre_mp", 0, 1)
    return hp


def _apply_params_to_cfg(cfg_local, hp: Dict[str, Any]):
    for k, v in hp.items():
        _update_cfg_pathlike(cfg_local, k, v)


def _single_run_with_cfg(args, trial_tag: str) -> Tuple[Optional[float], Optional[float], str]:
    # set_cfg(cfg)
    # load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, f"{cfg.name_tag}optuna_{trial_tag}", args.gpu)
    dump_cfg(cfg)
    torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)

    if args.gpu == -1:
        auto_select_device(strategy="greedy")
    else:
        if cfg.device == "auto":
            cfg.device = f"cuda:{args.gpu}"

    setup_printing()
    loaders, dataset = create_loader(returnDataset=True)
    loggers = create_logger()
    model = create_model(dataset=dataset)

    # 参数总量检查
    n_params = params_count(model)
    cfg.params = n_params
    if args.param_limit > 0 and n_params > args.param_limit:
        raise TrialPruned(f"Too many params: {n_params} > {args.param_limit}")

    optimizer = create_optimizer(model.named_parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

    run_id = 0
    cfg.run_id = run_id
    custom_set_run_dir(cfg, run_id)

    best_test, best_val = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")

    return best_val, best_test, cfg.run_dir


def build_objective(args):
    set_cfg(cfg)
    load_cfg(cfg, args)
    direction = _maybe_maximize(cfg.metric_agg)

    def objective(trial: optuna.trial.Trial) -> float:
        set_cfg(cfg)
        load_cfg(cfg, args)
        hp = _set_search_params(trial)
        _apply_params_to_cfg(cfg, hp)

        best_val, best_test, run_dir = _single_run_with_cfg(args, str(trial.number))

        if best_val is None:
            best_val = -1e12 if direction == "maximize" else 1e12

        trial.set_user_attr("run_dir", run_dir)
        trial.set_user_attr("best_test", best_test)
        trial.set_user_attr("params_count", getattr(cfg, "params", None))
        for k, v in hp.items():
            trial.set_user_attr(k, v)
        return float(best_val)

    return objective, direction





def main():
    args = parse_search_args()
    objective, direction = build_objective(args)

    study = optuna.create_study(
        study_name=args.study_name,
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=getattr(cfg, "seed", 42)),
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print("\n=== Optuna Best Trial ===")
    print(f"Trial #{best.number} | Value={best.value}")
    print(f"Best Test={best.user_attrs.get('best_test', None)}")
    print(f"Params={best.user_attrs.get('params_count', None)}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out_dir = getattr(cfg, "out_dir", "./results")
    os.makedirs(out_dir, exist_ok=True)
    best_path = osp.join(out_dir, f"optuna_best_{args.study_name}.json")
    with open(best_path, "w") as f:
        json.dump({"value": best.value, "params": best.params,
                   "user_attrs": best.user_attrs}, f, indent=2)
    print(f"\nBest trial saved to: {best_path}")

    # === 导出所有 trial ===
    records = []
    for t in study.trials:
        rec = {"trial": t.number, "value": t.value, "state": str(t.state)}
        rec.update({f"p.{k}": v for k, v in t.params.items()})
        rec.update({f"a.{k}": v for k, v in t.user_attrs.items()})
        if "best_test" in t.user_attrs:
            rec["best_test"] = t.user_attrs["best_test"]
        if "params_count" in t.user_attrs:
            rec["params_count"] = t.user_attrs["params_count"]
        records.append(rec)
    df = pd.DataFrame(records)
    all_path = osp.join(out_dir, f"optuna_all_{args.study_name}.csv")
    df.to_csv(all_path, index=False)
    print("\n=== All Trials Summary ===")
    print(df.to_markdown(index=False))
    print(f"\nAll trials saved to: {all_path}")


if __name__ == "__main__":
    main()
