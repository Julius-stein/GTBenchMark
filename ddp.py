# train_ddp_main.py
import os
import datetime
import logging
import torch
import torch.distributed as dist

import GTBenchmark
from torch_geometric import seed_everything

from GTBenchmark.graphgym.cmd_args import parse_args
from GTBenchmark.graphgym.config import (
    cfg, set_cfg, load_cfg, dump_cfg
)
from GTBenchmark.graphgym.logger import setup_printing
from GTBenchmark.graphgym.model_builder import create_model
from GTBenchmark.graphgym.optimizer import create_optimizer, create_scheduler
from GTBenchmark.graphgym.utils.device import auto_select_device
from GTBenchmark.graphgym.utils.comp_budget import params_count
from GTBenchmark.graphgym.register import train_dict

from GTBenchmark.finetuning import (
    load_pretrained_model_cfg,
    init_model_from_pretrained
)

from GTBenchmark.logger import create_logger
from GTBenchmark.utils.utils import (
    new_optimizer_config,
    new_scheduler_config,
    custom_set_out_dir,
    custom_set_run_dir,
)

from GTBenchmark.utils.dataset_cache import (
    load_dataset_from_cache,
    save_dataset_to_cache,
)


#############################################
# 分布式初始化
#############################################
def init_ddp():
    if "RANK" not in os.environ:
        raise RuntimeError("You must run this script with torchrun.")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return rank, world, local_rank


#############################################
# 从缓存加载 dataset；若不存在则仅 rank0 创建
#############################################
def load_or_create_dataset(cfg, rank):
    dataset, cache_path = load_dataset_from_cache(cfg)

    if dataset is not None:
        if rank == 0:
            print(f"[DDP] Using dataset cache: {cache_path}")
        return dataset

    # ====== 缓存不存在：只有 rank0 创建 ======
    if rank == 0:
        print("[DDP] No dataset cache detected. Building dataset...")

        # ★★★ 这里你将 create_dataset() 替换成图划分前处理的过程 ★★★
        # 比如：
        # dataset = do_graph_partition(cfg)
        #
        # 你现在要求 "loader 先空着"，所以这里返回一个 placeholder dict：
        dataset = {
            "train": None,  # 稍后你自己填回真实的划分结果
            "val": None,
            "test": None,
        }

        cache_path = save_dataset_to_cache(dataset, cfg)
        print(f"[DDP] Saved dataset cache: {cache_path}")

    # 等 rank0 建完缓存
    dist.barrier()

    # 所有 rank 重新加载
    dataset, cache_path = load_dataset_from_cache(cfg)
    if dataset is None:
        raise RuntimeError("Fatal: dataset cache missing after build.")

    return dataset


#############################################
# DDP 主逻辑（训练）
#############################################
def main():
    global cfg
    args = parse_args()

    # ========= 基本配置 ============
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
    dump_cfg(cfg)

    # ========= DDP 初始化 ============
    rank, world, local_rank = init_ddp()
    is_master = (rank == 0)

    if is_master:
        logging.info(f"[DDP] world_size={world}")
        logging.info(f"[DDP] Training starts at {datetime.datetime.now()}")

    seed_everything(cfg.seed)

    # 设置 GPU
    if cfg.device == "auto":
        cfg.device = f"cuda:{local_rank}"

    # ========= 数据集：从缓存读取 ============
    dataset = load_or_create_dataset(cfg, rank)

    # ========= 创建模型 ============
    model = create_model(dataset=dataset)
    if cfg.pretrained.dir:
        model = init_model_from_pretrained(
            model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
            cfg.pretrained.reset_prediction_head, seed=cfg.seed
        )

    device = torch.device(cfg.device)
    model.to(device)

    # ========= 分布式包装模型 ============
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # ========= Optimizer / Scheduler ============
    optimizer = create_optimizer(model.named_parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

    # ========= Logger ============
    if is_master:
        setup_printing()
    loggers = create_logger() if is_master else None

    # ======= 训练前信息输出 ========
    if is_master:
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info(f"Num parameters: {cfg.params}")

    # ========= ★★★ DataLoader 你稍后自己实现 ★★★ ============
    # 暂时空着：
    # loaders = {"train": your_train_loader, "val": your_val_loader, ...}
    loaders = {"train": None, "val": None, "test": None}

    # ========= 训练启动 ============
    train_func = train_dict[cfg.train.mode]
    train_func(loggers, loaders, model, optimizer, scheduler)

    # ========= 清理 ============
    if is_master:
        logging.info(f"[DDP] Training finished at: {datetime.datetime.now()}")
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
