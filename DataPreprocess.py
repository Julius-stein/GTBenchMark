# preprocess_dataset.py
import os
import datetime

import GTBenchmark  # noqa: F401 注册自定义模块
from torch_geometric import seed_everything

from GTBenchmark.graphgym.cmd_args import parse_args
from GTBenchmark.graphgym.config import cfg, set_cfg, load_cfg
from GTBenchmark.graphgym.loader import create_dataset

from GTBenchmark.utils.dataset_cache import (
    save_dataset_to_cache,
    get_cache_path,
    load_dataset_from_cache,
)


def main():
    global cfg
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)

    seed_everything(cfg.seed)

    print(f"[*] Data PreProcess seed={cfg.seed}")
    print(f"    Starting at: {datetime.datetime.now()}")

    # 如果已经有缓存，就直接提示一下（避免重复干重活）
    dataset_cached, cache_path = load_dataset_from_cache(cfg)
    if dataset_cached is not None:
        print(f"[!] Cache already exists: {cache_path}")
        print("    如果要重建缓存，先手动删除该文件。")
        return

    # 正常创建数据集（这里会做 LapPE / posenc / metis 划分等各种 heavy transform）
    dataset = create_dataset()

    cache_path = save_dataset_to_cache(dataset, cfg)
    print(f"[+] Saved processed dataset cache to: {cache_path}")
    print(f"    Finished at: {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
