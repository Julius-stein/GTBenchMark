
# GraphGym 的 perf 配置补充：增加 E2E 与 DDP 通信计时开关
from GTBenchmark.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config("perf")
def set_cfg_perf(cfg):
    cfg.perf = CN()
    cfg.perf.logdir = "" #勿动！

    # 模式：off | light | e2e | modules | op-only | full
    cfg.perf.mode = "light"

    # 写入与文件管理
    cfg.perf.rank_zero_only= True
    cfg.perf.enable_tensorboard= True
    cfg.perf.tb_flush_secs= 120
    cfg.perf.tb_max_queue= 4000
    cfg.perf.tb_filename_suffix= ""
    cfg.perf.keep_latest_event_files= 2   # <=0 则不清理

    # 采样/写频
    cfg.perf.scalar_every_n_steps= 1

    # E2E
    cfg.perf.enable_e2e= True
    cfg.perf.e2e_log_every_n_steps= 1
    cfg.perf.ewma_alpha= 0.2              # E2E 平滑系数

    # 模块计时
    cfg.perf.enable_module_timer= False   # light/e2e 默认 False，modules/full 默认 True
    cfg.perf.use_cuda_events= True
    cfg.perf.include_name= []
    cfg.perf.include_regex= []
    cfg.perf.exclude_regex= []
    cfg.perf.module_topk_per_window= 20

    # 随机/窗口采样（模块+OP 可共用）
    cfg.perf.random_sampling= False
    cfg.perf.sample_prob= 0.02
    cfg.perf.random_warmup= 3
    cfg.perf.random_active= 10
    cfg.perf.random_cooldown= 50
    cfg.perf.start_after_steps= 100

    # OP profiler
    cfg.perf.enable_op_profiler= False
    cfg.perf.op_exclusive= True           # True 时独占：只跑 OP profiler
    cfg.perf.record_shapes= True
    cfg.perf.profile_memory= True
    cfg.perf.with_stack= True
    cfg.perf.with_modules= True

    # CSV 选项
    cfg.perf.enable_csv= True
    cfg.perf.csv_every_n_steps= 1
    cfg.perf.csv_filename_e2e= "e2e.csv"
    cfg.perf.csv_filename_modules= "modules.csv"

    # NVTX（可用 nsight profile）
    cfg.perf.emit_nvtx= False  
