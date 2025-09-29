import copy
import logging
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import mask_to_index, index_to_mask
from GTBenchmark.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.loader import create_loader, get_loader
from GTBenchmark.graphgym.loss import compute_loss
from GTBenchmark.graphgym.register import register_train
from GTBenchmark.graphgym.utils.comp_budget import params_count
from GTBenchmark.utils.act_capture import ActivationCatcher



STEP_SCHED_NAMES = {
    'linear_with_warmup', 'cosine_with_warmup',
    'polynomial_with_warmup', 'polydecay_warmup_steps'
}
EPOCH_SCHED_NAMES = {
    'polydecay_warmup_epoch', 'plateau', 'reduce_on_plateau'
}
'''Graphormer Model:
GraphGymModule(
  (model): GraphormerModel(
    (encoder): FeatureEncoder(
      (node_encoder): Concat2NodeEncoder(
        (encoder1): TypeDictNodeEncoder(
          (encoder): Embedding(28, 80)
        )
        (encoder2): GraphormerEncoder(
          (0): BiasEncoder(
            (spatial_encoder): Embedding(21, 8)
            (edge_dis_encoder): Embedding(1280, 1)
            (edge_encoder): Embedding(4, 8)
          )
          (1): NodeEncoder(
            (in_degree_encoder): Embedding(64, 80)
            (out_degree_encoder): Embedding(64, 80)
            (input_dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (layers): Sequential(
      (0): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (1): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (2): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (3): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (4): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (5): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (6): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (7): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (8): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (9): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (10): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
      (11): GraphormerLayer(
        (attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=80, out_features=80, bias=True)
        )
        (input_norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (mlp): Sequential(
          (0): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=80, out_features=80, bias=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=80, out_features=80, bias=True)
          (5): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (post_mp): GraphormerHead(
      (ln): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
      (layers): Sequential(
        (0): Linear(in_features=80, out_features=1, bias=True)
      )
    )
  )
)
'''
# —— 精确抓取的正则 —— #
regex_targets = [
    # 编码阶段：你关心“进入 Transformer 前”的表示（可选）
    r"^model\.encoder$",                                        # FeatureEncoder 整体输出（最实用）
    # 若想更细：以下三条按需打开
    # r"^model\.encoder\.node_encoder\.encoder1\.encoder$",     # TypeDictNodeEncoder.Embedding(28,80)
    # r"^model\.encoder\.node_encoder\.encoder2\.0$",           # GraphormerEncoder.(0)=BiasEncoder
    # r"^model\.encoder\.node_encoder\.encoder2\.1$",           # GraphormerEncoder.(1)=NodeEncoder

    # Transformer 主体：12 层
    r"^model\.layers\.\d+$",                    # 每个 GraphormerLayer 块级输出（残差之后）
    r"^model\.layers\.\d+\.input_norm$",        # pre-norm 输出（送入 attention 之前的 x̂）
    r"^model\.layers\.\d+\.attention$",         # attention 输出（包含 out_proj 之后）
    r"^model\.layers\.\d+\.mlp\.1$",            # FFN 第一层 Linear 输出（激活前）
    r"^model\.layers\.\d+\.mlp\.4$",            # FFN 第二层 Linear 输出（激活后、dropout 前）

    # Head
    r"^model\.post_mp\.ln$",                    # 最后 LayerNorm
    r"^model\.post_mp\.layers\.0$",             # 最终 Linear -> logits
]
REGEX_TARGETS_AUTO = [
    r"^(?:model\.)?encoder$",                                          # 进入 Transformer 前
    r"^(?:model\.)?(?:layers|convs)\.\d+$",                            # 块级输出（每层）
    r"^(?:model\.)?(?:layers|convs)\.\d+\.(?:input_norm|norm1)$",      # pre-norm
    r"^(?:model\.)?(?:layers|convs)\.\d+\.attention$",                 # attention 输出
    r"^(?:model\.)?(?:layers|convs)\.\d+\.(?:mlp\.1|fc1)$",            # FFN Linear1
    r"^(?:model\.)?(?:layers|convs)\.\d+\.(?:mlp\.4|fc2)$",            # FFN Linear2
    r"^(?:model\.)?(?:post_mp\.ln|post_gt\.ln)$",                      # Head LN
    r"^(?:model\.)?(?:post_mp\.layers\.0|post_gt\.fc)$",               # Head FC/Logits
]

@register_train('debug')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """

    # include_prefix = ('model.encoder', 'model.layers.0', 'model.layers.1',
    #               'model.layers.2')

    # load_partial(
    #     model,
    #     "4999.ckpt",
    #     key_in_ckpt="model_state",
    #     include_prefix=include_prefix,
    # )
    # model.load_state_dict(torch.load("4999.ckpt", weights_only=True)['model_state'])
    msg = load_ckpt_gtmodel(model, "../GraphGPS_origin/GraphGPS-main/4999.ckpt", strict=False, verbose=1)
    cfg.run_dir = "results/debug/log"
    batch = torch.load("results/debug/first_batch.pt", map_location="cpu",weights_only=False)
    model.eval()
    optimizer.zero_grad()
    it = 0
    torch.cuda.empty_cache() 
    it += 1
    catcher = ActivationCatcher(
        model,
        include=None,  # 名字匹配 or 模块类型匹配任选
            # 也可以写类型：torch.nn.TransformerEncoderLayer, torch.nn.LayerNorm
        exclude=["Dropout"],
        # 也可以用正则精准锁定：比如只抓 encoder.layers.[0-9]+$
        regex=REGEX_TARGETS_AUTO,           # 例如：regex=[r"^encoder\.layers\.\d+$"]
        save_dir="results/debug/acts",
        cast_fp32=True,     # 把半精度/bfloat16转成fp32保存，便于比较
        capture_inputs=False  # 若想同时保存每层输入，设为 True
    ).register()
    if isinstance(batch, Data) or isinstance(batch, HeteroData):
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
    else: # NAGphormer, HINo
        batch = [x.to(torch.device(cfg.device)) for x in batch]
    with torch.inference_mode():
        pred, true = model(batch)

    loss, pred_score = compute_loss(pred, true)

    _true = true.detach().to('cpu', non_blocking=True)
    _pred = pred_score.detach().to('cpu', non_blocking=True)
    catcher.remove()
    #! 记得改tag
    dump_path = catcher.save(tag="run2")
    print(f"[OK] 每层输出已保存到: {dump_path}")
    cfg.params = params_count(model)
    loggers[0].update_stats(true=_true,
                        pred=_pred,
                        loss=loss.detach().cpu().item(),
                        params=cfg.params,
                        lr=0,
                        time_used=0,
                        dataset_name=cfg.dataset.name)
    loggers[0].write_epoch(0)
    logging.info('Task done, results saved in %s', cfg.run_dir)

import re, torch
from typing import Iterable, Tuple

def load_ckpt_gtmodel(
    model,
    ckpt_path: str,
    key_candidates: Tuple[str, ...] = ("model_state", "state_dict", "model", "net"),
    map_location="cpu",
    strip_prefixes=("module.",),
    include_regex: Iterable[str] = (),
    exclude_regex: Iterable[str] = (),
    strict: bool = False,
    dry_run: bool = False,
    freeze_loaded: bool = False,
    verbose: int = 1,
):
    import re, torch
    raw = torch.load(ckpt_path, map_location=map_location)
    src = None
    for k in key_candidates:
        if k in raw:
            src = raw[k]
            break
    if src is None:
        src = raw

    def _strip(k: str) -> str:
        for sp in strip_prefixes:
            if k.startswith(sp):
                return k[len(sp):]
        return k
    src = { _strip(k): v for k, v in src.items() }

    dst = model.state_dict()
    new = {}

    inc_res = [re.compile(p) for p in include_regex]
    exc_res = [re.compile(p) for p in exclude_regex]
    def _keep(k: str) -> bool:
        if inc_res and not any(r.search(k) for r in inc_res):
            return False
        if exc_res and any(r.search(k) for r in exc_res):
            return False
        return True

    # 基础重命名
    def basic_rename(k: str) -> str:
        k = re.sub(r'^model\.', '', k)
        k = re.sub(r'^encoder\.node_encoder\.encoder1\.', 'encoder.encoder.0.', k)
        k = re.sub(r'^encoder\.node_encoder\.encoder2\.', 'encoder.encoder.1.', k)
        k = re.sub(r'^layers\.(\d+)\.', r'convs.\1.', k)
        k = re.sub(r'^convs\.(\d+)\.input_norm\.(.*)$', r'convs.\1.norm1.0.\2', k)
        k = re.sub(r'^convs\.(\d+)\.mlp\.0\.(.*)$',     r'convs.\1.norm2.0.\2', k)
        k = re.sub(r'^convs\.(\d+)\.mlp\.1\.(.*)$',     r'convs.\1.fc1.\2', k)
        k = re.sub(r'^convs\.(\d+)\.mlp\.4\.(.*)$',     r'convs.\1.fc2.\2', k)
        k = re.sub(r'^post_mp\.ln\.',         'post_gt.ln.', k)
        k = re.sub(r'^post_mp\.layers\.0\.',  'post_gt.fc.', k)
        return k

    # 注意力映射
    def map_attention(k_src: str, v, dst):
        m = re.match(r'^(?:model\.)?layers\.(\d+)\.attention\.(.+)$', k_src)
        if not m: return False
        lid, tail = int(m.group(1)), m.group(2)
        base = f'convs.{lid}.attention.'

        if tail.startswith('out_proj.'):
            for cand in [base + tail, base + tail.replace('out_proj.', 'o_proj.')]:
                if cand in dst and dst[cand].shape == v.shape:
                    new[cand] = v; return True
            return False
        #!新模型和MAH的命名差别
        if tail in ('in_proj_weight', 'in_proj_bias'):
            cand = base + ('in_proj.weight' if tail == 'in_proj_weight' else 'in_proj.bias')
            # 1) 模型里本来就有 in_proj_*，直接对齐
            if cand in dst and dst[cand].shape == v.shape:
                new[cand] = v
                return True

            # 2) 否则再尝试拆成 q/k/v
            if tail == 'in_proj_weight':
                D = v.shape[0] // 3
                new[base+'q_proj.weight'] = v[0:D, :]
                new[base+'k_proj.weight'] = v[D:2*D, :]
                new[base+'v_proj.weight'] = v[2*D:3*D, :]
                return True
            else:
                D = v.shape[0] // 3
                new[base+'q_proj.bias'] = v[0:D]
                new[base+'k_proj.bias'] = v[D:2*D]
                new[base+'v_proj.bias'] = v[2*D:3*D]
                return True


        k_dst = basic_rename(k_src)
        if k_dst in dst and dst[k_dst].shape == v.shape:
            new[k_dst] = v; return True

        if tail.startswith("W_"):
            mapping = {"W_q":"q_proj", "W_k":"k_proj", "W_v":"v_proj", "W_o":"out_proj"}
            prefix = tail.split('.')[0]
            name = mapping.get(prefix, prefix) + "." + tail.split('.',1)[1]
            k_dst = base + name
            if k_dst in dst and dst[k_dst].shape == v.shape:
                new[k_dst] = v; return True

        return False

    # 遍历
    skipped_shape = []
    unmatched_ckpt = []   # ckpt里没用上的
    for k, v in src.items():
        if not _keep(k): continue
        if map_attention(k, v, dst):
            continue
        k2 = basic_rename(k)
        if k2 in dst and dst[k2].shape == v.shape:
            new[k2] = v
        else:
            if k2 in dst and dst[k2].shape != v.shape:
                skipped_shape.append((k, k2, tuple(v.shape), tuple(dst[k2].shape)))
            else:
                unmatched_ckpt.append((k, tuple(v.shape)))

    # 模型缺失的（即目标有参数但没在 new 中出现）
    unmatched_model = [ (k, tuple(v.shape)) for k,v in dst.items() if k not in new ]

    if verbose:
        print(f"[load_ckpt_gtmodel] ckpt_keys={len(src)}, matched={len(new)}")
        if skipped_shape:
            print("[shape mismatch]")
            for rec in skipped_shape[:10]:
                print(" ", rec)
        if unmatched_ckpt:
            print("[ckpt keys not matched]")
            for rec in unmatched_ckpt[:10]:
                print(" ", rec)
        if unmatched_model:
            print("[model params not initialized from ckpt]")
            for rec in unmatched_model[:10]:
                print(" ", rec)

    if dry_run:
        class _Dummy:
            missing_keys=[]; unexpected_keys=[]
        return _Dummy()

    msg = model.load_state_dict(new, strict=strict)

    if verbose:
        print("load_state_dict.missing_keys:", msg.missing_keys[:])
        print("load_state_dict.unexpected_keys:", msg.unexpected_keys[:])

    if freeze_loaded:
        loaded_set = set(new.keys())
        for n,p in model.named_parameters():
            if n in loaded_set:
                p.requires_grad=False
        if verbose:
            n_fz=sum(int(not p.requires_grad) for _,p in model.named_parameters())
            print(f"[freeze] frozen={n_fz}")

    return msg

