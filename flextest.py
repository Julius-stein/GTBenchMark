# bench_flex_vs_sdpa.py
import math, time, torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 100      # 或 64
dynamo.config.suppress_errors = False     # 出错别静默回退


# -------------------
# 全局配置
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
TORCH_COMPILE = True
ENABLE_PROFILER = False
SDP_BACKEND = SDPBackend.EFFICIENT_ATTENTION
WARMUP_ITERS = 5
MEASURE_ITERS = 20
MASK_STYLE = "random"   # "random"（块随机）或 "banded"（对角带状）
SEED = 42
BLOCK_SIZE = 128
torch.manual_seed(SEED)

# GRID = [
#     (8, 8,  128, 128),
#     (1024,8,  128, 128),
#     (8,  8, 2048, 128),
#     (8,  8, 1024, 256),
#     (8,  8, 4096, 128),
# ]
GRID = [
    (32, 8,  128, 256),
    (1,8,  4096, 256),

]

# 注意：这是“块级保留比例”，不是逐元素
DENSITIES = [1.0,0.5, 0.25, 0.125,0.01,0.005]
BLOCK_SIZES = [64]  # 可按需加减

# -------------------
# 实用函数
# -------------------
def make_qkv(B, H, L, E):
    x  = torch.randn(B, L, H * E, device=DEVICE, dtype=DTYPE)
    Wq = torch.nn.Linear(H * E, H * E, bias=False, device=DEVICE, dtype=DTYPE)
    Wk = torch.nn.Linear(H * E, H * E, bias=False, device=DEVICE, dtype=DTYPE)
    Wv = torch.nn.Linear(H * E, H * E, bias=False, device=DEVICE, dtype=DTYPE)
    q = Wq(x).reshape(B, L, H, E).transpose(1, 2).contiguous()
    k = Wk(x).reshape(B, L, H, E).transpose(1, 2).contiguous()
    v = Wv(x).reshape(B, L, H, E).transpose(1, 2).contiguous()
    return q, k, v

def make_bias(B, H, L):
    return (torch.randn(B, H, L, L, device=DEVICE, dtype=DTYPE) * 0.1).contiguous()

def ceil_div(a, b): return (a + b - 1) // b

def build_tile_keep(B, H, L, density, block_size, style="random"):
    """
    返回 tile_keep[B,H,gq,gk]：按“块”保留与否（True=保留）
    """
    gq = ceil_div(L, block_size)
    gk = ceil_div(L, block_size)
    if style == "random":
        # 每个块独立伯努利采样
        keep = (torch.rand(1, 1, gq, gk, device=DEVICE) < float(density)).expand(B, H, gq, gk).contiguous()
    elif style == "banded":
        # 对角带宽（以块为单位）：density∈(0,1] → 近似成带宽 round(density * max(gq,gk))
        bw = max(1, int(round(density * max(gq, gk))))
        grid_q = torch.arange(gq, device=DEVICE)[:, None]
        grid_k = torch.arange(gk, device=DEVICE)[None, :]
        band = (torch.abs(grid_q - grid_k) < bw)  # [gq,gk]
        keep = band.to(torch.bool)[None, None, :, :].expand(B, H, gq, gk).contiguous()
    else:
        raise ValueError(style)
    return keep  # bool

def make_block_mask_from_tilekeep(tile_keep, block_size, L):
    """
    基于 tile_keep 构建 create_block_mask 所需的谓词。
    不使用 .item()，返回 torch.bool 张量以兼容 vmap。
    """
    B, H, gq, gk = tile_keep.shape
    def pred(b, h, q_idx, kv_idx):
        qb = q_idx // block_size
        kb = kv_idx // block_size
        # 越界块按最后一块处理（便于 L 不是 block 的整数倍时工作）
        qb = torch.clamp(qb, max=gq - 1)
        kb = torch.clamp(kb, max=gk - 1)
        return tile_keep[b, h, qb, kb]
    return create_block_mask(pred, B=B, H=H, Q_LEN=L, KV_LEN=L, device=DEVICE,BLOCK_SIZE=block_size)


@torch.compile(fullgraph=True, dynamic=False,mode="max-autotune")
def flex_call(q, k, v, bias, block_mask):
    def score_mod(score, b, h, qi, ki):
        return score + bias[b, h, qi, ki]  # 先用最简单可融合的加性改分
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask,
                          kernel_options={
                                "BLOCK_M": 32,
                                "BLOCK_N": 32,
                                # "BLOCK_M1": 32,
                                # "BLOCK_N1": 32,
                                # "BLOCK_M2": 32,
                                # "BLOCK_N2": 32,                
                            })

def sdpa_call(q, k, v, bias):
    return scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)

def time_op(fn, *args):
    if DEVICE == "cuda": torch.cuda.synchronize()
    for _ in range(WARMUP_ITERS): _ = fn(*args)
    if DEVICE == "cuda": torch.cuda.synchronize()
    start = torch.cuda.Event(True) if DEVICE == "cuda" else None
    end   = torch.cuda.Event(True) if DEVICE == "cuda" else None
    t0 = time.perf_counter()
    if DEVICE == "cuda": start.record()
    for _ in range(MEASURE_ITERS): _ = fn(*args)
    if DEVICE == "cuda":
        end.record(); torch.cuda.synchronize()
        return start.elapsed_time(end) / MEASURE_ITERS
    else:
        return (time.perf_counter() - t0) * 1000.0 / MEASURE_ITERS

# def maybe_compile(f):
#     if TORCH_COMPILE and hasattr(torch, "compile"):
#         try: return torch.compile(f, fullgraph=False)
#         except Exception: return f
#     return f

# -------------------
# 主流程
# -------------------
def main():
    torch.set_float32_matmul_precision("high")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"SDPA backend preference: {SDP_BACKEND.name}")
    header = f"{'B':>4} {'H':>3} {'L':>6} {'E':>5} {'blk':>4} {'tile_d':>7} | {'Flex(ms)':>9} {'SDPA(ms)':>9} {'speedup':>8} {'winner':>7}"
    print(header); print("-"*len(header))

    with sdpa_kernel(SDPBackend.DEFAULT if SDP_BACKEND is None else SDP_BACKEND):
        for (B, H, L, E) in GRID:
            if (E & (E - 1)) != 0:
                print(f"Skip (E not power-of-two): B={B},H={H},L={L},E={E}")
                continue
            try:
                q, k, v = make_qkv(B, H, L, E)
                bias = make_bias(B, H, L)
                flex_fn = (lambda Q, K, V, BI, BM: flex_call(Q, K, V, BI, BM))
                sdpa_fn = (lambda Q, K, V, BI: sdpa_call(Q, K, V, BI))
                for blk in BLOCK_SIZES:
                    for dens in DENSITIES:
                        # dens==1.0：不传 block_mask，避免额外开销
                        if dens >= 0.999:
                            block_mask = None
                            tile_d = 1.0
                        else:
                            tile_keep = build_tile_keep(B, H, L, dens, blk, style=MASK_STYLE)
                            tile_d = float(tile_keep.float().mean().item())  # 真·块级密度
                            block_mask = make_block_mask_from_tilekeep(tile_keep, blk, L)

                        t_flex = time_op(flex_fn, q, k, v, bias, block_mask)
                        t_sdpa = time_op(sdpa_fn, q, k, v, bias)
                        spdup = (t_sdpa / t_flex) if t_flex > 0 else float("inf")
                        winner = "flex" if t_flex < t_sdpa else "sdpa"
                        print(f"{B:4d} {H:3d} {L:6d} {E:5d} {blk:4d} {tile_d:7.4f} | {t_flex:9.3f} {t_sdpa:9.3f} {spdup:8.3f} {winner:>7}")

            except RuntimeError as e:
                print(f"Skip B={B},H={H},L={L},E={E} due to error: {repr(e)}")
                if "out of memory" in str(e).lower() and DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue

if __name__ == "__main__":
    main()
    # Profiler 代码保留但禁用
