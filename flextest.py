# bench_flex_vs_sdpa.py
import math, time, torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 100      # 或 64
dynamo.config.suppress_errors = False     # 出错别静默回退
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


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
    # (1, 8,  128, 256),
    (1, 8,  512, 256),
    (1, 8,  1024, 256),
    (1, 8,  4096, 256),
    (1, 8,  8192, 256),
    (1, 8,  10112, 256),

]

# 注意：这是“块级保留比例”，不是逐元素
DENSITIES = [1.0,0.5, 0.125,0.01,0.005,0.00005]
BLOCK_SIZES = [128]  # 可按需加减

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

def _params_text_from_row(row: pd.Series) -> str:
    """把无关参数整理成一段放图内的说明文字。"""
    backend = getattr(SDP_BACKEND, "name", str(SDP_BACKEND))
    return (
        f"B={int(row.B)}, H={int(row.H)}, E={int(row.E)}, blk={int(row.blk)}\n"
        f"BLOCK_SIZE={BLOCK_SIZE}, MASK_STYLE='{MASK_STYLE}'\n"
        f"SDP_BACKEND={backend}, DTYPE={str(DTYPE).split('.')[-1]}, DEVICE={DEVICE}\n"
        # f"WARMUP={WARMUP_ITERS}, MEASURE={MEASURE_ITERS}, torch.compile={TORCH_COMPILE}"
    )

def plot_speedup_vs_L_with_params(df: pd.DataFrame):
    """
    画 'Speedup vs L' 曲线，并在图内列出其它固定参数。
    如果 (B,H,E,blk) 有多组配置，则逐组单独出图。
    """
    # 只考虑有效 speedup
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["speedup", "L", "density_nominal"])

    # 以 (B,H,E,blk) 分组，一组出一张图
    group_keys = ["B", "H", "E", "blk"]
    for gkey, grp in df.groupby(group_keys):
        plt.figure(figsize=(6, 4))
        # 每条曲线一个 density
        for dens, sub in grp.groupby("density_nominal"):
            sub = sub.sort_values("L")
            # 只画 L 维度变化的趋势；如果该 density 下 L 只有一个点则仍然画点
            xs = sub["L"].to_numpy()
            ys = sub["speedup"].to_numpy()
            plt.plot(xs, ys, "o-", label=f"density={dens:.5f}")

        # 轴与标题
        plt.xscale("log")
        plt.xlabel("Sequence length L (log scale)")
        plt.ylabel("Speedup (SDPA / Flex)")
        plt.title("Flex vs SDPA — Speedup vs L")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Block density", loc="upper left")

        # 在图内角落加固定参数说明
        any_row = grp.iloc[0]
        info_txt = _params_text_from_row(any_row)
        plt.gca().text(
            0.98, 0.02, info_txt,
            transform=plt.gca().transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.08)
        )

        # 如果 L 点足够多，也可以给每条 density 做一个粗略“随 L 幂律”拟合并打印斜率
        # 注：只做文本输出，不额外画拟合线，避免图面过杂
        for dens, sub in grp.groupby("density_nominal"):
            sub = sub.sort_values("L")
            if len(sub) >= 3:
                # 拟合: speedup ≈ a * L^b + c 过于不适定；这里用简化的 log-log 线性拟合 speedup ≈ A * L^b
                x = sub["L"].to_numpy(dtype=float)
                y = sub["speedup"].to_numpy(dtype=float)
                # 仅使用正值
                mask = (x > 0) & (y > 0)
                x, y = x[mask], y[mask]
                if len(x) >= 3:
                    lx, ly = np.log(x), np.log(y)
                    b, a = np.polyfit(lx, ly, 1)  # ly ≈ a + b*lx -> y ≈ exp(a) * L^b
                    print(f"[L-trend] group={gkey}, density={dens:.5f}: speedup ~ L^{b:.3f}")

        # 保存单图，文件名写上关键参数
        B, H, E, blk = map(int, gkey)
        fname = f"speedup_vs_L_B{B}_H{H}_E{E}_blk{blk}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

def analyze_results(df: pd.DataFrame):
    # ---- 统计摘要 ----
    avg_speedup = df['speedup'].mean()
    flex_win_ratio = (df['winner'] == 'flex').mean()
    print("\n=== Summary ===")
    print(f"平均加速比: {avg_speedup:.3f}x")
    print(f"Flex胜出比例: {flex_win_ratio*100:.1f}%")

    # ---- 按density绘图 ----
    plt.figure(figsize=(6,4))
    for key, grp in df.groupby(['B','H','L','E']):
        plt.plot(grp['density_actual'], grp['speedup'], 'o-', label=f"B{key[0]}L{key[2]}")
    plt.xlabel("Actual density (block-level keep ratio)")

    plt.xlabel("Density (block-level keep ratio)")
    plt.ylabel("Speedup (SDPA / Flex)")
    plt.title("Flex vs SDPA Speedup vs Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("speedup_vs_density.png", dpi=200)
    plot_speedup_vs_L_with_params(df)

    # ---- 可选：拟合曲线 ----
    def model(x, a, b, c):  # 幂律拟合: speedup = a * x^b + c
        return a * np.power(x, b) + c
    xdata = df['density_actual'].to_numpy()
    ydata = df['speedup'].to_numpy()
    popt, _ = curve_fit(model, xdata, ydata, maxfev=10000)
    print(f"拟合结果: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}")

    xs = np.linspace(min(xdata), max(xdata), 200)
    plt.figure(figsize=(5,4))
    plt.scatter(xdata, ydata, s=10, alpha=0.5)
    plt.plot(xs, model(xs, *popt), 'r--', label='Power-law fit')
    plt.xlabel("Density")
    plt.ylabel("Speedup")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fit_speedup_curve.png", dpi=200)

def main():
    torch.set_float32_matmul_precision("high")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"SDPA backend preference: {SDP_BACKEND.name}")

    results = []  # ← 新增结果收集
    header = f"{'B':>4} {'H':>3} {'L':>6} {'E':>5} {'blk':>4} {'tile_d':>7} | {'Flex(ms)':>9} {'SDPA(ms)':>9} {'speedup':>8} {'winner':>7}"
    print(header); print("-"*len(header))

    with sdpa_kernel(SDP_BACKEND):
        for (B, H, L, E) in GRID:
            q, k, v = make_qkv(B, H, L, E)
            bias = make_bias(B, H, L)
            for blk in BLOCK_SIZES:
                for dens in DENSITIES:
                    tile_d, block_mask = (1.0, None)
                    if dens < 0.999:
                        tile_keep = build_tile_keep(B,H,L,dens,blk,style=MASK_STYLE)
                        tile_d = float(tile_keep.float().mean().item())
                        block_mask = make_block_mask_from_tilekeep(tile_keep, blk, L)

                    t_flex = time_op(flex_call, q,k,v,bias,block_mask)
                    t_sdpa = time_op(sdpa_call, q,k,v,bias)
                    spdup = (t_sdpa / t_flex) if t_flex>0 else float('inf')
                    winner = "flex" if t_flex < t_sdpa else "sdpa"
                    results.append(dict(
                        B=B, H=H, L=L, E=E, blk=blk,
                        density_nominal=dens,
                        density_actual=tile_d,
                        t_flex=t_flex, t_sdpa=t_sdpa,
                        speedup=spdup, winner=winner
                    ))
                    print(f"{B:4d} {H:3d} {L:6d} {E:5d} {blk:4d} {tile_d:7.4f} | {t_flex:9.3f} {t_sdpa:9.3f} {spdup:8.3f} {winner:>7}")

    # === 保存与分析 ===
    df = pd.DataFrame(results)
    df.to_csv("results_flex_vs_sdpa.csv", index=False)
    analyze_results(df)

if __name__ == "__main__":
    main()
    # Profiler 代码保留但禁用
