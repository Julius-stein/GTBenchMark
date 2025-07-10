import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

# —— 环境检查 —— 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda', "需在支持 CUDA 的 GPU 上运行"

# —— 测试参数 —— 
batch_size, num_heads, seq_len, head_dim = 16, 8, 2048, 64
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# —— 基准配置 —— 
warmup_iters = 5    # 预热次数
measure_iters = 20  # 计时次数

_backend_names = {
    "cudnn":         SDPBackend.CUDNN_ATTENTION,
    "flash":         SDPBackend.FLASH_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math":          SDPBackend.MATH,
}

results = []

def benchmark_backend(name, backend):
    # 清理显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 1) 预热调用
    with sdpa_kernel(backend):
        for _ in range(warmup_iters):
            _ = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
        torch.cuda.synchronize()
    
    # 2) 正式测量
    torch.cuda.reset_peak_memory_stats()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    with sdpa_kernel(backend):
        for _ in range(measure_iters):
            start_evt.record()
            _ = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
            end_evt.record()
            torch.cuda.synchronize()
            total_time += start_evt.elapsed_time(end_evt)
    
    avg_time = total_time / measure_iters
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    return avg_time, peak_mem

# 针对各后端基准测试
for name, backend in _backend_names.items():
    t, m = benchmark_backend(name, backend)
    results.append((name, t, m))

# 再测一次默认调度
def benchmark_default():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # 预热
    for _ in range(warmup_iters):
        _ = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
    torch.cuda.synchronize()

    # 测量
    torch.cuda.reset_peak_memory_stats()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    for _ in range(measure_iters):
        start_evt.record()
        _ = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
        end_evt.record()
        torch.cuda.synchronize()
        total_time += start_evt.elapsed_time(end_evt)
    avg_time = total_time / measure_iters
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    return avg_time, peak_mem

t_def, m_def = benchmark_default()
results.append(("default", t_def, m_def))

# 打印结果
print(f"{'Backend':<15} {'Avg Time (ms)':>15}  {'Peak Memory (MB)':>18}")
print("-"*53)
for name, t, m in results:
    print(f"{name:<15} {t:15.2f}  {m:18.2f}")

def test_enable():

    import torch.backends.cuda as cuda
    from torch._C import _SDPAParams as SDPAParams  # 内部类

    # —— 1. 准备张量 —— 
    device = torch.device("cuda")
    # 确保半精度，FlashAttention 只在 float16/bfloat16 下可用
    q = torch.randn(8, 8, 512, 128, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attn_mask = None
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False  # PyTorch 2.6 中的第七个参数：是否启用 GQA

    # —— 2. 构造 SDPAParams —— 
    # 参数顺序：Q, K, V, attn_mask, dropout_p, is_causal, enable_gqa
    params = SDPAParams(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa)

    # —— 3. 依次诊断各后端可用性 —— 
    print("cuDNN available?       ", cuda.can_use_cudnn_attention(params, debug=True))
    print("FlashAttention avail?  ", cuda.can_use_flash_attention(params, debug=True))
    print("EfficientAttention avl?", cuda.can_use_efficient_attention(params, debug=True))

    # —— 4. 查看全局开关状态 —— 
    print("  cudnn_sdp_enabled:", cuda.cudnn_sdp_enabled())
    print("  flash_sdp_enabled:", cuda.flash_sdp_enabled())
    print("  mem_eff_sdp_enbd: ", cuda.mem_efficient_sdp_enabled())
    print("  math_sdp_enabled: ", cuda.math_sdp_enabled())


test_enable()
import torch
import torch.backends.cuda as cuda
from torch._C import _SDPAParams as SDPAParams

def test_enable_for_benchmark():
    device = torch.device("cuda")
    # —— 用跟基准相同的大输入 —— 
    q = torch.randn(16, 8, 2048, 64, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    attn_mask = None
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False  # PyTorch 2.6 的第七个 flag

    # 构造内部参数（位置参数构造）
    params = SDPAParams(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa)

    print("—— 可用性诊断（大输入 16×8×2048×64） ——")
    print("cuDNN available?       ", cuda.can_use_cudnn_attention(params, debug=True))
    print("FlashAttention avail?  ", cuda.can_use_flash_attention(params, debug=True))
    print("EfficientAttention avl?", cuda.can_use_efficient_attention(params, debug=True))

    print("\n—— 全局开关状态 ——")
    print("  cudnn_sdp_enabled:   ", cuda.cudnn_sdp_enabled())
    print("  flash_sdp_enabled:   ", cuda.flash_sdp_enabled())
    print("  mem_eff_sdp_enabled: ", cuda.mem_efficient_sdp_enabled())
    print("  math_sdp_enabled:    ", cuda.math_sdp_enabled())

if __name__ == "__main__":
    test_enable_for_benchmark()
