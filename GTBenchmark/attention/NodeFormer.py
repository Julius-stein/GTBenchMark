import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from GTBenchmark.graphgym.config import cfg
from contextlib import nullcontext
from typing import Optional

def numerator(qs, ks, vs):
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)

def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

def numerator_gumbel(qs, ks, vs):
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)

def denominator_gumbel(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)

def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None,
                                K=10, tau=0.25, return_weight=True):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size, K = number of Gumbel sampling
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    gumbels = (
        -torch.empty(key_prime.shape[:-1]+(K, ), memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).to(query.device) / tau # [N, B, H, K]
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4) # [N, B, H, K, M]
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value) # [N, B, H, K, D]
    z_den = denominator_gumbel(query_prime, key_t_gumbel) # [N, B, H, K]

    z_num = z_num.permute(1, 0, 2, 3, 4) # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3) # [B, N, H, K]
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = torch.mean(z_num / z_den, dim=3) # [B, N, H, D]

    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

        return z_output, A_weight

    else:
        return z_output

def _to_additive_from_full_mask(full_mask: Tensor, Np: int) -> Tensor:
    """
    full_mask: [B,1,N,N]，bool(True=keep) 或 float(0/-inf)
    返回：     [B,1,Np,Np] 的 additive 浮点掩码（与 graph token 对齐）
    """
    assert full_mask.dim() == 4
    B, _, N, _ = full_mask.shape
    if full_mask.dtype == torch.bool:
        add = torch.zeros_like(full_mask, dtype=torch.float, device=full_mask.device)
        add = add.masked_fill(~full_mask, float('-inf'))
    else:
        add = full_mask if full_mask.device else full_mask
    if Np == N + 1:
        add = F.pad(add, (1, 0, 1, 0), value=0.0)
    elif Np != N:
        raise RuntimeError(f"Padding mask size mismatch: full N={N}, attn N'={Np}")
    return add  # [B,1,Np,Np]