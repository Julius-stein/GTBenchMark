import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

B = 1; H = 1; L = 4; D = 8
q = torch.randn(B, H, L, D)
k = torch.randn(B, H, L, D)
v = torch.randn(B, H, L, D)

# 允许自己注意自己和相邻（构造一个“允许矩阵”）
allow = torch.tensor([
 [1,1,0,0],
 [1,1,1,0],
 [0,1,1,1],
 [0,0,1,1]
], dtype=torch.bool).unsqueeze(0)  # [1,4,4]

# 转成 SDPA 需要的 bool mask（True=屏蔽）
attn_mask = ~allow                 # [1,4,4]
attn_mask = attn_mask.unsqueeze(1) # [1,1,4,4]

out = sdpa(q, k, v, attn_mask=attn_mask)
