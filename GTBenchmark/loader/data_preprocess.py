import torch
import copy
from torch_geometric.data import Data


def hop2token_data(data: Data, K: int) -> Data:
    """
    Hop2Token on a single graph Data object.

    - No adjacency normalization
    - Dense adjacency matmul
    - Output x shape: (N, K+1, d)
    """

    # ---- shallow copy: keep other fields untouched ----
    data = copy.copy(data)

    x = data.x                    # (N, d)
    edge_index = data.edge_index  # (2, E)
    device = x.device
    N, d = x.size()

    # ---- 1. dense adjacency ----
    adj = torch.zeros((N, N), dtype=x.dtype, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0

    # ---- 2. hop aggregation ----
    hop_feat = torch.zeros((N, K + 1, d), dtype=x.dtype, device=device)
    hop_feat[:, 0, :] = x

    x_prop = x
    for h in range(1, K + 1):
        x_prop = adj @ x_prop      # no normalization
        hop_feat[:, h, :] = x_prop

    # ---- 3. write back ----
    data.x = hop_feat             # (N, K+1, d)

    # 注意：不要在这里乱改 num_node_features
    # GraphGym / encoder 会自己从 x.shape 读

    return data
