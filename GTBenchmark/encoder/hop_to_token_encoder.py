import torch
import torch.nn as nn
from GTBenchmark.graphgym.register import register_node_encoder

@register_node_encoder('HopToTokenEncoder')
class HopToTokenEncoder(nn.Module):
    """
    Propagate node features for K steps and record features at each step.

    After propagation, the node features will have shape (N, K+1, d).

    Args:
        K (int): Number of propagation steps.
    """
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, batch):
        """
        Args:
            batch: PyG Batch or Data object with fields:
                - batch.x: (N, d) node features
                - batch.edge_index: (2, E) edge index (used to build adj matrix)

        Returns:
            batch: with updated batch.x of shape (N, K+1, d)
        """
        x = batch.x  # (N, d)
        edge_index = batch.edge_index  # (2, E)
        N = x.size(0)
        d = x.size(1)

        # Build dense adjacency matrix
        adj = torch.zeros(N, N, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Initialize feature storage
        nodes_features = torch.empty(N, 1, self.K + 1, d, device=x.device)

        # Step 0: original features
        nodes_features[:, 0, 0, :] = x

        # Make a copy of x to propagate
        x_propagate = x.clone()

        for i in range(self.K):
            # Propagate one step
            x_propagate = torch.matmul(adj, x_propagate)
            nodes_features[:, 0, i + 1, :] = x_propagate

        # Remove the extra dimension
        nodes_features = nodes_features.squeeze(1)  # (N, K+1, d)

        # Update batch.x
        batch.x = nodes_features

        return batch

