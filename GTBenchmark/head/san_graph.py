import torch.nn as nn

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_head


@register_head('san_graph')
class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch, size=batch.num_graphs)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label



@register_head('san_graph_df')
class SANGraphHeadDF(nn.Module):
    """
    SAN prediction head for graph-level tasks (DenseFirst version).

    Expected batch fields:
        batch.x         : [B, M, F]
        batch.num_nodes : [B]
        batch.y         : [B, ...] or None
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()

        # --------------------------------------------------
        # Pooling (DenseFirst)
        # --------------------------------------------------
        # cfg.model.graph_pooling should be: add_df / mean_df / max_df
        self.pooling_fun = register.pooling_dict["add_df"]

        # --------------------------------------------------
        # MLP head
        # --------------------------------------------------
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True)
        )

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, batch):
        """
        batch.x         : [B, M, F]
        batch.num_nodes : [B]
        """

        # ---- DenseFirst graph pooling ----
        graph_emb = self.pooling_fun(batch.x, batch.num_nodes)  # [B, F]

        # ---- MLP head ----
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)

        graph_emb = self.FC_layers[self.L](graph_emb)

        # ---- attach & return (GraphGym convention) ----
        batch.graph_feature = graph_emb
        return graph_emb, batch.y
