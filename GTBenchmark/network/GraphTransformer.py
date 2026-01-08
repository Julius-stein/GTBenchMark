import torch
import torch.nn as nn
import torch.nn.functional as F

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_network
from GTBenchmark.network.utils import FeatureEncoder


@register_network('GTModel')
class GTModel(nn.Module):
    """
    Minimal Graph Transformer Model
    - Pure GraphTransformer layers
    - No GNN
    - No Mask
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.dim_h = cfg.gt.dim_hidden
        self.l2_norm = getattr(cfg.gt, "l2_norm", False)

        # ---- Encoder ----
        self.encoder = FeatureEncoder()

        # ---- Graph Transformer stack ----
        GTLayer = register.layer_dict["GraphTransformerLayer"]
        self.layers = nn.ModuleList(
            [GTLayer(dim_h=self.dim_h) for _ in range(cfg.gt.layers)]
        )

        # ---- Head ----
        GNNHead = register.head_dict[cfg.gt.head]
        self.head = GNNHead(self.dim_h, dim_out)

    def forward(self, batch):
        # Encode node / edge features if any
        batch = self.encoder(batch)

        # Transformer stack
        for layer in self.layers:
            batch = layer(batch,batch.key_padding_mask())

        # Optional normalization
        if self.l2_norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)

        return self.head(batch)
