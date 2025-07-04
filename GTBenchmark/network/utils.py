import torch
import torch.nn as nn
import torch.nn.functional as F

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.models.layer import BatchNorm1dNode, BatchNorm1dEdge



class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features
    """
    def __init__(self):
        super(FeatureEncoder, self).__init__()


        # merge several encoders
        Encoder = []
        for name in cfg.gt.node_encoder_list:
            Encoder.append(register.node_encoder_dict[name]())
        for name in cfg.gt.edge_encoder_list:
            Encoder.append(register.edge_encoder_dict[name]())
        self.encoder = nn.Sequential(*Encoder)

    def forward(self, batch):
        return self.encoder(batch)
