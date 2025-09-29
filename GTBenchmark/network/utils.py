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

        dim_emb = cfg.gt.dim_hidden
        # merge several encoders
        pe_enabled_list = []
        dim_out = [dim_emb]
        tmp = dim_emb
        for key, pecfg in cfg.items():
            if key.startswith('posenc_') and pecfg.enable:
                pe_name = key.split('_', 1)[1]
                pe_enabled_list.append(pe_name)
                if hasattr(pecfg, 'kernel'):
                    # Generate kernel times if functional snippet is set.
                    if pecfg.kernel.times_func:
                        pecfg.kernel.times = list(eval(pecfg.kernel.times_func))

                dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
                tmp-=dim_pe
                dim_out.append(tmp)
        Encoder = []
        if cfg.gt.encoder_type == "concatenate":
            for name in cfg.gt.node_encoder_list:
                Encoder.append(register.node_encoder_dict[name](dim_out.pop()))
            for name in cfg.gt.edge_encoder_list:
                Encoder.append(register.edge_encoder_dict[name](dim_emb))
            self.encoder = nn.Sequential(*Encoder)

        elif cfg.gt.encoder_type == "cascade":
            for name in cfg.gt.node_encoder_list:
                Encoder.append(register.node_encoder_dict[name](dim_emb))
            for name in cfg.gt.edge_encoder_list:
                Encoder.append(register.edge_encoder_dict[name](dim_emb))
            self.encoder = nn.Sequential(*Encoder)
            

    def forward(self, batch):
        return self.encoder(batch)
