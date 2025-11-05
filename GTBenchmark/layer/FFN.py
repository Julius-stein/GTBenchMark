import torch
import torch.nn as nn
import torch.nn.functional as F
import GTBenchmark.graphgym.register as register 
from GTBenchmark.graphgym.register import register_layer 
from GTBenchmark.graphgym.config import cfg 

class FFN_block(nn.Module):
    def __init__(self,dim_h) -> None:
        super().__init__()
        self.p_act = float(getattr(cfg.gt, "activation_dropout", 0.0))
        ffn_dim = (cfg.gt.ffn_dim if cfg.gt.ffn_dim != 0 else dim_h * 2)
        self.fc1 = nn.Linear(dim_h, ffn_dim)
        self.act = register.act_dict[cfg.gt.act]
        self.drop_act = nn.Dropout(self.p_act) if self.p_act > 0 else nn.Identity()
        self.fc2 = nn.Linear(ffn_dim, dim_h)

    def forward(self,h):
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop_act(h)
        h = self.fc2(h)
        return h