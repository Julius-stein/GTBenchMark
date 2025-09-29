# readouthead_regression.py
import torch
import torch.nn as nn
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_head
from GTBenchmark.transform.graph2dense import to_dense_batch

def graph_token_pooling(x, batch):
    # x: (N,C), batch: (N,)
    x, _, _ = to_dense_batch(x, batch)  # -> (B, N', C)
    return x[:, 0, :]                   # 取第 0 个（graph token）

def masked_mean_pooling(x, batch):
    # x: (N,C), batch: (N,)
    x_dense, mask, _ = to_dense_batch(x, batch)           # (B, N_max, C), (B, N_max)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x_dense.dtype)  # (B,1)
    return (x_dense * mask.unsqueeze(-1)).sum(dim=1) / denom  

@register_head('reg_graph')
class GraphormerRegressionHead(nn.Module):
    """
    Graphormer 回归头：图级单标量回归（ZINC）
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        assert dim_out == 1, "ZINC 是单标量回归，请把 dim_out 设为 1"
        self.pool = graph_token_pooling

        # 官方常见：LN + Linear；你也可以加一层 MLP（LN->Linear->GELU->Dropout->Linear）
        self.ln = nn.LayerNorm(dim_in)
        self.fc = nn.Linear(dim_in, 1)

        # （可选）训练时用到标签标准化
        self.register_buffer('y_mean', torch.tensor(0.0), persistent=False)
        self.register_buffer('y_std', torch.tensor(1.0), persistent=False)
        self.use_label_norm = False

    def set_label_norm(self, mean, std):
        self.y_mean.fill_(float(mean))
        self.y_std.fill_(float(std) if std != 0 else 1.0)
        self.use_label_norm = True

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        x = self.ln(batch.x)
        if cfg.posenc_GraphormerBias.use_graph_token:
            g = self.pool(x, batch.batch)   # (B, C)
        else:
             g = masked_mean_pooling(x, batch.batch)
        pred = self.fc(g)               # (B, 1)

        # 反标准化（若训练时对 y 做了标准化，这里把输出还原）
        if self.use_label_norm:
            pred = pred * self.y_std + self.y_mean

        batch.graph_feature = pred
        pred, label = self._apply_index(batch)
        return pred, label
