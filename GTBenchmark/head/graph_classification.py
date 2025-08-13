import torch

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_head
from GTBenchmark.transform.graph2dense import to_dense_batch


@register_head('class_graph')
class GraphormerHead(torch.nn.Module):
    """
    Graphormer prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        print(f"Initializing {cfg.model.graph_pooling} pooling function")
        self.pooling_fun = graph_token_pooling

        self.ln = torch.nn.LayerNorm(dim_in)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out)
        )

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        x = self.ln(batch.x)
        graph_emb = self.pooling_fun(x, batch.batch)
        graph_emb = self.layers(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

def graph_token_pooling(x, batch, *args):
    """Extracts the graph token from a batch to perform graph-level prediction.
    Typically used together with Graphormer when GraphormerEncoder is used and
    the global graph token is used: `cfg.graphormer.use_graph_token == True`.
    """
    x, _,_ = to_dense_batch(x, batch)
    return x[:, 0, :]