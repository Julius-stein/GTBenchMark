import torch.nn as nn
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.models.layer import MLP
from GTBenchmark.graphgym.register import register_head


@register_head('only_mask')
class OnlyMaskNodeHead(nn.Module):
    """
    GNN prediction head for mini node prediction tasks, select the batchsize node for pred.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(OnlyMaskNodeHead, self).__init__()

    def _apply_index(self, batch):
        bs = batch.batch_size
        return batch.x[:bs], batch.y[:bs]

        # mask = '{}_mask'.format(batch.split)
        # return batch.x[batch[mask]], batch.y[batch[mask]]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label
