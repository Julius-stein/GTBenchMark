import torch.nn as nn
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.models.layer import MLP
from GTBenchmark.graphgym.register import register_head


@register_head('cluster_node')
class GNNClusterNodeHead(nn.Module):
    """
    GNN prediction head for mini node prediction tasks, select the batchsize node for pred.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNClusterNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt))
                             #has_act=False, has_bias=True)

    def _apply_index(self, batch):
        global_nid = batch.n_id  # 映射回原图索引
        split = batch.split      # e.g., "train" / "val" / "test"
        mask = getattr(self.data, f"{split}_mask")[global_nid]  # 对应 batch 中哪些节点属于该 split

        return batch.x[mask], batch.y[mask]


        # mask = '{}_mask'.format(batch.split)
        # return batch.x[batch[mask]], batch.y[batch[mask]]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label
