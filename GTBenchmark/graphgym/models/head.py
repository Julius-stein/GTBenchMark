""" GNN heads are the last layer of a GNN right before loss computation.

They are constructed in the init function of the gnn.GNN.
"""

import torch
import torch.nn as nn

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.models.layer import MLP
from GTBenchmark.graphgym.models.pooling import pooling_dict


# Head
class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''
    def __init__(self, dim_in, dim_out):
        super(GNNNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)

    def _apply_index(self, batch):
        mask = '{}_mask'.format(batch.split)
        return batch.x[batch[mask]], batch.y[batch[mask]]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


class GNNEdgeHead(nn.Module):
    '''Head of GNN, edge prediction'''
    def __init__(self, dim_in, dim_out):
        ''' Head of Edge and link prediction models.

        Args:
            dim_out: output dimension. For binary prediction, dim_out=1.
        '''
        # Use dim_in for graph conv, since link prediction dim_out could be
        # binary
        # E.g. if decoder='dot', link probability is dot product between
        # node embeddings, of dimension dim_in
        super(GNNEdgeHead, self).__init__()
        # module to decode edges from node embeddings

        if cfg.model.edge_decoding == 'concat':
            self.layer_post_mp = MLP(dim_in * 2,
                                     dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(
                    'Binary edge decoding ({})is used for multi-class '
                    'edge/link prediction.'.format(cfg.model.edge_decoding))
            self.layer_post_mp = MLP(dim_in,
                                     dim_in,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif cfg.model.edge_decoding == 'cosine_similarity':
                self.decode_module = nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError('Unknown edge decoding {}.'.format(
                    cfg.model.edge_decoding))

    def _apply_index(self, batch):
        index = '{}_edge_index'.format(batch.split)
        label = '{}_edge_label'.format(batch.split)
        return batch.x[batch[index]], batch[label]

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat':
            batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        nodes_first = pred[0]
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)
        return pred, label


class GNNGraphHead(nn.Module):
    '''Head of GNN, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out):
        super(GNNGraphHead, self).__init__()
        # todo: PostMP before or after global pooling
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)
        self.pooling_fun = pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


# Head models for external interface
head_dict = {
    'node': GNNNodeHead,
    'edge': GNNEdgeHead,
    'link_pred': GNNEdgeHead,
    'graph': GNNGraphHead
}

register.head_dict = {**register.head_dict, **head_dict}
