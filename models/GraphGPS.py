import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn import GCNConv
from torch_geometric.nn.attention import PerformerAttention
from models.Exphormer import ExphormerFullLayer
from models.MHGA import MultiHeadGraphAttentionLayer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch


class GPSConv(torch.nn.Module):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    The GPS layer is based on a 3-part recipe:

    1. Inclusion of positional (PE) and structural encodings (SE) to the input
       features (done in a pre-processing step via
       :class:`torch_geometric.transforms`).
    2. A local message passing layer (MPNN) that operates on the input graph.
    3. A global attention layer that operates on the entire graph.

    .. note::

        For an example of using :class:`GPSConv`, see
        `examples/graph_gps.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graph_gps.py>`_.

    Args:
        channels (int): Size of each input sample.
        conv (MessagePassing, optional): The local message passing layer.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability of intermediate
            embeddings. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        attn_type (str): Global attention type, :obj:`multihead` or
            :obj:`performer`. (default: :obj:`multihead`)
        attn_kwargs (Dict[str, Any], optional): Arguments passed to the
            attention layer. (default: :obj:`None`)
    """
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        layer_norm: bool = True,
        batch_norm: bool = False,
        attn_type: str = 'multihead',
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
            )
        elif attn_type == 'exphormer':
            self.attn = ExphormerFullLayer(
                channels, channels, num_heads=heads, dropout=dropout, attn_dropout=attn_dropout,
                layer_norm=layer_norm, batch_norm=batch_norm, activation=act,
            )
        elif attn_type == 'graphattn':
            self.attn = MultiHeadGraphAttentionLayer(
                channels, channels, n_heads=heads, dropout=attn_dropout, bias=True
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )
        self.norm1 = self.norm2 = self.norm3 = None

        if layer_norm:
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
            self.norm3 = nn.LayerNorm(channels)
            
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(channels)
            self.norm2 = nn.BatchNorm1d(channels)
            self.norm3 = nn.BatchNorm1d(channels)
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, mask = to_dense_batch(x, batch)
            h, _ = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=False)
            h = h[mask]
        elif isinstance(self.attn, PerformerAttention):
            h, mask = to_dense_batch(x, batch)
            h = self.attn(h, mask=mask)
            h = h[mask]
        elif isinstance(self.attn, ExphormerFullLayer):
            h = self.attn(h, edge_index)
        elif isinstance(self.attn, MultiHeadGraphAttentionLayer):
            h,_ = self.attn(h, edge_index)
            
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
    
class GPSModel(torch.nn.Module):
    def __init__(
            self,
            n_layers,
            num_heads,
            input_dim,
            hidden_dim,
            out_dim,
            dropout,
            attn_dropout,
            in_feat_dropout,
            net_params
        ):
        super().__init__()

        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.attn_type = net_params['attn_type']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim), num_heads,
                                              dropout, attn_dropout, 'relu', self.layer_norm, self.batch_norm, self.attn_type) for _ in range(n_layers)])
        self.MLP_layer = MLPReadout(hidden_dim, out_dim)


    def forward(self, batch):

        # input embedding
        h = batch.x
        h_lap_pos_enc = batch.lap_pos_enc
        h_wl_pos_enc = batch.wl_pos_enc.squeeze(1)

        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(h, batch.edge_index, batch.batch)
            
        # output
        h_out, mask = to_dense_batch(h, batch.batch)
        h_out = h_out[:,0,:]
        h_out = self.MLP_layer(h_out)

        return F.log_softmax(h_out)

class FullGraphGPSModel(torch.nn.Module):
    def __init__(
            self,
            n_layers,
            num_heads,
            input_dim,
            hidden_dim,
            out_dim,
            dropout,
            attn_dropout,
            in_feat_dropout,
            net_params
        ):
        super().__init__()

        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.attn_type = net_params['attn_type']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim), num_heads,
                                              dropout, attn_dropout, 'relu', self.layer_norm, self.batch_norm, self.attn_type) for _ in range(n_layers)])
        self.MLP_layer = MLPReadout(hidden_dim, out_dim)


    def forward(self, batch):

        # input embedding
        h = batch.x
        h_lap_pos_enc = batch.lap_pos_enc
        h_wl_pos_enc = batch.wl_pos_enc.squeeze(1)

        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(h, batch.edge_index, batch.batch)
            
        # output
        h_out = self.MLP_layer(h)

        return F.log_softmax(h_out)

