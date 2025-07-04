import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch


"""
    Graph Transformer
    
"""

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False, is_sparse=False):
        super().__init__()

        self.out_channels = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        self.is_sparse = is_sparse
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=use_bias, batch_first=True)
        

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(embed_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(embed_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(embed_dim, embed_dim*2)
        self.FFN_layer2 = nn.Linear(embed_dim*2, embed_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(embed_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(embed_dim)
        
    def forward(self, x, edge_index, batch=None):
        h_in1 = x # for first residual connection
        h, mask = to_dense_batch(x, batch)
        print(batch)
        print(mask)
        
        # multi-head attention out
        if self.is_sparse:
            attn_mask = ~(to_dense_adj(edge_index, batch).bool())
            #对角线置1否则softmax会出nan
            attn_mask[:,torch.arange(attn_mask.shape[-1]),torch.arange(attn_mask.shape[-1])] = False
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        else:
            attn_mask = None
        attn_out, _ = self.attention(h, h, h, need_weights=False, key_padding_mask=~mask, attn_mask=attn_mask)
        h = attn_out[mask]
        
        #h = attn_out.view(-1, self.out_channels)
        
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.out_channels, self.num_heads, self.residual)
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
    
class GraphTransformerNet(nn.Module):

    def __init__(
            self,
            n_layers,
            num_heads,
            input_dim,
            hidden_dim,
            out_dim,
            dropout,
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
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, num_heads,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers)])
        self.MLP_layer = MLPReadout(hidden_dim, out_dim)


    def forward(self, batch):

        # input embedding
        h = batch.x

        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = batch.lap_pos_enc
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = batch.wl_pos_enc.squeeze(1)
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




