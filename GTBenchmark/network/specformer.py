import torch
import torch.nn as nn
from GTBenchmark.layer.spec_layer import SpecLayer
from GTBenchmark.encoder.sine_encoder import SineEncoder
from GTBenchmark.network.utils import FeatureEncoder

from torch_geometric.nn import Sequential
import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.models.layer import  MLP, GCNConv, Linear
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_network


class swapex(nn.Module):
    """
    Swaps the x and EigVals attributes of the input batch. 

    Input: 
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.EigVals (torch.Tensor): Eigenvalues of the graph Laplacian.

    Output:
        batch (torch_geometric.data.Batch): Output batch with swapped x and EigVals.
            
    """
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        batch.x, batch.EigVals = batch.EigVals, batch.x
        return batch
    

@register_network("SpecFormer")
class SpecFormer(nn.Module):
    """
    SpecFormer model. Adapted from https://github.com/DSL-Lab/Specformer
    Only supports the case where the input is a batch of graphs with the same number of nodes.
    Needs preprocessing for LapRaw positional encoding.

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.dim_hidden: Hidden dimension for GNN layers and SpecFormer layers.
            - cfg.gt.n_heads: Number of attention heads.
            - cfg.gt.dropout: Dropout rate for the model.
            - cfg.gt.attn_dropout: Dropout rate for the attention mechanism.
            - cfg.gnn.head: Type of head to use for the final output layer.
    
    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.edge_index (torch.Tensor): Edge indices of the graph.
            - batch.EigVals (torch.Tensor): Eigenvalues of the graph Laplacian.
            - batch.EigVecs (torch.Tensor): Eigenvectors of the graph Laplacian.
    
    Output:
        batch (task dependent type, see output head): Output after model processing.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pre_mp = MLP(dim_in = dim_in, dim_out = cfg.gt.dim_hidden, num_layers = 2, has_act = True, has_bias = True)

        ### dirty hack
        self.swap1 = swapex()
        ### eig v
        self.eig_encoder = SineEncoder(cfg.gt.dim_hidden)

        self.mha_eig=Sequential('x',[
            (LayerNorm(cfg.gt.dim_hidden), 'x -> x1'), 
            (MultiHeadAttention(dim_hidden = cfg.gt.dim_hidden, n_heads = cfg.gt.n_heads, dropout = cfg.gt.attn_dropout), 'x1 -> x1'),
            (lambda x1, x2: self.aggregate_batches_add(x1, x2), 'x, x1 -> x')
        ])

        self.ffn_eig=Sequential('x',[
            (LayerNorm(cfg.gt.dim_hidden), 'x -> x1'), 
            (MLP(dim_in = cfg.gt.dim_hidden, dim_out = cfg.gt.dim_hidden, num_layers = 2, has_act = True, has_bias = True), 'x1-> x1'),
            (lambda x1, x2: self.aggregate_batches_add(x1, x2), 'x, x1 -> x')
        ])

        self.decoder=Linear(dim_in = cfg.gt.dim_hidden, dim_out = cfg.gt.n_heads, num_layers = 0, has_act = True, has_bias = True)
        ### eig ^
        ### dirty hack
        self.swap2 = swapex()

        if cfg.gt.layer_norm:
            norm = 'layer'
        elif cfg.gt.batch_norm:
            norm = 'batch'
        else:
            norm = 'none'
        
        self.spec_layers = SpecLayer(dim_out = cfg.gt.dim_hidden, n_heads = cfg.gt.n_heads, dropout = cfg.gt.dropout, norm = norm)
        
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def aggregate_batches_add(self, x1, x2):
        new_batch = x1.clone()
        new_batch.x = x1.x + x2.x
        return new_batch

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch



class LayerNorm(nn.Module):
	"""
	Layer normalization module for PyTorch Geometric.
	Applies layer normalization to the input node features.

	Parameters:
		normalized_shape (int or tuple): Shape of the input features to be normalized.
		Can be a single integer or a tuple of integers.
	
	Input:
		batch.x (Tensor): Input node features.
	
	Output:
		batch.x (Tensor): Output node features after applying layer normalization.
	"""
	def __init__(self, normalized_shape):
		super(LayerNorm, self).__init__()
		self.layer_norm = nn.LayerNorm(normalized_shape)

	def forward(self, batch):
		batch.x = self.layer_norm(batch.x)
		return batch
     
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer wrapper for PyTorch Geometric.

    Parameters:
        dim_hidden (int): Number of input features.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    
    Input:
        batch.x (Tensor): Input node features.
    
    Output:
        batch.x (Tensor): Output node features after applying the Multi-Head Attention layer.
    """

    def __init__(self, dim_hidden: int, n_heads: int, dropout: float, **kwargs):
        super().__init__()
        self.model = nn.MultiheadAttention(
            dim_hidden,
            n_heads,
            dropout,
        )

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch, batch, batch, need_weights=False)[0]
        else:
            batch.x = self.model(batch.x, batch.x, batch.x, need_weights=False)[0]
        return batch