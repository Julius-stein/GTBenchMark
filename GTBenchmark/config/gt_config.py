from GTBenchmark.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_gt')
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Prediction head. Use cfg.dataset.task by default
    cfg.gt.head = 'default'
    
    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'GraphTransformerLayer'

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout for input.
    cfg.gt.input_dropout = 0.0

    # norm-first or traditional Transofrmer post LN
    cfg.gt.prepend_norm = True

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    cfg.gt.l2_norm = False

    cfg.gt.residual = True

    cfg.gt.virtual_nodes = 0

    #Attention type.
    cfg.gt.attn_type = 'TorchFullAttention'

    cfg.gt.node_encoder_list = []
    cfg.gt.edge_encoder_list = []
    
    cfg.gt.act = 'relu'

    cfg.gt.activation_dropout =0.0

    cfg.gt.encoder_type = 'concatenate'

    cfg.gt.use_extra_loss = False
    # BigBird model/GPS-BigBird layer.
    cfg.gt.bigbird = CN()

    cfg.gt.bigbird.attention_type = "block_sparse"

    cfg.gt.bigbird.chunk_size_feed_forward = 0

    cfg.gt.bigbird.is_decoder = False

    cfg.gt.bigbird.add_cross_attention = False

    cfg.gt.bigbird.hidden_act = "relu"

    cfg.gt.bigbird.max_position_embeddings = 128

    cfg.gt.bigbird.use_bias = False

    cfg.gt.bigbird.num_random_blocks = 3

    cfg.gt.bigbird.block_size = 3

    cfg.gt.bigbird.layer_norm_eps = 1e-6

    cfg.gt.ffn_dim = 0
    cfg.gt.layers_post_gt = 1

    # cfg.gt.use_graph_token = False

    # NodeFormer
    cfg.gt.edge_loss_weight= 0.1
    cfg.gt.graph_weight= 0.8
    cfg.gt.kernel_trans= "softmax"
    cfg.gt.nb_random_features= 30
    cfg.gt.rb_order= 1
    cfg.gt.rb_trans= "sigmoid"
    cfg.gt.use_act= True
    cfg.gt.use_edge_loss= True
    cfg.gt.use_gumbel= True
    cfg.gt.use_jk = False
    cfg.gt.nb_gumbel_sample = 10
    cfg.gt.projection_matrix_type = 'a'

    cfg.gt.tau = 1.0   # also used by CoBFormer

    # CoBformer
    cfg.gt.use_patch_attn = True
    cfg.gt.alpha = 0.1   # also used by Difformer, cannot be 0
    
    # DIFFormer
    cfg.gt.kernel = "simple"
    cfg.gt.use_source = True

    # SGFormer
    cfg.gt.use_weight = True

    cfg.gt.use_residual = True

    cfg.gt.use_act = True

    cfg.gt.use_graph = True

    cfg.gt.graph_weight = 0.8

    cfg.gt.aggregate = "add"

    # DeGTA
    cfg.gt.K = 4