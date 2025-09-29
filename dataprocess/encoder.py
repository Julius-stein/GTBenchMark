import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import scipy.sparse as sp
from numpy.linalg import inv
# from pygsp import graphs
from torch.nn.functional import normalize
import torch_geometric.transforms as T
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.utils import degree, to_dense_adj
from torch_sparse import coalesce
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
import hashlib
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GATConv, MessagePassing
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj
import time

def laplacian_positional_encoding(data, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                shape=(data.y.shape[0], data.y.shape[0]),
                                dtype=np.float32)
    N = sp.diags(degree(data.edge_index[1],num_nodes=data.num_nodes).numpy().clip(1) ** -0.5, dtype=float)
    L = sp.eye(data.num_nodes) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    k = pos_enc_dim+1
    if k >= data.num_nodes - 1:
        L = L.toarray()
    # print(L)
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # 尝试执行的操作
            EigVal, EigVec = sp.linalg.eigs(L, k=k, which='SR', tol=1e-2) # for 40 PEs
            break
        except Exception as e:  # 捕获特定的异常类型
            print('failed')
            if attempt == max_retries - 1:
                print(data)
                torch.save(data, './data/error.pt')
                print("Max retries reached. Operation failed.")
                raise e # 如果达到最大重试次数，抛出异常
    
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    data.lap_pos_enc = torch.nested.nested_tensor([torch.from_numpy(EigVec[:,1:k]).float()]).to_padded_tensor(0,output_size=(1,data.num_nodes,pos_enc_dim)).squeeze(0)

    return data


def wl_positional_encoding(data):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    torch.nn.Softmax

    edge_list = data.edge_index.t().numpy()
    node_list = torch.arange(data.num_nodes).numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    data.wl_pos_enc = torch.LongTensor(list(node_color_dict.values())).unsqueeze(1)
    return data

