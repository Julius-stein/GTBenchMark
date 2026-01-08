import logging
from torch_geometric.utils import remove_self_loops, add_self_loops
import copy

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from torch_sparse import SparseTensor


def pre_transform_in_memory(dataset, transform_func, show_progress=False,side=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset
    data_list=[]
    for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20):
        data = transform_func(dataset.get(i))
        data.sample_idx = torch.tensor([i])
        data_list.append(data) 

    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data

def move_node_feat_to_x(data):
    """For ogbn-proteins, move the attribute node_species to attribute x."""
    species_raw = data.node_species.squeeze(-1).long()  # [num_nodes]
    # 找出唯一的物种 ID 并建立映射表
    unique_species = torch.unique(species_raw)
    id_map = {int(v.item()): i for i, v in enumerate(unique_species)}
    # 将每个节点的原始ID映射为连续索引
    species_idx = torch.tensor(
        [id_map[int(v.item())] for v in species_raw],
        dtype=torch.long
    )
    # 写回到 data.x，并附带 num_species
    data.x = species_idx.unsqueeze(-1)  # [num_nodes, 1]
    data.pop("node_species")

    return data

def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data

def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t

def add_self_loops_data(data):
    """
    Add self-loops to a single PyG Data object.

    - Removes existing self-loops first
    - Preserves edge_attr semantics
    """
    data = copy.copy(data)

    edge_index = data.edge_index
    edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    edge_index, edge_attr = add_self_loops(
        edge_index,
        edge_attr=edge_attr,
        num_nodes=data.num_nodes
    )

    data.edge_index = edge_index
    data.edge_attr = edge_attr  # edge_attr can be None, that's OK

    return data