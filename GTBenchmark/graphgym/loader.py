from typing import Callable

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset)
from torch_geometric.loader import (ClusterLoader, DataLoader,
                                    GraphSAINTEdgeSampler,
                                    GraphSAINTNodeSampler,
                                    GraphSAINTRandomWalkSampler,
                                    NeighborSampler, RandomNodeSampler)
from torch_geometric.utils import (index_to_mask, negative_sampling,
                                   to_undirected)

import GTBenchmark.graphgym.register as register
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.models.transform import create_link_label, neg_sampling_transform
from GTBenchmark.loader.DFDataset import DenseFirstDataset, densefirst_collate
from yacs.config import CfgNode as CN
from torch.utils.data import Subset
from torch.utils.data import DataLoader as TorchDataloader

def planetoid_dataset(name: str) -> Callable:
    return lambda root: Planetoid(root, name)


register.register_dataset('Cora', planetoid_dataset('Cora'))
register.register_dataset('CiteSeer', planetoid_dataset('CiteSeer'))
register.register_dataset('PubMed', planetoid_dataset('PubMed'))
register.register_dataset('PPI', PPI)


def load_pyg(name, dataset_dir):
    """
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset = TUDataset(dataset_dir, name[3:])
    elif name == 'Karate':
        dataset = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset = Coauthor(dataset_dir, name='CS')
        else:
            dataset = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset = Amazon(dataset_dir, name='Computers')
        else:
            dataset = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def load_ogb(name, dataset_dir):
    r"""

    Load OGB dataset objects.


    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset
    from ogb.nodeproppred import PygNodePropPredDataset

    if name[:4] == 'ogbn':
        dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i, key in enumerate(splits.keys()):
            mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_names[i], mask, len(mask))
        edge_index = to_undirected(dataset.data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])

    elif name[:4] == 'ogbg':
        dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name[:4] == "ogbl":
        dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_edge_split()
        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            dataset.transform = neg_sampling_transform
        else:
            id_neg = negative_sampling(edge_index=id,
                                       num_nodes=dataset.data.num_nodes,
                                       num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = create_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')
    return dataset


def load_dataset():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        dataset = func(format, name, dataset_dir)
        if dataset is not None:
            return dataset
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    # Load from OGB formatted data
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(format))
    return dataset

def set_dataset_info(dataset):
    r"""
    Set global dataset information
    """
    # --- dim_in ---
    try:
        if isinstance(dataset.data, HeteroData):  # Hetero graph
            cfg.share.dim_in = CN()
            task = cfg.dataset.task_entity
            try:
                if hasattr(dataset.data, "x_dict"):
                    for node_type in dataset.data.node_types:
                        if node_type in dataset.data.x_dict:
                            cfg.share.dim_in[node_type] = dataset.data.x_dict[node_type].shape[1]
                        else:
                            cfg.share.dim_in[node_type] = None
            except Exception:
                pass  # No x_dict
            try:
                if hasattr(dataset.data, "edge_attr_dict"):
                    for edge_type in dataset.data.edge_types:
                        if edge_type in dataset.data.edge_attr_dict:
                            cfg.share.dim_in["__".join(edge_type)] = dataset.data.edge_attr_dict[edge_type].shape[1]
                        else:
                            cfg.share.dim_in["__".join(edge_type)] = None
            except Exception:
                pass  # No edge_attr_dict
        else:
            cfg.share.dim_in = dataset.data.x.shape[-1]
    except Exception:
        cfg.share.dim_in = 1

    try:
        cfg.share.edge_dim_in = dataset.data.edge_attr.shape[-1]
    except Exception:
        cfg.share.edge_dim_in = 1

    # --- get label tensor y (graph-level or edge-level depending on task) ---
    y = None
    try:
        if isinstance(dataset.data, HeteroData):
            task = cfg.dataset.task_entity
            if hasattr(dataset.data[task], 'y'):
                y = dataset.data[task].y
            elif hasattr(dataset.data[task], 'edge_label'):
                y = dataset.data[task].edge_label
        else:
            if hasattr(dataset.data, 'y'):
                y = dataset.data.y
            elif hasattr(dataset.data, 'edge_label'):
                y = dataset.data.edge_label
    except Exception:
        y = None

    if y is None:
        cfg.share.dim_out = 1
    else:
        # --- normalize shape (ensure tensor) ---
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)

        # --- classification vs regression ---
        if cfg.dataset.task_type == 'classification':
            if y.dim() == 1:  # class ids
                if torch.is_floating_point(y):
                    # Rare case: float class ids? fallback to unique count
                    cfg.share.dim_out = int(torch.unique(y).numel())
                else:
                    cfg.share.dim_out = int(y.max().item()) + 1
            elif y.dim() == 2:
                cfg.share.dim_out = y.size(-1)  # one-hot / multi-label
            else:
                # flatten trailing dims
                cfg.share.dim_out = int(torch.prod(torch.tensor(y.shape[1:])).item())
        else:
            # regression / scores
            if y.dim() == 0:
                cfg.share.dim_out = 1
            elif y.dim() == 1:
                cfg.share.dim_out = 1      # <== KEY FIX
            elif y.dim() == 2:
                cfg.share.dim_out = y.size(-1)
            else:
                cfg.share.dim_out = int(torch.prod(torch.tensor(y.shape[1:])).item())

    # --- count splits ---
    cfg.share.num_splits = 1
    for key in dataset.data.keys():
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in dataset.data.keys():
        if 'test' in key:
            cfg.share.num_splits += 1
            break

    if hasattr(dataset, 'dynamicTemporal'):
        cfg.share.num_splits = len(dataset)
    # 直接索引 dataset[i]
    cfg.share.side = False
    if len(dataset)>1 and cfg.dataset.task=='graph':
        slices = dataset.slices["x"]   # x 的切片位置，比如 [0, 34, 67, ...]
        # 每个图的节点数就是相邻差值
        num_nodes_list = slices[1:] - slices[:-1]
        cfg.share.max_num_nodes = max(num_nodes_list).item()
    #!Graphormer记录度数
    if "TypeDictNode" in cfg.gt.node_encoder_list:
        cfg.share.num_types = int(dataset.data.x.max().item() + 1)
    
    count_degree(dataset)


def set_dataset_info_densefirst(dataset):
    r"""
    Set global dataset information for Dense-First datasets.
    Assumes:
      - dataset.x          : [G, M, F]
      - dataset.num_nodes  : [G]
      - dataset.y          : [G, ...] or None
      - dataset.index_tensors : dict (train/val/test)
    """

    # --------------------------------------------------
    # Input feature dimension
    # --------------------------------------------------
    try:
        cfg.share.dim_in = dataset.x.size(-1)
    except Exception:
        cfg.share.dim_in = 1

    # --------------------------------------------------
    # Edge feature dimension (optional)
    # --------------------------------------------------
    try:
        if dataset.edge_attr is not None:
            cfg.share.edge_dim_in = dataset.edge_attr[0].size(-1)
        else:
            cfg.share.edge_dim_in = 1
    except Exception:
        cfg.share.edge_dim_in = 1

    # --------------------------------------------------
    # Output dimension (y)
    # --------------------------------------------------
    y = getattr(dataset, "y", None)

    if y is None:
        cfg.share.dim_out = 1
    else:
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)

        if cfg.dataset.task_type == "classification":
            if y.dim() == 1:
                if torch.is_floating_point(y):
                    cfg.share.dim_out = int(torch.unique(y).numel())
                else:
                    cfg.share.dim_out = int(y.max().item()) + 1
            elif y.dim() == 2:
                cfg.share.dim_out = y.size(-1)
            else:
                cfg.share.dim_out = int(torch.prod(torch.tensor(y.shape[1:])).item())
        else:  # regression
            if y.dim() <= 1:
                cfg.share.dim_out = 1
            else:
                cfg.share.dim_out = y.size(-1)

    # --------------------------------------------------
    # Split information
    # --------------------------------------------------
    if hasattr(dataset, "index_tensors") and dataset.index_tensors:
        cfg.share.num_splits = len(dataset.index_tensors)
    else:
        cfg.share.num_splits = 1

    # --------------------------------------------------
    # Graph-level info
    # --------------------------------------------------
    cfg.share.side = False  # DenseFirst: no ragged slicing
    # cfg.share.max_num_nodes = int(dataset.num_nodes.max().item())

    # --------------------------------------------------
    # Graphormer / type encoder support (optional)
    # --------------------------------------------------
    if "TypeDictNode" in cfg.gt.node_encoder_list:#！！！
        # assume node type encoded in x[..., 0]
        cfg.share.num_types = 28

    # --------------------------------------------------
    # Degree statistics (if needed)
    # --------------------------------------------------
    try:
        count_degree(dataset)
    except Exception:
        pass


# def set_dataset_info(dataset):
#     r"""
#     Set global dataset information

#     Args:
#         dataset: PyG dataset object

#     """

#     # get dim_in and dim_out
#     try:
#         if isinstance(dataset.data, HeteroData): # Hetero graph
#             cfg.share.dim_in = CN()
#             task = cfg.dataset.task_entity
#             try:
#                 if hasattr(dataset.data, "x_dict"):
#                     for node_type in dataset.data.node_types:
#                         if node_type in dataset.data.x_dict:
#                             cfg.share.dim_in[node_type] = dataset.data.x_dict[node_type].shape[1]
#                         else:
#                             cfg.share.dim_in[node_type] = None
#             except:
#                 pass # No x_dict
#             try:
#                 if hasattr(dataset.data, "edge_attr_dict"):
#                     for edge_type in dataset.data.edge_types:
#                         if edge_type in dataset.data.edge_attr_dict:
#                             # Key doesn't support tuple
#                             cfg.share.dim_in["__".join(edge_type)] = dataset.data.edge_attr_dict[edge_type].shape[1]
#                         else:
#                             cfg.share.dim_in["__".join(edge_type)] = None
#             except:
#                 pass # No edge_attr_dict
#         else:
#             cfg.share.dim_in = dataset.data.x.shape[-1]
#     except Exception:
#         cfg.share.dim_in = 1
        
#     try:
#         if cfg.dataset.task_type == 'classification':
#             if isinstance(dataset.data, HeteroData): # Hetero graph
#                 task = cfg.dataset.task_entity
#                 if hasattr(dataset.data[task], 'y'):
#                     y = dataset.data[task].y
#                 elif hasattr(dataset.data[task], 'edge_label'):
#                     y = dataset.data[task].edge_label
#             else:
#                 if hasattr(dataset.data, 'y'):
#                     y = dataset.data.y
#                 elif hasattr(dataset.data, 'edge_label'):
#                     y = dataset.data.edge_label

#             if y.numel() == y.size(0) and not torch.is_floating_point(y):
#                 cfg.share.dim_out = int(y.max()) + 1
#             elif y.numel() == y.size(0) and torch.is_floating_point(y):
#                 cfg.share.dim_out = torch.unique(y).numel()
#             else:
#                 cfg.share.dim_out = y.size[-1]
#         else:
#             if isinstance(dataset.data, HeteroData):
#                 task = cfg.dataset.task_entity
#                 if hasattr(dataset.data[task], 'y'):
#                     y = dataset.data[task].y
#                 elif hasattr(dataset.data[task], 'edge_label'):
#                     y = dataset.data[task].edge_label
#             else:
#                 if hasattr(dataset.data, 'y'):
#                     y = dataset.data.y
#                 elif hasattr(dataset.data, 'edge_label'):
#                     y = dataset.data.edge_label

#             cfg.share.dim_out = y.shape[-1]
#     except Exception:
#         cfg.share.dim_out = 1

#     # count number of dataset splits
#     cfg.share.num_splits = 1
#     for key in dataset.data.keys():
#         if 'val' in key:
#             cfg.share.num_splits += 1
#             break
#     for key in dataset.data.keys():
#         if 'test' in key:
#             cfg.share.num_splits += 1
#             break

#     if hasattr(dataset, 'dynamicTemporal'):
#         cfg.share.num_splits = len(dataset)

def count_degree(dataset):
    try:
        data = dataset.data
        # ---- 异构图 ----
        if isinstance(data, HeteroData):
            cfg.share.num_nodes = sum([data[nt].num_nodes for nt in data.node_types])
            cfg.share.num_edges = sum([data[et].num_edges for et in data.edge_types])
            cfg.share.max_indegree = -1
            cfg.share.max_outdegree = -1

        # ---- 同构图 ----
        else:
            # 判断是“单大图”还是“小图集合”
            if hasattr(dataset, '__len__') and len(dataset) > 1:
                # 说明是小图集合，如 ZINC / QM9
                if getattr(dataset.data, "in_degree", None) is not None:
                    # 已经在预处理阶段计算好度数（直接取最大值）
                    cfg.share.num_nodes = int(data.num_nodes)
                    cfg.share.num_edges = int(data.num_edges)
                    cfg.share.max_indegree = int(data.in_degree.max().item())
                    cfg.share.max_outdegree = int(data.out_degree.max().item())
                else:
                    # 没有预计算度数，则逐图计算
                    max_in, max_out = 0, 0
                    total_nodes, total_edges = 0, 0
                    for d in dataset:
                        if hasattr(d, 'edge_index'):
                            row, col = d.edge_index
                            indeg = torch.bincount(col, minlength=d.num_nodes)
                            outdeg = torch.bincount(row, minlength=d.num_nodes)
                            max_in = max(max_in, int(indeg.max().item()))
                            max_out = max(max_out, int(outdeg.max().item()))
                            total_nodes += d.num_nodes
                            total_edges += d.num_edges
                    cfg.share.num_nodes = int(total_nodes)
                    cfg.share.num_edges = int(total_edges)
                    cfg.share.max_indegree = max_in
                    cfg.share.max_outdegree = max_out
            else:
                # 单图类（如 OGBN）
                cfg.share.num_nodes = int(data.num_nodes)
                cfg.share.num_edges = int(data.num_edges)

                if hasattr(data, "edge_index"):
                    row, col = data.edge_index
                    indeg = torch.bincount(col, minlength=data.num_nodes)
                    outdeg = torch.bincount(row, minlength=data.num_nodes)
                    cfg.share.max_indegree = int(indeg.max().item())
                    cfg.share.max_outdegree = int(outdeg.max().item())
                else:
                    cfg.share.max_indegree = -1
                    cfg.share.max_outdegree = -1

    except Exception as e:
        print(f"[Warning] Degree stats failed: {e}")
        cfg.share.num_nodes = -1
        cfg.share.num_edges = -1
        cfg.share.max_indegree = -1
        cfg.share.max_outdegree = -1


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    if isinstance(dataset,DenseFirstDataset):
        set_dataset_info_densefirst(dataset)
    else:
        set_dataset_info(dataset)

    return dataset


def get_loader(dataset, sampler, batch_size, shuffle=True, split='train'):
    # Try to use customized graph sampler
    func = register.sampler_dict.get(sampler, None)
    if func is not None:
        return func(dataset, batch_size=batch_size, shuffle=shuffle, split=split)
    
    if cfg.share.side:
        from torch.utils.data import DataLoader as torch_dataloader
        from GTBenchmark.utils.side_data import global_collate
        loader_train = torch_dataloader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=global_collate,
            pin_memory=True
        )

    elif sampler == "full_batch" or (len(dataset) > 1 and cfg.dataset.task=='graph'):
        loader_train = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)

    elif sampler == "neighbor":
        loader_train = NeighborSampler(
            dataset[0],
            sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeSampler(dataset[0],
                                         num_parts=cfg.train.train_parts,
                                         shuffle=shuffle,
                                         num_workers=cfg.num_workers,
                                         pin_memory=True)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "cluster":
        loader_train = \
            ClusterLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=cfg.num_workers,
                            pin_memory=True)
    else:
        raise NotImplementedError("%s sampler is not implemented!" % sampler)
    return loader_train


def create_loader(dataset = None, shuffle = True, returnDataset = False):
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """

    dataset = create_dataset()


    # train loader
    # if cfg.dataset.task == 'graph':
    #     id = dataset.data['train_graph_index']
    #     loaders = [
    #         get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
    #                     shuffle=True)
    #     ]
    #     delattr(dataset.data, 'train_graph_index')
    if cfg.dataset.task == 'graph':
        train_idx = dataset.index_tensors['train']

        loaders = [ TorchDataloader(
            Subset(dataset, train_idx.tolist()),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=densefirst_collate
        )]

    else:
        loaders = [
            get_loader(dataset,
                        cfg.train.sampler,
                        cfg.train.batch_size,
                        shuffle=True,
                        split='train')
        ]
    print('Create train loader')

    # val and test loaders
    split_names = ['val', 'test']
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            # split_names = ['val_graph_index', 'test_graph_index']
            # id = dataset.data[split_names[i]]
            # loaders.append(
            #     get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
            #                shuffle=shuffle))
            # delattr(dataset.data, split_names[i])
            idx = dataset.index_tensors[split_names[i]]

            loaders.append(TorchDataloader(
                Subset(dataset, idx.tolist()),
                batch_size=cfg.train.batch_size,
                collate_fn=densefirst_collate))
            
        else:
            loaders.append(
                get_loader(dataset,
                            cfg.val.sampler,
                            cfg.train.batch_size,
                            shuffle=shuffle,
                            split=split_names[i]))
            
    print('Create val/test loader')

    if returnDataset:
        return loaders, dataset
    else:
        return loaders
