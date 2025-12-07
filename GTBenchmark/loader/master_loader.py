import logging
import gdown, os, time
import os.path as osp
from functools import partial
from typing import Union
from torch_geometric.utils import index_to_mask
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData 
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

from torch_geometric.data import (HeteroData, InMemoryDataset)
from ogb.lsc import MAG240MDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import (DBLP, IMDB, OGB_MAG, Planetoid, MovieLens)
from GTBenchmark.datasets.oag_dataset import OAGDataset
from GTBenchmark.datasets.mag_dataset import MAGDataset
from GTBenchmark.datasets.rcdd_dataset import RCDDDataset
from GTBenchmark.datasets.pokec_dataset import PokecDataset, PokecHomoSGDataset
from GTBenchmark.datasets.ieee_cis_dataset import IeeeCisDataset
from GTBenchmark.datasets.pdns_dataset import PDNSDataset
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from GTBenchmark.graphgym.register import register_loader
from GTBenchmark.transform.posenc_stats import compute_posenc_stats
from GTBenchmark.transform.task_preprocessing import task_specific_preprocessing
from GTBenchmark.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size,to_sparse_tensor)
from GTBenchmark.transform.multihop_prep import generate_multihop_adj
from GTBenchmark.transform.dist_transforms import (add_dist_features, add_reverse_edges,
                                                 effective_resistances, 
                                                 effective_resistance_embedding,
                                                 effective_resistances_from_embedding)
from GTBenchmark.transform.graph_partition import GraphPartitionTransform, MiniBatchFromPartition
from GTBenchmark.transform.reorder import reorder_pyg_dataset
from GTBenchmark.transform.graph_partitionV2 import GraphPartitionTransformV2,GraphPartitionDataset

from torch_geometric.datasets import (Actor, GNNBenchmarkDataset, Planetoid,
                                      TUDataset, WebKB, WikipediaNetwork, ZINC)
from torch_geometric.utils import (index_to_mask, to_undirected,remove_self_loops,add_self_loops)
from GTBenchmark.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from GTBenchmark.loader.encoding_generator import (preprocess_Node2Vec, check_Node2Vec, load_Node2Vec,
                                                 preprocess_Metapath, check_Metapath, load_Metapath,\
                                                 preprocess_KGE, check_KGE, load_KGE)
from GTBenchmark.loader.data_preprocess import hop2token
from GTBenchmark.utils.dataset_cache import load_dataset_from_cache



def get_sparse_tensor(edge_index, num_nodes=None, num_src_nodes=None, num_dst_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t


    if (num_nodes is None) and (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes = int(edge_index.max()) + 1
        num_dst_nodes = num_src_nodes
    elif (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes, num_dst_nodes = num_nodes, num_nodes


    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_src_nodes, num_dst_nodes)).t()


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    try:
        train_idx = torch.where(dataset.data['train_mask'])[0]
        val_idx   = torch.where(dataset.data['val_mask'])[0]
        test_idx  = torch.where(dataset.data['test_mask'])[0]

        logging.info(f"train_idx len: {len(train_idx)}, val_idx len: {len(val_idx)}, test_idx len: {len(test_idx)}")

    except:
        pass
    # logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'num_nodes_dict'):
        total_num_nodes = sum(dataset.data.num_nodes_dict.values())
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )

@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    dataset,_ = load_dataset_from_cache(cfg)
    if dataset is not None:
        return dataset

    use_Hetero = cfg.dataset.heteroProcess
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        if pyg_dataset_id in ['ogbn-mag', 'mag-year']:
            dataset_dir = osp.join(dataset_dir, 'mag')
        else:
            dataset_dir = osp.join(dataset_dir, pyg_dataset_id)
        
        if pyg_dataset_id == 'DBLP':
            dataset = preformat_DBLP(dataset_dir)

        elif pyg_dataset_id == 'IMDB':
            dataset = preformat_IMDB(dataset_dir)
        
        elif pyg_dataset_id == 'OGB_MAG':
            dataset = preformat_OGB_MAG(dataset_dir)

        elif pyg_dataset_id in ['ogbn-mag', 'mag-year']:
            dataset = preformat_MAG(dataset_dir, pyg_dataset_id)

        elif pyg_dataset_id == 'OAG':
            dataset = preformat_OAG(dataset_dir, name)
        
        elif pyg_dataset_id == 'Planetoid':
            dataset = preformat_Planetoid(dataset_dir, name,use_Hetero)

        elif pyg_dataset_id == 'MovieLens':
            dataset = preformat_MovieLens(dataset_dir)

        elif pyg_dataset_id == 'RCDD':
            dataset = preformat_RCDD(dataset_dir)

        elif pyg_dataset_id == 'Pokec':
            dataset = preformat_HomoPokec(dataset_dir)

        elif pyg_dataset_id == 'PDNS':
            dataset = preformat_PDNS(dataset_dir)
                
        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
        
        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbn') or name in ['MAG240M', 'mag-year']:
            dataset = preformat_ogbn(dataset_dir, name)
            # dataset = preformat_OGB_Node(dataset_dir, name.replace('_', '-'))

        elif name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")

    elif format == 'IEEE-CIS':
        dataset = preformat_IEEE_CIS(osp.join(dataset_dir, 'IEEE-CIS'))
    else:
        raise ValueError(f"Unknown data format: {format}")

    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    #--------------------------------------#
    # Deep breath positional Encoding below#
    #--------------------------------------#

    # Precompute structural encodings
    if cfg.posenc_Hetero_Node2Vec.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Node2Vec(pe_dir):
            preprocess_Node2Vec(pe_dir, dataset)
        
        model = load_Node2Vec(pe_dir)
        if len(dataset) == 1:
            homo_data = dataset.data.to_homogeneous()
            for idx, node_type in enumerate(dataset.data.node_types):
                mask = homo_data.node_type == idx
                dataset.data[node_type]['pestat_Hetero_Node2Vec'] = model[mask]
        else:
            data_list = []
            for idx in range(len(dataset)):
                data = dataset.get(idx)
                data.pestat_Hetero_Node2Vec = model[idx]
                data_list.append(data)
                # if idx >= 10:
                #     break
            data_list = list(filter(None, data_list))

            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

    if cfg.posenc_Hetero_Metapath.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Metapath(pe_dir):
            preprocess_Metapath(pe_dir, dataset)
        
        model = load_Metapath(pe_dir)
        emb = model['model'].weight.data.detach().cpu()
        if hasattr(dataset, 'dynamicTemporal'):
            data_list = []
            for split in range(3):
                data = dataset[split]
                for node_type in dataset.data.node_types:
                    data[node_type]['pestat_Hetero_Metapath'] = \
                        emb[model['start'][node_type]:model['end'][node_type]]
                data_list.append(data)
            dataset._data, dataset.slices = dataset.collate(data_list)
        else:
            for node_type in dataset.data.node_types:
                # if hasattr(dataset.data[node_type], 'x'):
                dataset.data[node_type]['pestat_Hetero_Metapath'] = \
                    emb[model['start'][node_type]:model['end'][node_type]]
        print(dataset.data)
    if cfg.posenc_Hetero_TransE.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'TransE'):
            preprocess_KGE(pe_dir, dataset, 'TransE')
        
        model = load_KGE(pe_dir, 'TransE', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_TransE'] = model[node_type].detach().cpu()
    if cfg.posenc_Hetero_ComplEx.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'ComplEx'):
            preprocess_KGE(pe_dir, dataset, 'ComplEx')
        
        model = load_KGE(pe_dir, 'ComplEx', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_ComplEx'] = model[node_type]
    if cfg.posenc_Hetero_DistMult.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'DistMult'):
            preprocess_KGE(pe_dir, dataset, 'DistMult')
        
        model = load_KGE(pe_dir, 'DistMult', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_DistMult'] = model[node_type]
            
    # print(dataset[0])

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        # if 'LapPE' in pe_enabled_list:
        #     start = time.perf_counter()
        #     logging.info(f"Precomputing Positional Encoding statistics: "
        #                  f"{pe_enabled_list} for all graphs...")
        #     # Estimate directedness based on 10 graphs to save time.
        #     is_undirected = all(d.is_undirected() for d in dataset[:10])
        #     logging.info(f"  ...estimated to be undirected: {is_undirected}")

        #     pe_dir = osp.join(dataset_dir, 'LapPE')
        #     file_path = osp.join(pe_dir, f'{dataset.name}.pt')
        #     if not osp.exists(pe_dir) or not osp.exists(file_path):
        #         from tqdm import tqdm
        #         results = []
        #         for i in tqdm(range(len(dataset)),
        #                        mininterval=10,
        #                        miniters=len(dataset)//20):
        #             data = compute_posenc_stats(dataset.get(i), 
        #                                         pe_types=pe_enabled_list,
        #                                         is_undirected=is_undirected,
        #                                         cfg=cfg)
        #             results.append({'EigVals': data.EigVals, 'EigVecs': data.EigVecs})
        #         if not osp.exists(pe_dir):
        #             os.makedirs(pe_dir)
        #         torch.save(results, file_path)
            
        #     from tqdm import tqdm
        #     results = torch.load(file_path)
        #     data_list = []
        #     for i in tqdm(range(len(dataset)),
        #                 mininterval=10,
        #                 miniters=len(dataset)//20):
        #         data = dataset.get(i)
        #         data.EigVals = results[i]['EigVals']
        #         data.EigVecs = results[i]['EigVecs']
        #         data_list.append(data)
        #     data_list = list(filter(None, data_list))

        #     dataset._indices = None
        #     dataset._data_list = data_list
        #     dataset.data, dataset.slices = dataset.collate(data_list)

        #     elapsed = time.perf_counter() - start
        #     timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
        #               + f'{elapsed:.2f}'[-3:]
        #     logging.info(f"PE Done! Took {timestr}")

        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True,side=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

#-------------------------#
# Preprocess token choices#
#-------------------------#

    logging.info(f"Preprocessing dataset with {cfg.dataset.preprocess} ...")
    start = time.perf_counter()
    if cfg.dataset.preprocess in ["None","HopToToken","Expand"]:
        if cfg.dataset.preprocess == "None":
            pass
        if cfg.dataset.preprocess == "Expand":
            from GTBenchmark.transform.expander_edges import generate_random_expander
            for j in range(cfg.dataset.exp_count):
                pre_transform_in_memory(dataset,
                            partial(generate_random_expander,
                                    degree = cfg.dataset.exp_deg,
                                    algorithm = cfg.dataset.exp_algorithm,
                                    rng = None,
                                    max_num_iters = cfg.dataset.exp_max_num_iters,
                                    exp_index = j),
                            show_progress=True
                            )

        if cfg.dataset.preprocess == "HopToToken":
            dataset = hop2token(dataset)
    else:
        raise ValueError(f"Unexpect preprocess type {cfg.dataset.preprocess}.")
    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                + f'{elapsed:.2f}'[-3:]
    logging.info(f"Done! Took {timestr}")

    if cfg.gt.rb_order > 1:
        start = time.perf_counter()
        logging.info(f"Generating multi-hop adjacency matrices ...")
        pre_transform_in_memory(dataset,
                                partial(generate_multihop_adj,
                                        cfg = cfg),
                                show_progress=True,side=False
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")
  
    if cfg.metis.patches > 0:
        start = time.perf_counter()
        logging.info(f"Precomputing graph partition transform ...")

        transform = GraphPartitionTransformV2(n_patches=cfg.metis.patches,
                                            algo='metis' if cfg.metis.enable else 'random',
                                            drop_rate=cfg.metis.drop_rate,
                                            num_hops=cfg.metis.num_hops,
                                            cut_type=cfg.metis.cut_type,
                                            patch_rw_dim=cfg.metis.patch_rw_dim,
                                            patch_num_diff=cfg.metis.patch_num_diff,
                                            mode= cfg.metis.mode
                                            )
        dataset = transform(dataset[0])

        
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # re_enabled_list = []
    # for key, recfg in cfg.items():
    #     if key.startswith('reorder_') and recfg.enable:
    #         pe_name = key.split('_', 1)[1]
    #         pe_enabled_list.append(pe_name)

    # if re_enabled_list:   

    #     start = time.perf_counter()
    #     logging.info(f"Reorder Graph index for GNN/MPAttn ...")

    #     # pre_transform_in_memory(dataset, GraphPartitionTransform(n_patches=cfg.metis.patches,
    #     #                                                          metis=cfg.metis.enable,
    #     #                                                          drop_rate=cfg.metis.drop_rate,
    #     #                                                          num_hops=cfg.metis.num_hops,
    #     #                                                          is_directed=False,
    #     #                                                          patch_rw_dim=cfg.metis.patch_rw_dim,
    #     #                                                          patch_num_diff=cfg.metis.patch_num_diff),
    #     #                         show_progress=True)
    #     data = dataset[0]

    #     # Step 1: 原图 edge_index 用 Random
    #     data = reorder_pyg_dataset(data, methods=['Random'], attr_name='edge_index')

    #     # Step 2: expander_edges 图用 Metis
    #     data = reorder_pyg_dataset(data, methods=['Metis'], attr_name='expander_edges')
    #     dataset[0] = data
        
    #     elapsed = time.perf_counter() - start
    #     timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
    #                 + f'{elapsed:.2f}'[-3:]
    #     logging.info(f"Done! Took {timestr}")
    if cfg.dataset.add_self_loops:
        for i, data in enumerate(dataset):
            edge_index, edge_attr = remove_self_loops(
                data.edge_index,
                data.edge_attr if hasattr(data, "edge_attr") else None
            )
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr=edge_attr,
                num_nodes=data.num_nodes
            )
            data.edge_index = edge_index
            if hasattr(data, "edge_attr"):
                data.edge_attr = edge_attr
            else:
                data.edge_attr = edge_attr


    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)
    # dataset.data['extra_loss'] = torch.Tensor([0.0])
    for data in dataset:
        data.extra_loss = torch.tensor([0.0])


    # # Precompute in-degree histogram if needed for PNAConv.
    # if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
    #     cfg.gt.pna_degrees = compute_indegree_histogram(
    #         dataset[dataset.data['train_graph_index']])
    #     # print(f"Indegrees: {cfg.gt.pna_degrees}")
    #     # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_DBLP(dataset_dir):
    """Load and preformat DBLP datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = DBLP(root=dataset_dir)
    dataset.data['conference'].x = torch.ones(dataset.data['conference'].num_nodes, 1)
    return dataset


def preformat_IMDB(dataset_dir):
    """Load and preformat IMDB datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = IMDB(root=dataset_dir)
    # data = dataset.data
    # m = data.num_nodes_dict['movie']
    # n = data.num_nodes_dict['director']
    # k = data.num_nodes_dict['movie']
    # edge_index_1 = data[('movie', 'to', 'director')].edge_index#.to(cfg.device)
    # edge_index_2 = data[('director', 'to', 'movie')].edge_index#.to(cfg.device)
    # import torch_sparse
    # edge_index, _ = torch_sparse.spspmm(edge_index_1, torch.ones(edge_index_1.shape[1], device=edge_index_1.device), 
    #                                     edge_index_2, torch.ones(edge_index_2.shape[1], device=edge_index_2.device), 
    #                                     m, n, k, True)
    # data[('movie', 'MDM', 'movie')].edge_index = edge_index

    # m = data.num_nodes_dict['movie']
    # n = data.num_nodes_dict['actor']
    # k = data.num_nodes_dict['movie']
    # edge_index_1 = data[('movie', 'to', 'actor')].edge_index#.to(cfg.device)
    # edge_index_2 = data[('actor', 'to', 'movie')].edge_index#.to(cfg.device)
    # edge_index, _ = torch_sparse.spspmm(edge_index_1, torch.ones(edge_index_1.shape[1], device=edge_index_1.device), 
    #                                     edge_index_2, torch.ones(edge_index_2.shape[1], device=edge_index_2.device), 
    #                                     m, n, k, True)
    # data[('movie', 'MAM', 'movie')].edge_index = edge_index
    return dataset


def preformat_OGB_MAG(dataset_dir):
    """Load and preformat OGB_MAG datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    # transform = T.ToUndirected(merge=True)
    # dataset = OGB_MAG(root=dataset_dir, preprocess='metapath2vec', transform=transform)
    dataset = MAGDataset(root=dataset_dir)
    # dataset = OGB_MAG(root=dataset_dir, preprocess='TransE', transform=transform)
    return dataset


def preformat_OAG(dataset_dir, name):
    """Load and preformat home-brewed OAG datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    transform = T.ToUndirected(merge=True)
    dataset = OAGDataset(root=dataset_dir, name=name, transform=transform)
    if cfg.dataset.rand_split:
        data = dataset.data
        total = (data['paper'].train_mask.sum() + data['paper'].val_mask.sum() + data['paper'].test_mask.sum()).item()
        train_size = data['paper'].train_mask.sum().item() / total
        valid_size= data['paper'].val_mask.sum().item() / (total - data['paper'].train_mask.sum().item())
        print(train_size, valid_size)
        train_paper, temp_paper = train_test_split(torch.where(data['paper'].y != -1)[0], train_size=train_size)
        valid_paper, test_paper = train_test_split(temp_paper, train_size=valid_size)
        data['paper'].train_mask = index_to_mask(train_paper, size=data.num_nodes_dict['paper'])
        data['paper'].val_mask = index_to_mask(valid_paper, size=data.num_nodes_dict['paper'])
        data['paper'].test_mask = index_to_mask(test_paper, size=data.num_nodes_dict['paper'])
    # data = dataset.data
    # data[('paper', 'PAP', 'paper')].adj_t = (data[('paper', 'rev_AP_write_first', 'author')].adj_t.t() @  data[('author', 'AP_write_first', 'paper')].adj_t.t()).t()
    return dataset


def preformat_Planetoid(dataset_dir, name, use_hetero):
    """Load and preformat Planetoid datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    if use_hetero:
        dataset = Planetoid(root=dataset_dir, name=name, transform=T.NormalizeFeatures(), \
                            split="random", num_train_per_class=3943, num_val=3943, num_test=3943)

        data = HeteroData()
        data['paper'].x = dataset[0].x # assuming there's just one node type.
        data['paper'].y = dataset[0].y.squeeze().masked_fill(torch.isnan(dataset[0].y.squeeze()), -1).type(torch.int64)
        # data['paper', 'cites', 'paper'].edge_index = to_undirected(dataset[0].edge_index) # dataset[0].edge_index # 
        adj_t = get_sparse_tensor(dataset[0].edge_index, num_nodes=dataset[0].x.size(0))
        adj_t = adj_t.to_symmetric()
        data['paper', 'cites', 'paper'].adj_t = adj_t

        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i in range(len(split_names)):
            data['paper'][split_names[i]] = dataset[0][split_names[i]]
            
        dataset.data = data
    else:
        dataset = Planetoid(dataset_dir, name)

    return dataset


def preformat_MovieLens(dataset_dir):
    """Load and preformat MovieLens datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = MovieLens(root=dataset_dir)
    return dataset


def preformat_RCDD(dataset_dir):
    """Load and preformat RCDD datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    # transform = T.ToUndirected(merge=True)
    dataset = RCDDDataset(root=dataset_dir) #, transform=transform)
    dataset.name = 'RCDD'
    return dataset


def preformat_HomoPokec(dataset_dir):
    root = osp.join(dataset_dir, "PokecHomoSG")
    dataset = PokecHomoSGDataset(root)
    return dataset


def preformat_Pokec(dataset_dir):
    """Load and preformat Pokec datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    # transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.NormalizeFeatures()])
    transform = T.ToUndirected()
    dataset = PokecDataset(root=dataset_dir, transform=transform)
    dataset.name = 'Pokec'
    if cfg.dataset.rand_split:
        data = dataset.data
        indices = torch.where(data['user'].y != -1)[0]
        train_size = 0.5 # 50% train
        val_size = 0.5 # 25% val, 25% test
        train_idx, temp_idx = train_test_split(indices, train_size=train_size)
        val_idx, test_idx = train_test_split(temp_idx, train_size=val_size)
        train_mask = index_to_mask(train_idx, data['user'].num_nodes)
        val_mask = index_to_mask(val_idx, data['user'].num_nodes)
        test_mask = index_to_mask(test_idx, data['user'].num_nodes)
        data['user'].train_mask = train_mask
        data['user'].val_mask = val_mask
        data['user'].test_mask = test_mask
    return dataset


def preformat_PDNS(dataset_dir):
    """Load and preformat DNS datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    transform = T.ToUndirected()
    start, end = 0, 60
    dataset = PDNSDataset(root=dataset_dir, start=start, end=end,
                         domain_file='domains2.csv', transform=transform)
    return dataset


def preformat_MAG(dataset_dir, name):
    """Load and preformat ogbn-mag or mag-year datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: dataset name

    Returns:
        PyG dataset object
    """
    dataset = MAGDataset(root=dataset_dir, name=name, 
                         rand_split=cfg.dataset.rand_split)
    return dataset


def preformat_ogbn(dataset_dir, name):
    dataset = PygNodePropPredDataset(name=name, root=dataset_dir)

    # === 预处理 ===
    if name == 'ogbn-arxiv':
        dataset._data['edge_index'] = to_undirected(dataset._data['edge_index'])
        # pre_transform_in_memory(dataset, partial(add_reverse_edges))

    if name == 'ogbn-proteins':
        edge_index_ = to_sparse_tensor(dataset._data['edge_index'],
                                   dataset._data['edge_attr'], dataset._data['num_nodes'])
        dataset._data['x'] = edge_index_.mean(dim=1)
        dataset._data['edge_attr'] = None
        # pre_transform_in_memory(dataset, partial(move_node_feat_to_x))
        # pre_transform_in_memory(dataset, partial(typecast_x, type_str='long'))

    data = dataset[0]
    data.y = data.y.squeeze(-1).to(torch.long)

    # === Split 逻辑 ===
    if not cfg.dataset.rand_split:
        # 官方 split
        split_idx = dataset.get_idx_split()
        if "valid" in split_idx:  # 统一 key
            split_idx['val'] = split_idx.pop('valid')

        # 显式写入 mask
        data.train_mask = index_to_mask(split_idx['train'], size=data.num_nodes)
        data.val_mask   = index_to_mask(split_idx['val'],   size=data.num_nodes)
        data.test_mask  = index_to_mask(split_idx['test'],  size=data.num_nodes)

        dataset.split_idx = split_idx

    else:
        # 自定义随机划分
        idx = torch.where(data.y != -1)[0]

        # 比例 (85% train, 9% val, 6% test)
        train_size = 0.85
        valid_size = 0.6  # 注意是对剩余部分的比例
        train_idx, temp_idx = train_test_split(idx, train_size=train_size)
        val_idx, test_idx   = train_test_split(temp_idx, train_size=valid_size)

        # 显式写入 mask
        data.train_mask = index_to_mask(torch.tensor(train_idx), size=data.num_nodes)
        data.val_mask   = index_to_mask(torch.tensor(val_idx),   size=data.num_nodes)
        data.test_mask  = index_to_mask(torch.tensor(test_idx),  size=data.num_nodes)

        dataset.split_idx = {
            "train": torch.tensor(train_idx),
            "val":   torch.tensor(val_idx),
            "test":  torch.tensor(test_idx),
        }

    # 确保 dataset[0] 上有 mask
    dataset.data = data  

    return dataset


def preformat_OGB_Node(dataset_dir, name):
    """Load and preformat OGB Node Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    start = time.time()

    path = osp.join(dataset_dir, '_'.join(name.split('-')), 'transformed')
    if not (osp.exists(path) and osp.isdir(path)) or cfg.dataset.rand_split:
        if name in ['ogbn-mag', 'mag-year']:
            dataset = PygNodePropPredDataset(name='ogbn-mag', root=dataset_dir)
            graph = dataset[0]
            data = HeteroData()

            # Add edges (sparse adj_t)
            for edge_type in graph.edge_reltype:
                src_type, rel, dst_type = edge_type
                data[(src_type, rel, dst_type)].adj_t = \
                    get_sparse_tensor(graph.edge_index_dict[edge_type], 
                                    num_src_nodes=graph.num_nodes_dict[src_type],
                                    num_dst_nodes=graph.num_nodes_dict[dst_type])
                if src_type == dst_type:
                    data[(src_type, rel, dst_type)].adj_t = \
                        data[(src_type, rel, dst_type)].adj_t.to_symmetric()
                else:
                    row, col = graph.edge_index_dict[edge_type]
                    rev_edge_index = torch.stack([col, row], dim=0)
                    data[(dst_type, 'rev_' + rel, src_type)].adj_t = \
                        get_sparse_tensor(rev_edge_index, 
                                        num_src_nodes=graph.num_nodes_dict[dst_type],
                                        num_dst_nodes=graph.num_nodes_dict[src_type])
            
            # data[('paper', 'PAP', 'paper')].adj_t = (data[('paper', 'rev_writes', 'author')].adj_t.t() @  data[('author', 'writes', 'paper')].adj_t.t()).t()
            # data[('paper', 'PSP', 'paper')].adj_t = (data[('paper', 'has_topic', 'field_of_study')].adj_t.t() @  data[('field_of_study', 'rev_has_topic', 'paper')].adj_t.t()).t()

            # Add node features
            def normalize(mx):
                """Row-normalize sparse matrix"""
                rowsum = np.array(mx.sum(1))
                r_inv = np.power(rowsum, -1).flatten()
                r_inv[np.isinf(r_inv)] = 0.
                r_mat_inv = sp.diags(r_inv)
                mx = r_mat_inv.dot(mx)
                return mx

            data['paper'].x = graph.x_dict['paper'] # torch.cat((graph.x_dict['paper'], torch.log10(deg['paper'].reshape(-1, 1))), axis=-1)
            data['paper'].y = graph.y_dict['paper'].squeeze()

            if not cfg.dataset.rand_split:
                split_idx = dataset.get_idx_split()
                train_paper = split_idx['train']['paper']
                valid_paper = split_idx['valid']['paper']
                test_paper  = split_idx['test']['paper']
            else:
                train_size=.85 # 85%
                valid_size=.6 # 9%
                train_paper, temp_paper = train_test_split(torch.where(data['paper'].y != -1)[0], train_size=train_size)
                valid_paper, test_paper = train_test_split(temp_paper, train_size=valid_size)
            data['paper'].train_mask = index_to_mask(train_paper, size=graph.y_dict['paper'].shape[0])
            data['paper'].val_mask = index_to_mask(valid_paper, size=graph.y_dict['paper'].shape[0])
            data['paper'].test_mask = index_to_mask(test_paper, size=graph.y_dict['paper'].shape[0])

            # Average paper's embedding to corresponding author and field of study
            for edge_type in [('author', 'writes', 'paper'), ('field_of_study', 'rev_has_topic', 'paper')]:
                src_type, rel, dst_type = edge_type
                edge_index = data.edge_index_dict[edge_type]
                v = torch.ones(edge_index.shape[1])
                m = normalize(sp.coo_matrix((v, edge_index), \
                    shape=(graph.num_nodes_dict[src_type], graph.num_nodes_dict[dst_type])))
                out = m.dot(data[dst_type].x)
                data[src_type].x = torch.from_numpy(out) # torch.cat((torch.from_numpy(out), torch.log10(deg[node_type].reshape(-1, 1))), axis=-1)

            # Average author's embedding to corresponding institution
            edge_type = ('institution', 'rev_affiliated_with', 'author')
            edge_index = data.edge_index_dict[edge_type]
            v = torch.ones(edge_index.shape[1])
            m = normalize(sp.coo_matrix((v, edge_index), \
                shape=(graph.num_nodes_dict['institution'], graph.num_nodes_dict['author'])))
            out = m.dot(data['author'].x)
            data['institution'].x = torch.from_numpy(out) # torch.cat((torch.from_numpy(out), torch.log10(deg['institution'].reshape(-1, 1))), axis=-1)  

            if name == 'mag-year':
                label = even_quantile_labels(graph['node_year']['paper'].squeeze().numpy(), nclasses=5, verbose=False)
                data['paper'].y = torch.from_numpy(label).squeeze() 
                train_size=.5 # 50%
                valid_size=.5 # 25%
                train_idx, temp_idx = train_test_split(torch.where(data['paper'].y != -1)[0], train_size=train_size)
                val_idx, test_idx = train_test_split(temp_idx, train_size=valid_size)
                data['paper'].train_mask = index_to_mask(train_idx, size=data['paper'].y.shape[0])
                data['paper'].val_mask = index_to_mask(val_idx, size=data['paper'].y.shape[0])
                data['paper'].test_mask = index_to_mask(test_idx, size=data['paper'].y.shape[0])

        elif name in ['ogbn-arxiv']:
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=dataset_dir)
            data = dataset[0]
            data.y = data.y.squeeze(-1).to(torch.long) 
            split_idx = dataset.get_idx_split()
            print(data)
            print(split_idx)
            # data.edge_index = get_sparse_tensor(data.edge_index, num_nodes=data.num_nodes)
            data.train_mask = index_to_mask(split_idx['train'], size=data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid'], size=data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test'], size=data.num_nodes)
        if not cfg.dataset.rand_split:
            os.mkdir(path)
            torch.save(data, osp.join(path, 'data.pt'))
    else:
        data = torch.load(osp.join(path, 'data.pt'),weights_only=False)
    
    r'''A fake OGB dataset. 
    Once the OGB dataset is preprocessed and transformed to sparse format, the original OGB
    dataset is no needed any more. We can simply use this fake class to load the preprocessed
    version, rather than load the original dataset and overwrite the data inside.
    Save space and time.
    '''
    dataset = InMemoryDataset(root=dataset_dir)
    dataset.name = name
    dataset.data = data

    end = time.time()
    print('Dataset loading took', round(end - start, 3), 'seconds')
    
    return dataset


def preformat_IEEE_CIS(dataset_dir):
    """Load and preformat custom IEEE-CIS datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    # transform = T.ToUndirected(merge=True)
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.NormalizeFeatures()])
    dataset = IeeeCisDataset(root=dataset_dir, transform=transform)
    dataset.name = 'IEEE-CIS'
    data = dataset.data
    num_nodes_train = int(0.8 * data["transaction"].num_nodes)
    num_nodes_val = int(0.1 * data["transaction"].num_nodes)
    data["transaction"].train_mask = torch.zeros(data["transaction"].num_nodes, dtype=bool)
    data["transaction"].train_mask[:num_nodes_train] = True
    data["transaction"].val_mask = torch.zeros(data["transaction"].num_nodes, dtype=bool)
    data["transaction"].val_mask[num_nodes_train:num_nodes_train + num_nodes_val] = True
    data["transaction"].test_mask = torch.zeros(data["transaction"].num_nodes, dtype=bool)
    data["transaction"].test_mask[num_nodes_train + num_nodes_val:] = True
    return dataset

def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]


#     @classmethod
#     def from_ogb(self, name: str, root='dataset'):
#         print('Obtaining dataset from ogb...')
#         return self.process_ogb(PygNodePropPredDataset(name=name, root=root))

#     @classmethod
#     def process_ogb(self, dataset):
#         print('Converting to fast dataset format...')
#         data = dataset.data
#         x = to_row_major(data.x).to(torch.float16)
#         y = data.y.squeeze()

#         if y.is_floating_point():
#             y = y.nan_to_num_(-1)
#             y = y.long()

#         adj_t = get_sparse_tensor(data.edge_index, num_nodes=x.size(0))
#         rowptr, col, _ = adj_t.to_symmetric().csr()
#         return self(name=dataset.name, x=x, y=y,
#                    rowptr=rowptr, col=col,
#                    split_idx=dataset.get_idx_split(),
#                    meta_info=dataset.meta_info.to_dict())

#     @classmethod
#     def from_path(self, _path, name):
#         path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
#         if not (path.exists() and path.is_dir()):
#             print(f'First time preprocessing {name}; may take some time...')
#             dataset = self.from_ogb(name, root=_path)
#             print(f'Saving processed data...')
#             dataset.save(_path, name)
#             return dataset
#         else:
#             return self.from_path_if_exists(_path, name)
        
#     def save(self, _path, name):
#         path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
#         # path.mkdir()
#         for i, field in enumerate(self._fields):
#             torch.save(self[i], path.joinpath(field + '.pt'))
    # @classmethod
    # def from_path_if_exists(self, _path, name):
    #     path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
    #     assert path.exists() and path.is_dir()
    #     data = {
    #         field: torch.load(path.joinpath(field + '.pt'))
    #         for field in self._fields
    #     }
    #     data['y'] = data['y'].long()
    #     data['x'] = data['x'].to(torch.float16)
    #     assert data['name'] == name
    #     return self(**data)

def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset

def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-ppa':
        # ogbg-ppa doesn't have any node features, therefore add zeros but do
        # so dynamically as a 'transform' and not as a cached 'pre-transform'
        # because the dataset is big (~38.5M nodes), already taking ~31GB space
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
        dataset.transform = add_zeros
    elif name == 'ogbg-code2':
        from GTBenchmark.datasets.ogbg_code2_utils import idx2vocab, \
            get_vocab_mapping, augment_edge, encode_y_to_arr
        num_vocab = 5000  # The number of vocabulary used for sequence prediction
        max_seq_len = 5  # The maximum sequence length to predict

        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        logging.info(f"Target sequences less or equal to {max_seq_len} is "
            f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

        # Building vocabulary for sequence prediction. Only use training data.
        vocab2idx, idx2vocab_local = get_vocab_mapping(
            [dataset.data.y[i] for i in s_dict['train']], num_vocab)
        logging.info(f"Final size of vocabulary is {len(vocab2idx)}")
        idx2vocab.extend(idx2vocab_local)  # Set to global variable to later access in CustomLogger

        # Set the transform function:
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
        dataset.transform = T.Compose(
            [augment_edge,
             lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

        # Subset graphs to a maximum size (number of nodes) limit.
        pre_transform_in_memory(dataset, partial(clip_graphs_to_size,
                                                 size_limit=1000))

    return dataset