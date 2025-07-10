@register_sampler('neighbor')
def get_NeighborLoader(dataset, batch_size, shuffle=True, split='train'):
    r"""
    A homogeneous graph sampler that performs neighbor sampling as introduced
    in the `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`neighbor_sizes` in the configuration denotes
    how much neighbors are sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    Args:
        dataset (Any): A :class:`~torch_geometric.data.InMemoryDataset` dataset object.
        batch_size (int): The number of seed nodes (first nodes in the batch).
        shuffle (bool): Whether to shuffle the data or not (default: :obj:`True`).
        split (str): Specify which data split (:obj:`train`, :obj:`val`, :obj:`test`) is
            for this sampler. This determines some sampling parameter loaded from the
            configuration file, such as :obj:`iter_per_epoch`.
    """

    data = dataset[0]
    sample_sizes = cfg.train.neighbor_sizes
    def reorder_by_seeds(batch):
        r'''
        reorder sampled graph by seed nodes
        '''
        batch.to(batch.x.device)
        idx = torch.argsort(batch.batch, stable=True)
        midx = torch.argsort(idx)
        batch.edge_index = midx[batch.edge_index]
        # batch.x = torch.gather(batch.x, 0, idx.unsqueeze(1).expand_as(batch.x))
        # batch.y = torch.gather(batch.y, 0, idx.unsqueeze(1).expand_as(batch.y))
        # batch.n_id = torch.gather(batch.n_id, 0, idx)
        for key in batch.node_attrs():
            v = getattr(batch, key)
            size = tuple([-1]+[1]*(v.dim()-1))
            setattr(batch, key, torch.gather(v, 0, idx.view(size).expand_as(v)))
        # batch.seed_mask = batch.n_id.new_zeros(batch.num_nodes, dtype=bool)
        # batch.seed_mask[:batch.batch_size] = True
        # batch.seed_mask = torch.gather(batch.seed_mask, 0, idx)
        
        # batch.batch = torch.gather(batch.batch, 0, idx)

        return batch

    from torch_geometric.transforms import Compose
    start = time.time()
    loader_train = \
        LoaderWrapper( \
            NeighborLoader(
                data,
                num_neighbors=sample_sizes,
                input_nodes=data[split + '_mask'],
                batch_size=batch_size,
                shuffle=shuffle,
                disjoint=True,
                num_workers=cfg.num_workers,
                persistent_workers=cfg.train.persistent_workers,
                pin_memory=cfg.train.pin_memory,
                transform=Compose([convert_batch_homo, reorder_by_seeds])),
            getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
            split
        )
    end = time.time()
    print(f'Data {split} loader initialization took:', round(end - start, 3), 'seconds.')
    
    return loader_train