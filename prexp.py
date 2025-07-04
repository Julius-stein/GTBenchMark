import torch
import torch_geometric.datasets as D
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
import tensorboard

device = torch.device("cuda:0")
data = PygNodePropPredDataset(name='ogbn-arxiv')[0].to(device)
print(data)
s=0
y=0
s0=0
import torch_geometric.loader as L
N = data.num_nodes
def collator(batch):
    global s, y, s0
    s0+=batch.num_nodes
    batch.to(batch.x.device)
    idx = torch.argsort(batch.batch, stable=True)
    midx = torch.argsort(idx)
    batch.edge_index = midx[batch.edge_index]
    batch.x = torch.gather(batch.x, 0, idx.unsqueeze(1).expand_as(batch.x))
    batch.y = torch.gather(batch.y, 0, idx.unsqueeze(1).expand_as(batch.y))
    batch.n_id = torch.gather(batch.n_id, 0, idx)
    batch.seed_mask = batch.n_id.new_zeros(batch.num_nodes, dtype=bool)
    batch.seed_mask[:batch.batch_size] = True
    batch.seed_mask = torch.gather(batch.seed_mask, 0, idx)
    batch.batch = torch.gather(batch.batch, 0, idx)

    ptr = list(torch.nonzero(batch.seed_mask).squeeze())
    ptr.append(batch.num_nodes)
    e0, e1 = [], []
    for i in range(batch.batch_size):
        num_nodes = ptr[i+1] - ptr[i]
        tensor = torch.arange(ptr[i], ptr[i+1], device=batch.x.device)
        t1 = tensor.unsqueeze(0).expand(num_nodes, -1).reshape(-1)
        t2 = tensor.unsqueeze(1).expand(-1, num_nodes).reshape(-1)
        e0.append(t1)
        e1.append(t2)

    batch.edge_full_index = torch.stack([torch.cat(e0), torch.cat(e1)], dim=1).t()
    result = batch.n_id[batch.edge_full_index[0]].long() * 100000000 + batch.n_id[batch.edge_full_index[1]].long()
    y+=torch.numel(torch.unique(result))
    s+=torch.numel(result)
    return batch

def random_col(batch):
    batch.to(batch.x.device)
    global s, y, s0
    n_id = []
    e0, e1 = [], []
    for i in range(batch.batch_size):
        n_id.append(batch.n_id[i:i+1])
        n_id.append(torch.randint(0,N,(15,),device=batch.n_id.device))
        num_nodes = (i+1)*16 - i*16
        tensor = torch.arange(i*16, (i+1)*16, device=batch.x.device)
        t1 = tensor.unsqueeze(0).expand(num_nodes, -1).reshape(-1)
        t2 = tensor.unsqueeze(1).expand(-1, num_nodes).reshape(-1)
        e0.append(t1)
        e1.append(t2)

    n_id = torch.cat(n_id)
    batch.edge_full_index = torch.stack([torch.cat(e0), torch.cat(e1)], dim=1).t()
    result = n_id[batch.edge_full_index[0]].long() * 100000000 + n_id[batch.edge_full_index[1]].long()
    y+=torch.numel(torch.unique(result))
    s+=torch.numel(result)

loader = L.NeighborLoader(
    data=data,
    input_nodes=None,
    num_neighbors=[15],
    batch_size=256,
    disjoint=True,
    shuffle=True,
    transform=random_col,
    #num_workers=16
)
import time
t0 = time.time()
from torch_geometric.utils import subgraph
for batch in tqdm(loader):
    pass
t1 = time.time()
print(t1-t0)
print((s-y)/s)
print(s)
print(s0)
from torch_geometric.graphgym import register