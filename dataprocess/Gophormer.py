import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.loader import NeighborLoader
import tqdm
import scipy.sparse as sp
import numpy as np
import torch.sparse

def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def process_data():
    # 加载ogbn-arxiv数据集
    name = 'ogbn_arxiv'
    #dataset = Planetoid(root='./dataset/', name=name)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    print(data)
    num_nodes = data.y.shape[0]
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                shape=(num_nodes, num_nodes),
                                dtype=np.float32)
    normalized_adj = adj_normalize(adj)
    #sp.save_npz('./dataset/'+name+'/normalized_adj.npz', normalized_adj)
    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])

    #power_adj_list = torch.cat([torch.tensor(i.todense()).unsqueeze(0) for i in power_adj_list])
    # print('ok')

    # all_rows, all_cols = [], []
    # batch = [0]
    # for mat in power_adj_list:
    #     mat = sp.coo_matrix(mat)
    #     all_rows.extend(mat.row)
    #     all_cols.extend(mat.col)
    #     batch.append(len(all_rows))
    # # 合并并去重
    # all_cols = torch.tensor(all_cols)
    # all_rows = torch.tensor(all_rows)
    # ind = torch.stack((all_rows,all_cols), dim=1)
    # print(ind)
    # out, inv = torch.unique(ind,return_inverse=True,dim=0)
    # print(out)
    # print(inv)
    # nv = []
    # for i, mat in enumerate(power_adj_list):
    #     temp = torch.zeros(out.shape[0], dtype=torch.float64)
    #     print(temp)
    #     print(torch.tensor(mat.data))
    #     temp[inv[batch[i]:batch[i+1]]]=torch.tensor(mat.data)
    #     print(temp)
    #     nv.append(temp)

    # ans = torch.stack(nv,dim=1)

    from torch_geometric.data import HeteroData
    pal = HeteroData()
    pal['N'].x=data.x
    pal['N'].y=data.y
    pal['N','E','N'].edge_index=data.edge_index
    for i, mat in enumerate(power_adj_list):
        mat = sp.coo_matrix(mat)
        pal['N',f'A^{i}','N'].edge_index=torch.stack((torch.tensor(mat.row), torch.tensor(mat.col)), dim=1).t().long()
        pal['N',f'A^{i}','N'].edge_attr=torch.tensor(mat.data, dtype=torch.float32)
    
    pal = pal.contiguous()
    print(pal)
    print(pal.edge_types)

    torch.save(pal, './dataset/' + name + '/pal.pt')
    torch.save(data, './dataset/'+name+'/data.pt')
    
    

    # 创建邻居采样器
    # loader = NeighborLoader(
    #     data,
    #     input_nodes=None,  # 所有节点作为中心节点
    #     num_neighbors=[15],  # 两层采样，分别采样25和10个邻居
    #     batch_size=1,             # 每个子图一个中心节点
    #     shuffle=False,            # 保持原始顺序
    # )
    # # 遍历并保存所有子图
    # data_list = []
    # for batch in loader:
    #     node_feature_id = batch.n_id
    #     attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])

    #     attn_bias = attn_bias.permute(1, 2, 0)

    #     # label = torch.cat([data.y[node_feature_id], torch.zeros(len(super_node_list))]).long()
    #     # feature_id = torch.cat([node_feature_id, torch.tensor(super_node_list + data.y.shape[0], dtype=int)])

    #     label = data.y[node_feature_id].long()
    #     feature_id = node_feature_id
    #     sub_data_list=[[attn_bias, feature_id, label]]
    #     data_list.append(sub_data_list)
        
    #     # 保存子图数据

    # torch.save(data_list, )

    # feature = data.x
    # torch.save(feature, './dataset/'+name+'/feature.pt')

if __name__ == '__main__':
    import time
    t=time.time()
    process_data()
    print(time.time()-t)

