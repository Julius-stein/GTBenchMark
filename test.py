import scipy.sparse as sp
import numpy as np
import torch
import torch_geometric.loader
import torch_geometric.utils
import torch.functional as F
row = np.array([0, 2, 1])
col = np.array([0, 1, 2])
data = np.array([1, 2.1, 3.0])
shape = (3, 3)
 
sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape)

power_adj_list = []
power_adj_list.append(sparse_matrix)

row = np.array([0, 0, 1, 2])
col = np.array([0, 1, 0, 1])
data = np.array([4, 5.1, 6, 7])
shape = (3, 3)
 
sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape)
power_adj_list.append(sparse_matrix)


all_rows, all_cols = [], []
batch = [0]
for mat in power_adj_list:
    mat = sp.coo_matrix(mat)
    all_rows.extend(mat.row)
    all_cols.extend(mat.col)
    batch.append(len(all_rows))
# 合并并去重
all_cols = torch.tensor(all_cols)
all_rows = torch.tensor(all_rows)
ind = torch.stack((all_rows,all_cols), dim=1)
print(ind)
out, inv = torch.unique(ind,return_inverse=True,dim=0)
print(out)
print(inv)
nv = []
for i, mat in enumerate(power_adj_list):
    temp = torch.zeros(out.shape[0], dtype=torch.float64)
    print(temp)
    print(torch.tensor(mat.data))
    temp[inv[batch[i]:batch[i+1]]]=torch.tensor(mat.data)
    print(temp)
    nv.append(temp)

ans = torch.stack(nv,dim=1)
print(ans)

edge_index = torch.tensor([[0, 1, 3, 2],
                           [1, 2, 2, 0]], dtype=torch.long)
edge_attr = torch.tensor([[0, 1, 3, 2],
                           [1, 2, 2, 0]], dtype=torch.long).t()
from torch_geometric.data import Data
data = Data(num_nodes = 4, edge_index=edge_index, edge_attr=edge_attr).subgraph(torch.tensor([0,2,3]))
print(data)
print(data.edge_index)
print(data.edge_attr)

a = torch.tensor([0,1,2])
b = torch.tensor([3,4,5])
print(torch.stack([a,b],dim=1).t())

print([10]*2)
import torch_geometric

i=1
print(f'A^{i}')

# data = torch.load('./data/error.pt')
# print(data)

# A = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
#                                 shape=(data.y.shape[0], data.y.shape[0]),
#                                 dtype=np.float32)
# N = sp.diags(torch_geometric.utils.degree(data.edge_index[1],num_nodes=data.num_nodes).numpy().clip(1) ** -0.5, dtype=float)
# L = sp.eye(data.num_nodes) - N * A * N
# from scipy.sparse.linalg import eigs
# # Eigenvectors with scipy
# #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
# k = 3
# if k >= data.num_nodes - 1:
#     L = L.toarray()
# # print(L)
# max_retries = 10
# o=0
# for attempt in range(max_retries):
#     try:
#         # 尝试执行的操作
#         EigVal, EigVec = sp.linalg.eigs(L, k=k, which='SR', tol=1e-2) # for 40 PEs
#     except Exception as e:  # 捕获特定的异常类型
#         o+=1
#         if attempt == max_retries - 1:
#             print("Max retries reached. Operation failed.")
#             raise e # 如果达到最大重试次数，抛出异常
# print(o)
# print(data.input_id)
# print(data.n_id)
# EigVec = EigVec[:, EigVal.argsort()] # increasing order
# data.lap_pos_enc = torch.nested.nested_tensor([torch.from_numpy(EigVec[:,1:k]).float()]).to_padded_tensor(0,output_size=(1,data.num_nodes,2)).squeeze(0)


A=torch.tensor([[1,0,0],[0,1,0]],dtype=torch.bool)
B=torch.tensor([[1,0,1]],dtype=torch.bool)
print(A.repeat_interleave(2,dim=0))
A=torch.tensor([[1,0,0],[0,1,0]],dtype=torch.bool)
print(A.view(2,1,3).expand(-1,2,-1).reshape(4,3))