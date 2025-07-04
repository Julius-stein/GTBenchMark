import torch
import random

def pad_1d_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, attn_bias, x, y, ids):
        super(Batch, self).__init__()
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.ids = ids

    def to(self, device):
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.ids = self.ids.to(device)
        return self

    def __len__(self):
        return self.y.size(0)


def collator(items, feature, shuffle=False, perturb=False):
    batch_list = []
    for item in items:
        for x in item:
            batch_list.append((x[0], x[1], x[2][0]))
    if shuffle:
        random.shuffle(batch_list)
    attn_biases, xs, ys = zip(*batch_list)
    max_node_num = max(i.size(0) for i in xs)
    y = torch.cat([i.unsqueeze(0) for i in ys])
    x = torch.cat([pad_2d_unsqueeze(feature[i], max_node_num) for i in xs])
    ids = torch.cat([i.unsqueeze(0) for i in xs])
    if perturb:
        x += torch.FloatTensor(x.shape).uniform_(-0.1, 0.1)
    attn_bias = torch.cat([i.unsqueeze(0) for i in attn_biases])
    print(attn_bias)

    return Batch(
        attn_bias=attn_bias,
        x=x,
        y=y,
        ids=ids,
    )

def Gophormer_collator(batch, enc_dim, device):
    x = []
    y = []
    ids = []
    max_node_num = max(data['N'].x.size(0) for data in batch)
    attn_bias = torch.zeros((len(batch), max_node_num, max_node_num, enc_dim), dtype=torch.float)
    for i, data in enumerate(batch):
        x.append(data['N'].x)
        y.append(data['N'].y[0])
        ids.append(data['N'].n_id)
        for str1, str2, str3 in data.edge_types:
            if 'A' in str2:
                _ = int(str2[-1])
                aa = data[str1,str2,str3]
                attn_bias[i, aa.edge_index[0], aa.edge_index[1], _] = aa.edge_attr
                
    # if shuffle:
    #     random.shuffle(batch_list)
    # attn_biases, xs, ys = zip(*batch_list)
    y = torch.cat(y)
    # x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in x])
    # ids = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in ids])
    # attn_bias = torch.cat([pad_edge_type_unsqueeze(i,max_node_num) for i in attn_bias])

    return Batch(
        attn_bias=attn_bias,
        x=torch.nested.nested_tensor(x).to_padded_tensor(0),
        y=y,
        ids=torch.nested.nested_tensor(ids).to_padded_tensor(0),
    )

from dataprocess.encoder import laplacian_positional_encoding, wl_positional_encoding
import torch_geometric.data
def GraphTransformer_collator(batch, pos_enc_dim, lap_pos_enc, wl_pos_enc):
    for data in batch:
        if lap_pos_enc:
            laplacian_positional_encoding(data, pos_enc_dim)
        if wl_pos_enc:
            wl_positional_encoding(data)
    return torch_geometric.data.Batch().from_data_list(batch)
