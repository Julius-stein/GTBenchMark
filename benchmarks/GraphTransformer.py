import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.utils
from utils.collator import GraphTransformer_collator
import random
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from models.GraphTransformer import GraphTransformerNet
from torch.optim.lr_scheduler import _LRScheduler
from dataprocess.BatchAccumulator import WrapperLoader

import time
from utils.timer import Timer
timer = Timer()

import torch.profiler


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup, tot, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup = warmup
        self.tot = tot
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer)


    def get_lr(self):
        if self._step_count <= self.warmup:
            self.warmup_factor = self._step_count / float(self.warmup)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot:
            lr = self.end_lr
        else:
            warmup = self.warmup
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (self.tot - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False
import argparse
import math
from tqdm import tqdm
from dataprocess.ANS_GT import node_sampling, process_data
from torch.nn.functional import normalize
import scipy.sparse as sp
from numpy.linalg import inv

from torch_geometric.utils import to_dense_batch

def train(args, model, device, loader, optimizer, lr_scheduler, prof = None):
    model.train()

    s0=0
    s1=0
    flg=0
    sum_time = 0
    #with torch.autograd.detect_anomaly():
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        print(batch)
        
        with timer.accumulate_timer('train'):
            # print(batch)
            # ids = to_dense_batch(batch.n_id, batch.batch)
            # if flg == 0:
            #     print(ids.shape)
            #     flg=1
            # tensor = ids.to(torch.int64)
            # n_rows, n_elements = tensor.shape
        
            # # 生成所有可能的组合坐标
            # x1 = tensor.unsqueeze(2).expand(-1, -1, n_elements)  # 形状 (n, m, m)
            # x2 = tensor.unsqueeze(1).expand(-1, n_elements, -1)  # 形状 (n, m, m)
            
            # # 合并坐标对并重塑形状
            # cartesian = torch.stack([x1, x2], dim=-1)            # 形状 (n, m, m, 2)
            # x = cartesian[..., 0].long()  # 转换为 int64
            # y = cartesian[..., 1].long()
            # result = x * 1000000 + y

            # s0+=torch.numel(torch.unique(result))
            # s1+=torch.numel(result)
            
            pred = model(batch)
            y, mask = to_dense_batch(batch.y, batch.batch)
            y = y[:,0,0]
            loss = F.nll_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            optimizer.step()
        # prof.step()

    # print((s1-s0)/s1)


def eval_train(args, model, device, loader):
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    sum_time = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            with timer.accumulate_timer('eval_train'):
                pred = model(batch)
                y, mask = to_dense_batch(batch.y, batch.batch)
                y = y[:,0,0]
                loss_list.append(F.nll_loss(pred, y).item())
                y_true.append(y)
                y_pred.append(pred.argmax(1))

    with timer.accumulate_timer('eval_train'):
        y_pred = torch.cat(y_pred).view(-1)
        y_true = torch.cat(y_true).view(-1)
        print(y_true.shape)
        correct = (y_pred == y_true).sum()
        acc = correct.item() / len(y_true)
    

    return acc, np.mean(loss_list)


def eval(args, model, device, loader, mode):
    y_true = []
    y_pred = []
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            with timer.accumulate_timer('eval_other'+mode):
                pred = model(batch)
                y, mask = to_dense_batch(batch.y, batch.batch)
                y = y[:,0,0]
                loss_list.append(F.nll_loss(pred, y).item())
                y_true.append(y)
                y_pred.append(pred.argmax(1))


    with timer.accumulate_timer('eval_other'+mode):
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        pred_list = []
        for i in torch.split(y_pred, args.num_data_augment, dim=0):
            pred_list.append(i.bincount().argmax().unsqueeze(0))
        y_pred = torch.cat(pred_list)
        y_true = y_true.view(-1, args.num_data_augment)[:, 0]
        correct = (y_pred == y_true).sum()
        acc = correct.item() / len(y_true)


    return acc, np.mean(loss_list)

def random_split(data_list, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(len(data_list))
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * len(data_list))]
    val_idx = all_idx[int(frac_train * len(data_list)):int((frac_train+frac_valid) * len(data_list))]
    test_idx = all_idx[int((frac_train+frac_valid) * len(data_list)):]
    train_list = []
    test_list = []
    val_list = []
    for i in train_idx:
        train_list.append(data_list[i])
    for i in val_idx:
        val_list.append(data_list[i])
    for i in test_idx:
        test_list.append(data_list[i])
    return train_list, val_list, test_list

def main():

    torch.cuda.reset_max_memory_allocated()

    parser = argparse.ArgumentParser(description='PyTorch implementation of graph transformer')
    parser.add_argument('--dataset_name', type=str, default='ogbn_arxiv')
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=6)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_data_augment', type=int, default=1)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--perturb_feature', type=bool, default=False)
    parser.add_argument('--weight_update_period', type=int, default=10000, help='epochs to update the sampling weight')
    net_params={}
    net_params['layer_norm'] = False
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['lap_pos_enc'] = True
    net_params['wl_pos_enc'] = False
    net_params['pos_enc_dim'] = 2
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    #data_list = 
    data = torch.load('./dataset/'+args.dataset_name+'/data.pt')
    data_trans = T.Compose([T.AddRemainingSelfLoops()])
    data = data_trans(data)
    feature = data.x

    print(data)
    y = data.y
    
    train_id, test_id, valid_id = random_split(np.arange(data.num_nodes), frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_id] = True
    test_mask[test_id] = True
    valid_mask[valid_id] = True
    print(train_mask)
    print('dataset load successfully')

    fanout = [4]

        
    from torch_geometric.loader import NeighborLoader
    loader_trans = []
    if net_params['lap_pos_enc']:
        loader_trans.append(T.AddLaplacianEigenvectorPE(net_params['pos_enc_dim'], attr_name='lap_pos_enc'))
    loader_trans = T.Compose(loader_trans)
    train_loader = NeighborLoader(
            data,
            input_nodes=train_mask,
            disjoint=True,
            num_neighbors=fanout,  # 两层采样，分别采样25和10个邻居
            batch_size=args.batch_size,             # 每个子图一个中心节点
            shuffle=True,
            transform=loader_trans,
        )
    test_loader = NeighborLoader(
            data,
            input_nodes=test_mask,
            disjoint=True,
            num_neighbors=fanout,  # 两层采样，分别采样25和10个邻居
            batch_size=args.batch_size,             # 每个子图一个中心节点
            shuffle=False,            # 保持原始顺序
            transform=loader_trans,
        )
    val_loader = NeighborLoader(
            data,
            input_nodes=valid_mask,
            disjoint=True,
            num_neighbors=fanout,  # 两层采样，分别采样25和10个邻居
            batch_size=args.batch_size,             # 每个子图一个中心节点
            shuffle=False,            # 保持原始顺序
            transform=loader_trans,
        )
    print(args)

    model = GraphTransformerNet(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=feature.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=y.max().item()+1,
        dropout=args.dropout_rate,
        in_feat_dropout=args.intput_dropout_rate,
        net_params=net_params
    )
    if not args.test and not args.validate:
        print(model)
    print('Total params:', sum(p.numel() for p in model.parameters()))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_epochs,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)

    val_acc_list, test_acc_list = [], []
    sampling_weight = np.ones(4)
    weight_history = []
    p_min = 0.05
    p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min
    timelist=[]
    # prof = torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,  # 记录 CPU 活动
    #         torch.profiler.ProfilerActivity.CUDA  # 记录 CUDA 活动
    #     ],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'+'Gophormer'),
    #     record_shapes=True,  # 记录每个操作的形状信息
    #     profile_memory=True,  # 记录内存使用情况
    #     with_stack=True  # 记录函数调用栈
    # )
    # prof.start()

    for epoch in range(1, args.epochs+1):
        


        print("====epoch " + str(epoch))

        with timer.timer('all_train'):
            train(args, model, device, train_loader, optimizer, lr_scheduler, prof=None)
            lr_scheduler.step()

        print("====Evaluation")
        with timer.timer('all_evaluation'):
            train_acc, train_loss = eval_train(args, model, device, train_loader)

            val_acc, val_loss = eval(args, model, device, val_loader, mode='val')
            test_acc, test_loss = eval(args, model, device, test_loader, mode='test')

        print("train_acc: %f val_acc: %f test_acc: %f" % (train_acc, val_acc, test_acc))
        print("train_loss: %f val_loss: %f test_loss: %f" % (train_loss, val_loss, test_loss))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        timer.reduce_time()

    # prof.stop()
    print(torch.cuda.max_memory_allocated(device=device)/1024.0/1024.0)
    timer.print_time()
    print('best validation acc: ', max(val_acc_list))
    print('best test acc: ', max(test_acc_list))
    print('best acc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
    # np.save('./exps/'+args.dataset_name+'/weight_history', np.array(weight_history))
    # np.save('./exps/' + args.dataset_name + '/test_acc_list', np.array(test_acc_list))
    # np.save('./exps/' + args.dataset_name + '/val_acc_list', np.array(val_acc_list))


if __name__ == "__main__":
    main()