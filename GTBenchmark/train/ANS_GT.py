import copy
import logging
import math
import random
import time

from tqdm import tqdm

import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from functools import partial

from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import mask_to_index, index_to_mask

from utils.collator import collator, Gophormer_collator#???
from dataprocess.ANS_GT import node_sampling, process_data#???
from models.GT import GT#???

from GTBenchmark.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.loader import create_loader, get_loader
from GTBenchmark.graphgym.loss import compute_loss
from GTBenchmark.graphgym.register import register_train
from GTBenchmark.graphgym.model_builder import create_model
from GTBenchmark.graphgym.optimizer import create_optimizer, create_scheduler
from GTBenchmark.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from GTBenchmark.graphgym.utils.comp_budget import params_count
from GTBenchmark.train.custom_train import eval_epoch
from GTBenchmark.utils import cfg_to_dict, flatten_dict, make_wandb_name, new_optimizer_config, new_scheduler_config
from GTBenchmark.timer import runtime_stats_cuda, is_performance_stats_enabled, enable_runtime_stats, disable_runtime_stats


def get_reward(model, loader, p):
    device = torch.device(cfg.device)
    column_normalized_adj = sp.load_npz('./dataset/' + args.dataset_name + '/column_normalized_adj.npz')
    normalized_adj = sp.load_npz('./dataset/'+args.dataset_name+'/normalized_adj.npz')
    data_x = torch.load('./dataset/' + args.dataset_name + '/x.pt')
    normalized_adj1 = normalized_adj*normalized_adj
    eigen_adj = 0.15 * inv((sp.eye(normalized_adj.shape[0]) - (1 - 0.15) * column_normalized_adj).toarray())
    eigen_adj1 = normalized_adj.toarray()
    eigen_adj2 = normalized_adj1.toarray()
    x = normalize(data_x, dim=1)
    eigen_adj3 = np.array(torch.matmul(x, x.transpose(1, 0)))
    r = [[], [], [], []]
    reward = np.zeros(4)
    model.eval()
    n_node = 10
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            scores = model(batch, get_score=True)
        scores = scores[:, 1:n_node]
        for i, score in enumerate(torch.split(scores, args.num_data_augment, dim=0)):
            ids = batch.ids[i*args.num_data_augment:(i+1)*args.num_data_augment]
            id = ids[0, 0].cpu().item()
            ids = ids[:, 1:n_node]
            s = eigen_adj[id]
            s1 = eigen_adj1[id]
            s2 = eigen_adj2[id]
            s3 = eigen_adj3[id]
            s[id], s1[id], s2[id], s3[id] = 0, 0, 0, 0
            s = torch.tensor(np.maximum(s, 0)).to(device)
            s = s/(s.sum()+1e-5)
            s1 = torch.tensor(np.maximum(s1, 0)).to(device)
            s1 = s1/(s1.sum()+1e-5)
            s2 = torch.tensor(np.maximum(s2, 0)).to(device)
            s2 = s2/(s2.sum()+1e-5)
            s3 = torch.tensor(np.maximum(s3, 0)).to(device)
            s3 = s3 / (s3.sum() + 1e-5)
            phi = p[0]*s + p[1]*s1 + p[2]*s2 + p[3]*s3 + 1e-5
            r[0].append(torch.sum(score * s[ids] / phi[ids]) / args.num_data_augment)
            r[1].append(torch.sum(score * s1[ids] / phi[ids]) / args.num_data_augment)
            r[2].append(torch.sum(score * s2[ids] / phi[ids]) / args.num_data_augment)
            r[3].append(torch.sum(score * s3[ids] / phi[ids]) / args.num_data_augment)
    reward[0] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[0]])).cpu().numpy()
    reward[1] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[1]])).cpu().numpy()
    reward[2] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[2]])).cpu().numpy()
    reward[3] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[3]])).cpu().numpy()
    return reward

def train_ANS_epoch(cur_epoch, logger, loader, model, optimizer, scheduler, batch_accumulation):
    pbar = tqdm(total=len(loader), disable=not cfg.train.tqdm)
    pbar.set_description(f'Train epoch')

    model.train()

    runtime_stats_cuda.start_epoch()

    runtime_stats_cuda.start_region("total")
    runtime_stats_cuda.start_region(
        "sampling", runtime_stats_cuda.get_last_event())
    iterator = iter(loader)
    runtime_stats_cuda.end_region("sampling")
    runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())

    if cfg.model.type == 'LPModel': # Handle label propagation specially
        # We don't need to train label propagation
        time_start = time.time()
        batch = next(iterator, None)
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=0,
                            dataset_name=cfg.dataset.name)
        pbar.update(1)
        return

    optimizer.zero_grad()
    it = 0
    time_start = time.time()
    # with torch.autograd.set_detect_anomaly(True):
    while True:
        try:
            torch.cuda.empty_cache() 
            runtime_stats_cuda.start_region(
                "total", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.start_region(
                "sampling", runtime_stats_cuda.get_last_event())
            # print('Crashed?')
            batch = next(iterator, None)
            # print('Crashed?')
            it += 1
            if batch is None:
                runtime_stats_cuda.end_region("sampling")
                runtime_stats_cuda.end_region(
                    "total", runtime_stats_cuda.get_last_event())
                break
            runtime_stats_cuda.end_region("sampling")

            runtime_stats_cuda.start_region("data_transfer", runtime_stats_cuda.get_last_event())
            if isinstance(batch, Data) or isinstance(batch, HeteroData):
                batch.split = 'train'
                batch.to(torch.device(cfg.device))
            else: # NAGphormer, HINo
                batch = [x.to(torch.device(cfg.device)) for x in batch]
            runtime_stats_cuda.end_region("data_transfer")

            runtime_stats_cuda.start_region("train", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.start_region("forward", runtime_stats_cuda.get_last_event())
            pred, true = model(batch)
            runtime_stats_cuda.end_region("forward")
            runtime_stats_cuda.start_region("loss", runtime_stats_cuda.get_last_event())
            if cfg.model.loss_fun == 'curriculum_learning_loss':
                loss, pred_score = compute_loss(pred, true, cur_epoch)
            else:
                loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            runtime_stats_cuda.end_region("loss")

            runtime_stats_cuda.start_region("backward", runtime_stats_cuda.get_last_event())
            loss.backward()
            runtime_stats_cuda.end_region("backward")
            # print(loss.detach().cpu().item())
            # check_grad(model)
            # Parameters update after accumulating gradients for given num. batches.
            if ((it + 1) % batch_accumulation == 0) or (it + 1 == len(loader)):
                if cfg.optim.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                cfg.optim.clip_grad_norm_value)
                optimizer.step()
                optimizer.zero_grad()
            runtime_stats_cuda.end_region("train")
            runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())
            cfg.params = params_count(model)
            logger.update_stats(true=_true,
                                pred=_pred,
                                loss=loss.detach().cpu().item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                dataset_name=cfg.dataset.name)
            pbar.update(1)
            time_start = time.time()
        except RuntimeError as e:
            if "cannot sample n_sample <= 0 samples" in str(e):
                print(f"Skipping batch due to error: {e}")
                continue
            else:
                # If it's a different error, re-raise it
                raise
    
    runtime_stats_cuda.end_epoch()
    runtime_stats_cuda.report_stats(
        {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train', \
        'attention': 'Attention', 'gt-layer': 'GT-Layer', 'forward': 'Forward', 'loss': 'Loss', 'backward': 'Backward'})


@register_train('ANS_GT')
def ANS_train(loggers, loaders, model, optimizer, scheduler):
    """
    ANS-GT training pipeline.  Update Dataloader periodly.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    # ANS-GT EXP4 init
    sampling_weight = np.ones(4)
    weight_history = []
    p_min = 0.05
    p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        # enable_runtime_stats()
        train_ANS_epoch(cur_epoch, loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        # disable_runtime_stats()
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch, start_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        # ANS-GT EXP4
        if cur_epoch % cfg.train.ans_weight_update == 0:
            r = get_reward(model,loaders[1], p)
            print('reward:', r)
            sampling_weight = sampling_weight*np.exp(2.0*(r+0.01/p))
            p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min
            print('p:', p)
            weight_history.append(p)
            data_list, feature = node_sampling(p)
            # train_dataset, valid_dataset, test_dataset = random_split(data_list, frac_train=0.6, frac_valid=0.2,
            #                                                           frac_test=0.2, seed=args.seed)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      collate_fn=partial(collator, feature=feature, shuffle=True,
                                                         perturb=args.perturb_feature))
            val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    collate_fn=partial(collator, feature=feature, shuffle=False))
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     collate_fn=partial(collator, feature=feature, shuffle=False))


        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch, start_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in %s', cfg.run_dir)
    



    