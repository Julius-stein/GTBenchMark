import torch.nn as nn
from GTBenchmark.graphgym.config import cfg
from GTBenchmark.graphgym.register import register_loss


@register_loss('l1')
def l1_losses(pred, true):

    l1_loss = nn.L1Loss()
    loss = l1_loss(pred, true)
    return loss, pred

@register_loss('smoothl1')
def smoothl1(pred,true):

    l1_loss = nn.SmoothL1Loss()
    loss = l1_loss(pred, true)
    return loss, pred
