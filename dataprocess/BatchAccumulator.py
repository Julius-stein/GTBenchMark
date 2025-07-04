import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.dataset



def default_collate(batch):
    return batch

class WrapperLoader(torch.utils.data.dataset.IterableDataset):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def __iter__(self):
        return self.original_loader.__iter__()

