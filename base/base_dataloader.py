import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers,
                 train_indxs = None, val_indxs = None, val = False):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        if val: self.train_sampler, self.val_sampler = (
            self._split_sampler(train_indxs, val_indxs))
        else: self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, train_indxs, val_indxs):
        self.nbr_examples = len(train_indxs)
        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        #self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)

class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break