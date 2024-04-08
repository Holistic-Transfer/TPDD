from torch.utils.data import DataLoader, Sampler, Dataset

import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader

from typing import Optional, Callable, Tuple, Any, List

import os
from typing import List


class DomainInfo:
    def __init__(self, all_classes, visible_classes, invisible_classes=None, num_classes=None):
        self.all_classes = all_classes
        self.visible_classes = visible_classes
        if invisible_classes is None:
            invisible_classes = list(set(all_classes) - set(visible_classes))
        if num_classes is None:
            num_classes = len(all_classes)
        self.invisible_classes = invisible_classes
        self.num_classes = num_classes


class PartialDomainDataset(Dataset):

    def __init__(self, dataset: Dataset, domain_info: DomainInfo):
        self.dataset = dataset
        self.domain_info = domain_info
        _validation_dataset = self._check_dataset_format(dataset)
        assert _validation_dataset, 'PartialDomainDataset only supports dataset with (data, target) format. '
        self._ind = None
        self._visible_ind = None
        self._invisible_ind = None
        self._all_ind = None
        self._init_indices()

    def _check_dataset_format(self, dataset):
        _sample = dataset[0]
        valid = isinstance(_sample, tuple) and len(_sample) == 2
        return valid

    def _init_indices(self):
        domain_info = self.domain_info
        visible_classes = domain_info.visible_classes
        invisible_classes = domain_info.invisible_classes
        _visible_ind = []
        _invisible_ind = []
        _all_ind = []
        for i, (_, _c) in enumerate(self.dataset):
            if _c in visible_classes:
                _visible_ind.append(i)
            elif _c in invisible_classes:
                _invisible_ind.append(i)
            _all_ind.append(i)
        self._visible_ind = _visible_ind
        self._invisible_ind = _invisible_ind
        self._all_ind = _all_ind
        self._ind = self._visible_ind

    def __getitem__(self, index):
        return self.dataset[self._ind[index]]

    def __len__(self):
        return len(self._ind)

    def set_scope(self, scope):
        assert scope in ['visible', 'invisible', 'all'], f'Invalid scope: {scope}. '
        if scope == 'visible':
            self._ind = self._visible_ind
        elif scope == 'invisible':
            self._ind = self._invisible_ind
        elif scope == 'all':
            self._ind = self._all_ind

    def train(self):
        if hasattr(self.dataset, 'train'):
            self.dataset.train()

    def eval(self):
        if hasattr(self.dataset, 'eval'):
            self.dataset.eval()


"""
Modified from: https://github.com/thuml/Transfer-Learning-Library 

@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com

Copyright (c) 2018 The Python Packaging Authority
"""


class ForeverDataIterator:

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.dataloader)
