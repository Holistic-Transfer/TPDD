import copy
import time
import logging
import torch
from torch.utils.data import DataLoader, Dataset


class DomainInfo:
    def __init__(self, all_classes, visible_classes, horizontal_visible, invisible_classes=None, num_classes=None):
        assert isinstance(all_classes, torch.Tensor), 'all_classes must be a tensor. '
        assert isinstance(visible_classes, torch.Tensor), 'visible_classes must be a tensor. '
        assert (invisible_classes is None) or (isinstance(invisible_classes, torch.Tensor)), 'invisible_classes must be a tensor. '
        self.all_classes = all_classes
        self.visible_classes = visible_classes
        self.horizontal_visible = horizontal_visible
        if invisible_classes is None:
            visible_clz_ind = torch.isin(all_classes, visible_classes)
            invisible_clz_ind = ~visible_clz_ind
            invisible_classes = all_classes[invisible_clz_ind]
        if num_classes is None:
            num_classes = len(all_classes)
        self.invisible_classes = invisible_classes
        self.num_classes = num_classes
    
    def to(self, device):
        self.all_classes = self.all_classes.to(device)
        self.visible_classes = self.visible_classes.to(device)
        self.invisible_classes = self.invisible_classes.to(device)
        if hasattr(self, 'visible_ind'):
            self.visible_ind = self.visible_ind.to(device)
        if hasattr(self, 'invisible_ind'):
            self.invisible_ind = self.invisible_ind.to(device)
        if hasattr(self, 'labels'):
            self.labels = self.labels.to(device)
        return self
    
    def __repr__(self):
        return f'DomainInfo(all_classes={self.all_classes}, visible_classes={self.visible_classes}, horizontal_visible={self.horizontal_visible}, invisible_classes={self.invisible_classes}, num_classes={self.num_classes})'


class PartialDomainDataset(Dataset):

    def __init__(self, dataset: Dataset, domain_info: DomainInfo, horizontal_all: bool = False):
        start = time.time()
        _validation_dataset = self._check_dataset_format(dataset)
        self.dataset = self._make_ordered_dataset(dataset)
        self.domain_info = copy.deepcopy(domain_info)
        self.horizontal_all = horizontal_all
        assert _validation_dataset, 'PartialDomainDataset only supports dataset with (data, target) format. '
        self._iterate_ind = None
        self.visible_ind = None
        self.invisible_ind = None
        self.all_ind = None
        self._init_indices()
        end = time.time()
        logging.info(f'PartialDomainDataset init time: {end - start:.3f} s. ')

    def _make_ordered_dataset(self, dataset):
        ordered_dataset = []
        for i, (data, target) in enumerate(dataset):
            ordered_dataset.append((i, (data, target),))
        return ordered_dataset

    def visible_mask(self, ind):
        mask = torch.isin(ind, self.visible_ind)
        return mask
    
    def invisible_mask(self, ind):
        mask = torch.isin(ind, self.invisible_ind)
        return mask

    def _check_dataset_format(self, dataset):
        _sample = dataset[0]
        valid = isinstance(_sample, tuple) and len(_sample) == 2
        return valid

    def _init_indices(self):
        domain_info = self.domain_info
        visible_classes = domain_info.visible_classes
        invisible_classes = domain_info.invisible_classes
        horizontal_visible = domain_info.horizontal_visible
        visible_ind = []
        invisible_ind = []
        all_ind = []
        labels = []
        for i, (_, _c) in self.dataset:
            if _c in visible_classes:
                if self.horizontal_all or torch.rand(1) < horizontal_visible:
                    visible_ind.append(i)
                else:
                    invisible_ind.append(i)
            elif _c in invisible_classes:
                invisible_ind.append(i)
            all_ind.append(i)
            labels.append(_c)
        self.visible_ind = visible_ind
        # Randomly sample from visible classes according to horizontal_visible
        self.invisible_ind = invisible_ind
        self.all_ind = all_ind
        assert len(self.visible_ind) + len(self.invisible_ind) == len(self.dataset), 'Visible and invisible indices do not match the dataset. '
        self._iterate_ind = self.visible_ind
        domain_info.visible_ind = torch.tensor(self.visible_ind)
        domain_info.invisible_ind = torch.tensor(self.invisible_ind)
        domain_info.labels = torch.tensor(labels)

    def __getitem__(self, index):
        return self.dataset[self._iterate_ind[index]]

    def __len__(self):
        return len(self._iterate_ind)

    def set_scope(self, scope):
        assert scope in ['visible', 'invisible', 'all'], f'Invalid scope: {scope}. '
        if scope == 'visible':
            self._iterate_ind = self.visible_ind
        elif scope == 'invisible':
            self._iterate_ind = self.invisible_ind
        elif scope == 'all':
            self._iterate_ind = self.all_ind

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
