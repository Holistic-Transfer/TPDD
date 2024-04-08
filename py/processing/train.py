from typing import DefaultDict
import numpy as np

import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..data import common as data_util
from ..util import math as math_util


class PartialDomainTrainer:

    def __init__(self, model, optimizer, loss_type, loss_scope, device):
        self.model = model.to(device)
        self._model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.device = device
        self.loss_type = loss_type
        self.loss_scope = loss_scope
        self.state = None
        self.f_loss = None
        self.training_loader = None
        self.training_iterator = None
        self.val_loader = {}
        self.training_config = None
        self._inited = False

    @staticmethod
    def _f_loss(loss_type, loss_scope, domain_info):
        # Unpack domain info
        visible_classes = domain_info.visible_classes
        num_classes = domain_info.num_classes
        # Define loss function
        if loss_type == 'cross-entropy':
            _f = F.cross_entropy
        # Define loss scope
        if loss_scope == 'all':
            def _loss(logits, y):
                return _f(logits, y)
        elif loss_scope == 'seen':
            visible_mask = torch.zeros(num_classes, dtype=torch.bool)
            visible_mask[visible_classes] = 1

            def _loss(logits, y):
                logits = logits[visible_mask]
                y = y[visible_mask]
                return _f(logits, y)
        return _loss

    def set_training_config(self, training_config):
        assert not self._inited, 'Trainer has already been initialized. '
        assert self.training_loader is not None, 'training_loader must be set before setting training_config. '
        self.training_config = training_config
        # Initialize state
        state = {}
        epochs = training_config['epochs']
        iterations = training_config['iterations']
        state['epochs'] = epochs
        state['iterations'] = iterations
        state['next_epoch'] = 0
        state['next_iteration'] = 0
        lr_scheduler = CosineAnnealingLR(self.optimizer, epochs * iterations)
        state['lr_scheduler'] = lr_scheduler
        self.state = state
        # Initialize loss function
        training_loader = self.training_loader
        training_data = training_loader.dataset
        domain_info = training_data.domain_info
        _loss = self._f_loss(self.loss_type, self.loss_scope, domain_info)
        self.f_loss = _loss
        # Set _inited flag
        self._inited = True

    def add_val_loader(self, k, loader):
        dataset = loader.dataset
        assert isinstance(dataset, data_util.PartialDomainDataset), 'Only PartialDomainDataset is supported. '
        self.val_loader[k] = loader

    def set_training_loader(self, loader):
        assert self.training_loader is None, 'training_loader has already been set. '
        dataset = loader.dataset
        assert isinstance(dataset, data_util.PartialDomainDataset), 'Only PartialDomainDataset is supported. '
        self.training_loader = loader
        self.training_iterator = data_util.ForeverDataIterator(loader)
        self.val_loader['Train'] = loader

    def _training_iteration(self):
        self.model.train()
        self.training_loader.dataset.train()
        optimizer = self.optimizer
        training_iterator = self.training_iterator
        X, y = next(training_iterator)
        X = X.to(self.device)
        y = y.to(self.device)
        logits, _, y, _ = self.extract_batch(X, y)
        loss = self.f_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _f_batch_postprocess(self, cosine_sim):
        if cosine_sim:
            _w = self.model.head.weight.data.clone()
            _b = self.model.head.bias.data.clone()
            _p = torch.cat([_w, _b.view(-1, 1)], dim=1)
            _p = F.normalize(_p, dim=1)
            norm_weight = _p[:, :-1]
            bias = _p[:, -1]

            def _f(_, features):
                features = F.normalize(features, dim=1)
                logits = torch.matmul(features, norm_weight.T) + bias
                return features, logits
        else:
            _f = None
        return _f

    def extract_batch(self, X, y, f_postprocess=None):
        if f_postprocess is None:
            f_postprocess = lambda x, y: (x, y)
        X = X.to(self.device)
        logits, features = self.model(X, return_feat=True)
        logits, features = f_postprocess(logits, features)
        y_pred = logits.argmax(dim=1)
        return logits, features, y, y_pred

    def extract(self, loader, cosine_sim=False):
        f_postprocess = self._f_batch_postprocess(cosine_sim)
        features = []
        logits = []
        y = []
        y_pred = []
        for _X, _y in loader:
            _logits, _features, _y, _y_pred = self.extract_batch(_X, _y, f_postprocess)
            features.append(_features.cpu().numpy())
            logits.append(_logits.cpu().numpy())
            y.append(_y.cpu().numpy())
            y_pred.append(_y_pred.cpu().numpy())
        features = np.concatenate(features, axis=0)
        logits = np.concatenate(logits, axis=0)
        y = np.concatenate(y, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        return features, logits, y, y_pred

    def evaluate(self, loader=None):
        self.model.eval()
        if loader is None:
            loader = self.val_loader
        result = DefaultDict(dict)
        for _k, _loader in loader.items():
            _loader.dataset.eval()
            with torch.no_grad():
                _features, _logits, _y, _y_pred = self.extract(_loader)
            _accuracy = math_util.topk_accuracy(_logits, _y)
            result[_k]['accuracy'] = _accuracy
            result[_k]['features'] = _features
            result[_k]['logits'] = _logits
            result[_k]['y'] = _y
            result[_k]['y_pred'] = _y_pred
        return result

    def evaluate_and_save(self):
        # Overwrite this method to save evaluation results
        return self.evaluate()

    def fit(self):
        assert self._inited, 'Trainer has not been initialized. '
        state = self.state
        epochs = state['epochs']
        iterations = state['iterations']
        training_config = self.training_config
        eval_freq = training_config['evaluate_freq']
        eval_every = int(iterations * eval_freq)
        for epoch in range(state['next_epoch'], epochs):
            for iteration in range(state['next_iteration'], iterations):
                self._training_iteration()
                state['next_iteration'] = iteration + 1
                if iteration % eval_every == 0:
                    eval_result = self.evaluate()
                    for _k, _r in eval_result.items():
                        accuracy = _r['accuracy']
                        print(f'Epoch {epoch}, iteration {iteration}, {_k} accuracy: {accuracy}')
            self.evaluate_and_save()
            state['next_iteration'] = 0
            state['next_epoch'] = epoch + 1
