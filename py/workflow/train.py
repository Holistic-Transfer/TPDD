import copy
import torch
import logging
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data import common as data_util
from ..util import evaluate


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
        self.val_loaders = {}
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
        self.val_loaders[k] = loader

    def set_training_loader(self, loader):
        assert self.training_loader is None, 'training_loader has already been set. '
        dataset = loader.dataset
        assert isinstance(dataset, data_util.PartialDomainDataset), 'Only PartialDomainDataset is supported. '
        self.training_loader = loader
        self.training_iterator = data_util.ForeverDataIterator(loader)
        # self.val_loader['Train'] = loader

    def _training_iteration(self):
        self.model.train()
        dataset = self.training_loader.dataset
        dataset.train()
        dataset.set_scope('visible')
        optimizer = self.optimizer
        training_iterator = self.training_iterator
        X, y = next(training_iterator)
        X = X.to(self.device)
        y = y.to(self.device)
        extraction = self.extract_batch(X, y)
        logits = extraction.logits
        labels = extraction.labels
        loss = self.f_loss(logits, labels)
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
        y = y.to(self.device)
        logits, features = self.model(X, return_feat=True)
        logits, features = f_postprocess(logits, features)
        extraction = evaluate.Extraction(features, logits, y)
        return extraction

    def extract(self, loader, cosine_sim=False):
        f_postprocess = self._f_batch_postprocess(cosine_sim)
        features = []
        logits = []
        labels = []
        for _X, _y in loader:
            _extraction = self.extract_batch(_X, _y, f_postprocess)
            _features = _extraction.features
            _logits = _extraction.logits
            _labels = _extraction.labels
            features.append(_features)
            logits.append(_logits)
            labels.append(_labels)
        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        extraction = evaluate.Extraction(features, logits, labels)
        return extraction

    def evaluate(self):
        evaluation_result = {}
        # Validation on training data
        training_loader = self.training_loader
        training_dataset = training_loader.dataset
        domain_info = training_dataset.domain_info
        model = self.model
        model.eval()
        training_dataset.eval()
        # Extract training oracle features
        training_dataset.set_scope('all')
        logging.debug('Validating on training oracle data...')
        with torch.no_grad():
            oracle_training_extraction = self.extract(training_loader)
            oracle_training_evaluation = evaluate.evaluate(domain_info, oracle_training_extraction,
                                                           oracle_training_extraction)
        evaluation_result['oracle_training'] = oracle_training_evaluation
        # Extract training features
        training_dataset.set_scope('visible')
        logging.debug('Validating on training data... ')
        with torch.no_grad():
            training_extraction = self.extract(training_loader)
            training_evaluation = evaluate.evaluate(domain_info, training_extraction, oracle_training_extraction)
        evaluation_result['training'] = training_evaluation
        # Validation on validation data
        val_loaders = self.val_loaders
        for k, val_loader in val_loaders.items():
            logging.debug(f'Validating on {k} data... ')
            val_dataset = val_loader.dataset
            val_domain_info = val_dataset.domain_info
            val_dataset.eval()
            val_dataset.set_scope('all')
            # Extract validation features
            with torch.no_grad():
                val_extraction = self.extract(val_loader)
                val_evaluation = evaluate.evaluate(val_domain_info, val_extraction, oracle_training_extraction)
            evaluation_result[k] = val_evaluation
        return evaluation_result

    def evaluate_and_save(self):
        # Overwrite this method to save evaluation results
        return self.evaluate()

    def fit(self):
        assert self._inited, 'Trainer has not been initialized. '
        logging.info(f'Starting training, training_config: {self.training_config}... ')
        logging.info('Initializing... ')
        state = self.state
        epochs = state['epochs']
        iterations = state['iterations']
        training_config = self.training_config
        eval_freq = training_config['evaluate_freq']
        eval_every = max(1, int(iterations * eval_freq))
        logging.debug(f'Evaluation frequency: {eval_every}. ')
        for epoch in range(state['next_epoch'], epochs):
            for iteration in range(state['next_iteration'], iterations):
                logging.debug(f'Epoch {epoch}, iteration {iteration}... ')
                # Training iteration
                self._training_iteration()
                # Post iteration
                state['next_iteration'] = iteration + 1
                state['lr_scheduler'].step()
                # Evaluation
                if iteration % eval_every == 0:
                    eval_result = self.evaluate()
                    for _k, _r in eval_result.items():
                        logging.info(f'Epoch {epoch}, iteration {iteration}, {_k} metrics: \n{_r}')
            # TODO: Refactor this line
            self.evaluate_and_save()
            state['next_iteration'] = 0
            state['next_epoch'] = epoch + 1