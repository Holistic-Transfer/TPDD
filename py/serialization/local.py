from abc import abstractmethod
import logging
import os
from collections import defaultdict
from functools import reduce
from datetime import datetime
import torch

from ..util import constant as C


class CheckpointDirectory:

    def __init__(self, path):
        self.path = path
        self.epochs = []
        self.reload()

    def _check_format(self, f):
        return f.endswith('.pth') and f[:-4].isnumeric()

    def reload(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for f in os.listdir(self.path):
            if not self._check_format(f):
                continue
            epoch = int(f[:-4])
            self.epochs.append(epoch)
        self.epochs.sort()

    def add(self, epoch):
        if epoch in self.epochs:
            return
        self.epochs.append(epoch)
        self.epochs.sort()

    def save(self, ckpt, epoch):
        logging.debug(f'Adding epoch {epoch} to {self.path}')
        self.add(epoch)
        logging.debug(f'Saving checkpoint to {self._epoch_file(epoch)}')
        torch.save(ckpt, self._epoch_file(epoch))

    def epochs(self):
        return self.epochs
    
    def _epoch_file(self, epoch):
        return os.path.join(self.path, f'{epoch}.pth')

    def epoch_file(self, epoch):
        assert epoch in self.epochs, f'Epoch {epoch} not found in {self.path}'
        return self._epoch_file(epoch)

    def epoch_files(self):
        return [os.path.join(self.path, f'{epoch}.pth') for epoch in self.epochs]

    def __str__(self):
        return f'Checkpoint Dir Epochs: {self.epochs}'

class PathTree:

    def __init__(self, base_path):
        self.base_path = base_path
        self.dict_tree = {}
        self._construct()
    
    def _construct(self):
        for root, dirs, _ in os.walk(self.base_path):
            if root == self.base_path or dirs:
                continue
            self.add(root)

    def _rel_hierarchy(self, path):
        _relpath = os.path.relpath(path, self.base_path)
        _rel_hierarchy = _relpath.split(os.sep)
        return _rel_hierarchy

    def add(self, path):
        _path = self._rel_hierarchy(path)
        _tree = self.dict_tree
        for p in _path[:-1]:
            if p not in _tree:
                _tree[p] = {}
            _tree = _tree[p]
        if _path[-1] in _tree:
            assert isinstance(_tree[_path[-1]], CheckpointDirectory), f'Path {path} is not leaf dir.'
            ckpt_dir = _tree[_path[-1]]
        else:
            ckpt_dir = CheckpointDirectory(path)
            _tree[_path[-1]] = ckpt_dir
        return ckpt_dir
            
    def get(self, path):
        _path = self._rel_hierarchy(path)
        _tree = self.dict_tree
        for p in _path:
            _tree = _tree[p]
        return _tree

    def __str__(self):
        return str(self.tree)


class Serialization:
    def __init__(self, base_path, dataset, arch, source, target, n_seen_classes, model_config, optimizer,
                 optimizer_parameters, seed):
        self.base_path = base_path
        self.dataset = dataset
        self.arch = arch
        self.source = source
        self.target = target
        self.n_seen_classes = str(n_seen_classes)
        self.model_config = str(model_config)
        self.optimizer = optimizer
        self.optimizer_parameters = str(optimizer_parameters)
        self.seed = str(seed)
        self.files = None

    @property
    def exists(self):
        return os.path.exists(self.path)
    
    @property
    def path(self):
        return os.path.join(self.base_path, self.dataset, self.arch, self.source, self.target, self.n_seen_classes,
                            self.model_config, self.optimizer, self.optimizer_parameters, self.seed)

    @property
    def source_path(self):
        return os.path.join(self.base_path, self.dataset, self.arch, self.source)

    @staticmethod
    def parse_path(p):
        s = os.path.normpath(p).split(os.sep)
        dataset = s[0]
        arch = s[1]
        source = s[2]
        target = s[3]
        n_seen_classes = s[4]
        model_config = s[5]
        optimizer = s[6]
        optimizer_parameters = s[7]
        seed = s[8]
        return dataset, arch, source, target, n_seen_classes, model_config, optimizer, optimizer_parameters, seed

    @staticmethod
    def from_path(CLS, base_path, path):
        dataset, arch, source, target, n_seen_classes, model_config, optimizer, optimizer_parameters, seed = Serialization.parse_path(
            path)
        return CLS(base_path, dataset, arch, source, target, n_seen_classes, model_config, optimizer,
                   optimizer_parameters, seed)

    def create(self):
        assert not self.exists, f'Path {self.path} already exists. '
        os.makedirs(self.path)

    def __str__(self):
        return f'Dataset: {self.dataset}, Arch: {self.arch}, Source: {self.source}, Target: {self.target}, N Seen Classes: {self.n_seen_classes}, Model Config: {self.model_config}, Optimizer: {self.optimizer}, Optimizer Parameters: {self.optimizer_parameters}, Seed: {self.seed}'


class Experiment(Serialization):

    _DEFAULT_PREFIX = 'model'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = PathTree(self.path)
        self.tree._construct()

    def set_logging(self, debug):
        if logging.getLogger().hasHandlers():
            return
        
        if debug:
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        logger = logging.getLogger()
        logger.setLevel(level)

        fh = logging.FileHandler(self.log_path, mode='w')
        fh.setLevel(level)
        
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        formatter = logging.Formatter(C.LOG_FORMAT)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        logger.info(f'Experiment start time: {datetime.now()}')
        logger.info(f'Experiment path: {self.path}')
        logger.info(f'Experiment info: {self}')

    def abs_path(self, path):
        return os.path.join(self.path, path)

    @property
    def log_path(self):
        return self.abs_path('log.txt')

    def epochs(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_PREFIX
        return self.tree.get(prefix).epochs()
        
    def get_checkpoint(self, prefix=None, epoch=None):
        if epoch is None:
            epoch = self.epochs(prefix)[-1]
        return self.tree.get(prefix).epoch_file(epoch)

    def save_checkpoint(self, prefix, epoch, ckpt):
        path = self.abs_path(prefix)
        ckpt_dir = self.tree.add(path)
        ckpt_dir.save(ckpt, epoch)

    @property
    def source_model_path(self):
        _path = os.path.join(self.source_path, 'source.pth')
        return _path


class SerializationSpace:
    def __init__(self, base_path):
        self.instances = defaultdict(lambda: defaultdict(set))
        self.base_path = base_path
        self._construct()

    @abstractmethod
    def instance_from_path(self, path):
        pass

    def add(self, instance):
        self.instances['dataset'][instance.dataset].add(instance)
        self.instances['optimizer'][instance.optimizer].add(instance)
        self.instances['model_config'][instance.model_config].add(instance)
        self.instances['arch'][instance.arch].add(instance)
        self.instances['source'][instance.source].add(instance)
        self.instances['optimizer_parameters'][instance.optimizer_parameters].add(instance)
        self.instances['seed'][instance.seed].add(instance)
        self.instances['target'][instance.target].add(instance)

    def add_from_path(self, path):
        instance = self.instance_from_path(path)
        self.add(instance)
        return instance

    def find(self, key, value):
        if not isinstance(key, list):
            key = [key]
        if not isinstance(value, list):
            value = [value]
        instances_arr = []
        for k, v in zip(key, value):
            _instances = self.instances[k][v]
            instances_arr.append(_instances)
        instances = reduce(set.intersection, map(set, instances_arr))
        instances = list(instances)
        return instances

    def _construct(self):
        for p, _, _ in os.walk(self.base_path):
            _rel_p = os.path.relpath(p, self.base_path)
            if len(_rel_p.split(os.sep)) == 9:
                self.add_from_path(_rel_p)


class ExperimentSpace(SerializationSpace):
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = C.EXPERIMENT_PATH
        super().__init__(base_path)
        self._construct()

    def instance_from_path(self, path):
        return Serialization.from_path(Experiment, self.base_path, path)
    
    def start(self, dataset, arch, source, target, model_config, n_seen_classes, optimizer, optimizer_parameters, seed, debug):
        instance = Experiment(self.base_path, dataset, arch, source, target, n_seen_classes, model_config, optimizer, optimizer_parameters, seed)
        if not instance.exists:
            instance.create()
        self.add(instance)
        instance.set_logging(debug)
        return instance
