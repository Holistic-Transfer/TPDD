from abc import abstractmethod
import os
from collections import defaultdict
from functools import reduce
from datetime import datetime
import torch

from ..util import constant as C


class File:
    def __init__(self, path, epoch=-1):
        self.path = path
        self.epoch = epoch

    def __lt__(self, other):
        return self.epoch < other.epoch

    def __eq__(self, other):
        return self.epoch == other.epoch

    def __gt__(self, other):
        return self.epoch > other.epoch

    def __le__(self, other):
        return self.epoch <= other.epoch

    def __ge__(self, other):
        return self.epoch >= other.epoch


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
        if not os.path.exists(self.path):
            os.makedirs(self.path)

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


class Experiment(Serialization):
    _CHECKPOINT_PREFIX = ['model', 'feature', 'logit', 'y', 'y_pred']
    _DEFAULT_PREFIX = _CHECKPOINT_PREFIX[0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = None
        self._file_path = {}
        self._construct()

    @property
    def source_model_path(self):
        return os.path.join(self.source_path, 'source.pth')

    @property
    def log_path(self):
        _log_path = self._file_path['log']
        return _log_path

    def _search_log_path(self):
        latest_file = None
        latest_date = None
        for f in os.listdir(self.path):
            if f.startswith('train-'):
                date_str = f.split('.txt')[0].split('train-')[1]  # Extract the date from the filename
                date = datetime.strptime(date_str, '%Y-%m-%d-%H_%M_%S')  # Convert the date string to a datetime object
                if latest_date is None or date > latest_date:
                    latest_date = date
                    latest_file = f
        return latest_file

    def _sort(self):
        for prefix in Experiment._CHECKPOINT_PREFIX:
            if prefix not in self._file_path:
                continue
            prefix_checkpoint_path = self._file_path[prefix]
            prefix_checkpoint_path.sort()
            self._file_path[prefix] = prefix_checkpoint_path

    def _construct(self):
        # Construct checkpoint files
        for prefix in Experiment._CHECKPOINT_PREFIX:
            prefix_path = os.path.join(self.path, prefix)
            if not os.path.exists(prefix_path):
                os.makedirs(prefix_path)
            for f in os.listdir(prefix_path):
                path = os.path.join(prefix_path, f)
                epoch = int(f.split('.')[0])
                self.add_checkpoint(prefix, path, epoch)
        # TODO: Refactor log file
        # Construct log file
        _log_path = self._search_log_path()
        if _log_path is not None:
            _log_path = os.path.join(self.path, _log_path)
        self._file_path['log'] = _log_path

    def epochs(self, prefix=None):
        if prefix is None:
            prefix = 'model'
        assert prefix in self._CHECKPOINT_PREFIX, f'Unknown prefix: {prefix}. '
        if prefix not in self._file_path:
            return []
        else:
            return [f.epoch for f in self._file_path[prefix]]

    def checkpoint(self, prefix, epoch=None):
        assert prefix in self._CHECKPOINT_PREFIX, f'Unknown prefix: {prefix}. '
        if epoch is None:
            _checkpoint = self._file_path[prefix]
            return _checkpoint
        else:
            _epochs = self.epochs(prefix)
            if epoch not in _epochs:
                return None
            ind = _epochs.index(epoch)
            _checkpoint = self._file_path[prefix][ind]
            return _checkpoint

    def checkpoint_path(self, prefix=None):
        if prefix is None:
            prefix = Experiment._DEFAULT_PREFIX
        assert prefix in Experiment._CHECKPOINT_PREFIX, f'Unknown prefix: {prefix}. '
        _checkpoint_path = os.path.join(self.path, prefix)
        return _checkpoint_path

    def latest_checkpoint_path(self, prefix=None):
        if prefix is None:
            prefix = Experiment._DEFAULT_PREFIX
        assert prefix in Experiment._CHECKPOINT_PREFIX, f'Unknown prefix: {prefix}. '
        if prefix not in self._file_path:
            return None
        latest_checkpoint = self._file_path[prefix][-1]
        latest_checkpoint_path = latest_checkpoint.path
        return latest_checkpoint_path

    def add_checkpoint(self, prefix, path, epoch):
        assert isinstance(epoch, int), f'Epoch should be an integer. Got {epoch}'
        assert prefix in self._CHECKPOINT_PREFIX, f'Unknown prefix: {prefix}. '
        if prefix not in self._file_path:
            self._file_path[prefix] = []
        checkpoint = File(path, epoch)
        self._file_path[prefix].append(checkpoint)
        self._sort()

    def save_checkpoint(self, epoch, data, prefix):
        filename = f'{epoch}.pth'
        path = os.path.join(self.checkpoint_path(prefix), filename)
        self.add_checkpoint(prefix, path, epoch)
        torch.save(data, path)

    def load_checkpoint(self, epoch, prefix):
        # Epoch is number
        assert isinstance(epoch, int), f'Epoch should be an integer. Got {epoch}'
        path = self.checkpoint(prefix, epoch)
        return torch.load(path)

    def load_latest_checkpoint(self, prefix=None):
        if prefix is None:
            prefix = Experiment._DEFAULT_PREFIX
        assert prefix in Experiment._CHECKPOINT_PREFIX, f'Unknown prefix: {prefix}. '
        path = self.latest_checkpoint_path(prefix)
        _checkpoint = torch.load(path)
        return _checkpoint

    # def load_source_model(self):
    #     _source_model_path = self.source_model_path
    #     _model = torch.load(_source_model_path)
    #     return _model

    @property
    def source_model_path(self):
        _path = os.path.join(self.source_path, 'source.pth')
        return _path


class FeatureExtraction(Serialization):
    pass


class MetricResult(Serialization):
    pass


class PlotResult(Serialization):
    pass


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


# TODO: Refactor using generic types
class ExperimentSpace(SerializationSpace):
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = C.EXPERIMENT_PATH
        super().__init__(base_path)
        self._construct()

    def instance_from_path(self, path):
        return Serialization.from_path(Experiment, self.base_path, path)

    def new(self, *args, **kwargs):
        instance = Experiment(self.base_path, *args, **kwargs)
        self.add(instance)
        return instance


class PlotResultSpace(SerializationSpace):
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = C.PLOT_RESULT_PATH
        super().__init__(base_path)

    def instance_from_path(self, path):
        return Serialization.from_path(PlotResult, self.base_path, path)


class FeatureExtractionSpace(SerializationSpace):
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = C.FEATURE_EXTRACTION_PATH
        super().__init__(base_path)

    def instance_from_path(self, path):
        return Serialization.from_path(FeatureExtraction, self.base_path, path)


class MetricResultSpace(SerializationSpace):
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = C.METRIC_RESULT_PATH
        super().__init__(base_path)

    def instance_from_path(self, path):
        return Serialization.from_path(MetricResult, self.base_path, path)
