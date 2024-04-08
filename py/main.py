import torch
from torch.utils.data import DataLoader
import random
import numpy as np

from .data import common as data_api
from .data import dataset
from .serialization.local import ExperimentSpace
from .util import cmd as cmd_util
from .util import model as model_util
from .processing import train


def get_visible_classes(dataset, source, target, n_seen_classes):
    from . import CLASS_ORDER
    if dataset == 'OfficeHome':
        if source == 'Ar':
            clzes = CLASS_ORDER.OFFICEHOME_SOURCE_AR
        elif source == 'Rw':
            clzes = CLASS_ORDER.OFFICEHOME_SOURCE_RW
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return clzes, clzes[:n_seen_classes]


class HolisticTransfer:

    def __init__(self, experiment, trainer, serialization_config):
        self.experiment = experiment
        self.trainer = trainer
        self.serialization_config = serialization_config
        self.wrapped = False

    def _f_evaluate_and_save(self):
        experiment = self.experiment

        def _f(_self):
            result = _self.evaluate()
            for _, r in result.items():
                features = r['features']
                logits = r['logits']
                y = r['y']
                y_pred = r['y_pred']
                epoch = _self.state['next_epoch']
                experiment.save_checkpoint(epoch, features, 'feature')
                experiment.save_checkpoint(epoch, logits, 'logit')
                experiment.save_checkpoint(epoch, y, 'y')
                experiment.save_checkpoint(epoch, y_pred, 'y_pred')
            experiment.save_checkpoint(epoch, _self.model.state_dict(), 'model')

        return _f

    def _wrap_evaluate_and_save(self):
        if self.wrapped:
            return
        trainer = self.trainer
        f = self._f_evaluate_and_save()
        trainer.evaluate_and_save = f.__get__(trainer, train.PartialDomainTrainer)
        self.wrapped = True

    def fit(self):
        self._wrap_evaluate_and_save()
        self.trainer.fit()


def main(args):
    # Unpack arguments
    base_dir = args.base_dir
    device = args.device
    dataset_name = args.dataset
    source = args.source
    target = args.target
    seed = args.seed
    arch = args.arch
    serialization_config = args.serialization_config
    model_config = args.model_config
    n_seen_classes = args.n_seen_classes
    batch_size = args.batch_size
    workers = args.workers
    optimizer_type = args.optimizer
    optimizer_parameters = args.optimizer_parameters
    training_config = args.training_config

    # Unpack model config
    loss_type = model_config['loss_type']
    loss_scope = model_config['loss_scope']
    freeze_bn = model_config['freeze_bn']
    freeze_classifier = model_config['freeze_classifier']
    freeze_backbone = model_config['freeze_backbone']
    dropout = model_config['dropout']

    # Set random seed
    if seed is not None:
        print(f'Setting random seed to {seed}... ')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Construct experiment space
    space = ExperimentSpace(base_dir)

    # Create experiment instance
    experiment = space.new(dataset=dataset_name,
                           arch=arch,
                           source=source,
                           target=target,
                           model_config=model_config,
                           n_seen_classes=n_seen_classes,
                           optimizer=optimizer_type,
                           optimizer_parameters=optimizer_parameters,
                           seed=seed)

    # Load data
    if dataset_name == 'OfficeHome':
        training_data = dataset.officehome(target, 'training')
        testing_data = dataset.officehome(target, 'testing')

    all_classes, visible_classes = get_visible_classes(dataset_name, source, target, n_seen_classes)
    domain_info = data_api.DomainInfo(all_classes, visible_classes)
    training_data = data_api.PartialDomainDataset(training_data, domain_info)
    testing_data = data_api.PartialDomainDataset(testing_data, domain_info)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                                 drop_last=True)
    testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Init model
    source_model_path = experiment.source_model_path
    model = model_util.create_model(arch, freeze_bn, dropout, domain_info.num_classes, source_model_path)

    # Init optimizer
    optimizer = model_util.build_optimizer(model, optimizer_type, optimizer_parameters, freeze_classifier,
                                           freeze_backbone)

    # Create trainer
    trainer = train.PartialDomainTrainer(model, optimizer, loss_type, loss_scope, device)
    trainer.set_training_loader(training_loader)
    trainer.add_val_loader('Test', testing_loader)
    trainer.set_training_config(training_config)

    # Create HT instance
    ht = HolisticTransfer(experiment, trainer, serialization_config)
    ht.fit()


if __name__ == '__main__':
    args = cmd_util.parse_arguments()
    main(args)
