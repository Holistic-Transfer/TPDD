import torch
import torch.nn as nn

import timm

from ..model import _models as models
from ..model import _ibn_models as ibn_models
from ..model import _base as base

import torch
import torch.nn as nn
from torch.optim import SGD, Adam


def free_model(model):
    del model
    torch.cuda.empty_cache()


def create_model(arch, freeze_bn, dropout, num_classes, load_model_path):
    backbone = get_model(arch)
    model = base.ImageClassifier(
        backbone, num_classes, freeze_bn=freeze_bn,
        dropout_p=dropout,
        finetune=True, pool_layer=None)

    print(f'==> Loading weights from {load_model_path}')
    checkpoint = torch.load(load_model_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint)
    print(msg)
    return model


def get_parameters(model, base_lr=1.0, finetune=False, fixedcls=False, fixed_backbone=False):
    # assert not (finetune and fixedcls), 'finetune and fixedcls cannot be True at the same time'
    print(f'finetune: {finetune}, fixedcls: {fixedcls}')
    all_params = [{
        "params": model.backbone.parameters(),
        "lr": 0.1 * base_lr if finetune else 1.0 * base_lr
    }, {
        "params": model.bottleneck.parameters(),
        "lr": 1.0 * base_lr
    }, {
        "params": model.head.parameters(),
        "lr": 1.0 * base_lr
    }]
    if fixed_backbone:
        print('##### Fixed backbone')
        _ind = [1, 2]
    elif fixedcls:
        print('##### Fixed cls')
        _ind = [0, 1]
    else:
        print('##### Full ')
        _ind = [0, 1, 2]
    params = [all_params[i] for i in _ind]
    print(params)
    return params


def get_model(model_name, pretrained=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrained)
    elif model_name in ibn_models.__dict__:
        # load models (with ibn) from tllib.normalization.ibn
        backbone = ibn_models.__dict__[model_name](pretrained=pretrained)
    elif model_name == 'lenet':
        backbone = models.LeNet()
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrained)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_parameters(m, base_lr, freeze_classifier, freeze_backbone):
    all_params = [{
        "params": m.backbone.parameters(),
        "lr": 0.1 * base_lr
    }, {
        "params": m.bottleneck.parameters(),
        "lr": 1.0 * base_lr
    }, {
        "params": m.head.parameters(),
        "lr": 1.0 * base_lr
    }]
    if freeze_classifier:
        _ind = [0, 1]
    elif freeze_backbone:
        _ind = [1, 2]
    else:
        _ind = [0, 1, 2]
    params = [all_params[i] for i in _ind]
    return params


def build_optimizer(model, optimizer, optimizer_parameters, freeze_classifier, freeze_backbone):
    assert 'lr' in optimizer_parameters, 'lr must be specified in optimizer_parameters'
    base_lr = optimizer_parameters['lr']
    params = get_parameters(model, base_lr, freeze_classifier, freeze_backbone)
    if optimizer == 'SGD':
        optimizer = SGD(params, **optimizer_parameters)
    elif optimizer == 'Adam':
        optimizer = Adam(params, **optimizer_parameters)
    else:
        raise NotImplementedError()
    print(optimizer)
    return optimizer