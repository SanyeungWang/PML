#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import math
from pathlib import Path
import time
import torch
import numpy as np
import logging
from torch import optim
from config import cfg
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_margin_params(model, distributed):
    if distributed:
        model = model.module
    margin_params = list(map(id, model.conv1_d.parameters()))
    margin_params += list(map(id, model.bn1_d.parameters()))
    margin_params += list(map(id, model.layer1_d.parameters()))
    margin_params += list(map(id, model.layer2_d.parameters()))
    margin_params += list(map(id, model.layer3_d.parameters()))
    margin_params += list(map(id, model.layer4_d.parameters()))
    margin_params += list(map(id, model.avgpool_d.parameters()))
    margin_params += list(map(id, model.fc_d.parameters()))
    return margin_params


def make_optimizer(cfg, model, lr, distributed):
    if distributed:
        ignored_params = list(map(id, model.module.fc.parameters()))
        # if cfg.pretrained:
        #     ignored_params += get_margin_params(model, distributed)
        ignored_params += get_margin_params(model, distributed)
        igno_params = filter(lambda p: id(p) in ignored_params and p.requires_grad, model.module.parameters())
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.module.parameters())
    else:
        ignored_params = list(map(id, model.fc.parameters()))
        ignored_params += get_margin_params(model, distributed)
        igno_params = filter(lambda p: id(p) in ignored_params and p.requires_grad, model.parameters())
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.parameters())
    # trainable = filter(lambda x: x.requires_grad, model.module.parameters())
    if cfg.train.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': cfg.train.SGD_params.momentum,
            'dampening': cfg.train.SGD_params.dampening,
            'nesterov': cfg.train.SGD_params.nesterov
        }
    elif cfg.train.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (cfg.train.ADAM_params.beta1, cfg.train.ADAM_params.beta2),
            'eps': cfg.train.ADAM_params.epsilon,
            'amsgrad': cfg.train.ADAM_params.amsgrad
        }
    elif cfg.train.optimizer == 'ADAMAX':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (cfg.train.ADAM_params.beta1, cfg.train.ADAM_params.beta2),
            'eps': cfg.train.ADAM_params.epsilon
        }
    elif cfg.train.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': cfg.train.ADAM_params.epsilon,
            'momentum': cfg.train.SGD_params.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = lr
    if cfg.model.pretrained:
        lr_b = lr*cfg.model.finetune_factor
    else:
        lr_b = lr
    kwargs['weight_decay'] = cfg.train.weight_decay

    return optimizer_function([
        {'params': base_params, 'lr': lr_b}, {'params': igno_params, 'lr': lr}]
        , **kwargs)


def create_logger():
    root_output_dir = Path(cfg.log.output_dir)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.model.dataset_name
    model = cfg.model.model_name

    final_output_dir = root_output_dir / Path(dataset + '_' + model)
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)

    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.log.log_dir) / dataset / model / Path('_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    if cfg.model.pretrained:
        optimizer.param_groups[0]['lr'] = lr*cfg.model.finetune_factor
    else:
        optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr  # classifier
    return lr


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def cosine_distance(x, y):
    if x.ndim == 1:
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
    elif x.ndim == 2:
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)

    np.seterr(divide='ignore', invalid='ignore')
    s = np.dot(x, y.T)/(x_norm*y_norm)
    s *= -1
    dist = s + 1
    dist = np.clip(dist, 0, 2)
    if x is y or y is None:
        dist[np.diag_indices_from(dist)] = 0.0
    if np.any(np.isnan(dist)):
        if x.ndim == 1:
            dist = 1.
        else:
            dist[np.isnan(dist)] = 1.
    return dist


# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: target learning rate = base lr * multiplier
          total_epoch: target learning rate is reached at total_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch > self.total_epoch:
            self.after_scheduler.step(epoch - self.total_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)


def warmup_scheduler(optimizer, n_iter_per_epoch):
    # The learning rate will not be reset because T_max specifies the epoch as the last
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer, eta_min=0.000001,
        T_max=(cfg.train.epochs - cfg.train.warmup_epoch) * n_iter_per_epoch)
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=cfg.train.multiplier,
        total_epoch=cfg.train.warmup_epoch * n_iter_per_epoch,
        after_scheduler=cosine_scheduler)
    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
