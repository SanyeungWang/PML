import argparse
import logging
import os
import pprint
import ssl

import data as data
import numpy as np
import torch
import torch.distributed as dist
import utils
from config import cfg
from function import train, validate
from model import ThinAge, TinyAge
from model import resnet
from model.resnet import resnet50, resnet34
from tensorboardX import SummaryWriter
from torch.utils.data import dataloader
from torch.utils.data.distributed import DistributedSampler
from utils import create_logger

ssl._create_default_https_context = ssl._create_unverified_context  # bypass SSL authentication

# init
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description='Age Estimator')
    parser.add_argument('--config', type=str,
                        default='path_of_yml_file',         # please add your .yml file path here.
                        help='config file path')
    parser.add_argument("--local_rank", type=int, default=0)  # this distributed variable can only be given by args

    args, unknown = parser.parse_known_args()
    cfg.merge_from_file(args.config)  # merge config file args
    cfg.merge_from_list(unknown)  # merge command args
    cfg.freeze()
    return args


def init_model(cfg):
    models = {'ThinAge': ThinAge, 'TinyAge': TinyAge, 'Resnet50': resnet50, 'Resnet50_pre': resnet50,
              'Resnet34_pre': resnet34, 'Resnet34': resnet34}
    model = cfg.model.model_name
    assert model in models
    if model == 'Resnet50':
        model = models[model](num_classes=101)
    elif model == 'Resnet50_pre':
        model = models[model](num_classes=101, pretrained=True)
    elif model == 'Resnet34_pre' or model == 'Resnet34':
        model = models[model](pretrained=True)
        fc_in_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_in_features, 101)
    else:
        model = models[model]()
    return model


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger()

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # sign
    distributed = torch.cuda.device_count() > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    val_dataset = data.Data(cfg, mode='val').dataset
    if distributed:
        val_sampler = DistributedSampler(val_dataset)
    else:
        val_sampler = None
    val_loader = dataloader.DataLoader(val_dataset,
                                       batch_size=cfg.train.val_batch_size,
                                       num_workers=cfg.model.nThread,
                                       sampler=val_sampler,
                                       drop_last=False
                                       )

    model = resnet34(pretrained=True)
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 101)
    state = torch.load(
        'path_of_pth_file')         # # please add your .pth file path here.
    model.load_state_dict(state)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    mae = validate(cfg, val_loader, model, writer_dict, device)
    print(mae)


if __name__ == '__main__':
    main()
