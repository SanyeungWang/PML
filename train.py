import os
import pprint
import torch
import logging
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import data
import utils
from utils import create_logger
from model import ThinAge, TinyAge
from model.resnet import resnet50, resnet34
import torch.distributed as dist
from torch.utils.data import dataloader
from function import train, validate
from config import cfg
import argparse


# init
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'


def parse_args():
    parser = argparse.ArgumentParser(description='Age Estimator')
    parser.add_argument('--config', type=str, default='./configs/imdb_wiki/pre.yml',
                        help='config file path')
    parser.add_argument("--local_rank", type=int, default=0)  # This distributed variable can only be given by args
    args, unknown = parser.parse_known_args()
    cfg.merge_from_file(args.config)  # merge config file args
    cfg.merge_from_list(unknown)  # merge common args
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

    logger.info(pprint.pformat(cfg))

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # sign
    distributed = torch.cuda.device_count() > 1
    device = torch.device('cuda:{}'.format(args.local_rank))
    model = init_model(cfg)  # initialize model

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    # get data

    train_dataset = data.Data(cfg, mode="train").dataset

    if distributed:
        train_sampler = DistributedSampler(train_dataset)  # default shuffle
    else:
        train_sampler = None
    train_loader = dataloader.DataLoader(train_dataset,
                                         batch_size=cfg.train.train_batch_size,
                                         num_workers=cfg.model.nThread,
                                         sampler=train_sampler
                                         )
    val_dataset = data.Data(cfg, mode='val').dataset
    if distributed:
        val_sampler = DistributedSampler(val_dataset)
    else:
        val_sampler = None
    val_loader = dataloader.DataLoader(val_dataset,
                                       batch_size=cfg.train.val_batch_size,
                                       num_workers=cfg.model.nThread,
                                       sampler=val_sampler
                                       )
    # optimizer
    optimizer = utils.make_optimizer(cfg, model, cfg.train.lr, distributed)
    gpu_num = torch.cuda.device_count()
    data_num = data.Data(cfg, 'train').dataset.__len__()
    epoch_iters = np.int(data_num / cfg.train.train_batch_size / gpu_num)
    min_mae = 20
    last_epoch = 0
    if not distributed:
        model = model
    else:
        model = model.module
    if cfg.model.pretrained:  # load baseline, train margins and classifiers
        path = cfg.model.pretrained_path
        # load from .pth
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        if path[-3:] == 'tar':
            state_dict = state_dict['state_dict']
        model_dict = model.state_dict()
        # using imdb_wiki model fc
        # if path.find('imdb') >= 0:
        #     pretrained_dict = {k: v for k, v in state_dict.items()}
        # else:
        pretrained_dict = {k: v for k, v in state_dict.items() if k != 'fc.weight' and k != 'fc.bias'}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    if cfg.model.resume:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            min_mae = checkpoint['min_mae']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if cfg.train.optimizer == 'ADAM':
                optimizer.load_state_dict(checkpoint['optimizer'])  # if adam resume sgd will make error
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    end_epoch = cfg.train.epochs
    num_iters = cfg.train.epochs * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        if torch.cuda.device_count() > 1:
            train_sampler.set_epoch(epoch)
        train(cfg, epoch, cfg.train.epochs, epoch_iters, cfg.train.lr, num_iters, train_loader, optimizer, model, writer_dict,
              device)

        # validate
        mae = validate(cfg, val_loader, model, writer_dict, device)
        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + '/checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'min_mae': mae,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

            if min_mae >= mae:
                min_mae = mae
                torch.save(model.state_dict(), os.path.join(final_output_dir, 'best.pth'))
            msg = 'MIN_MAE: {:.3f}, Curr_MAE:{:.3f}'.format(min_mae, mae)
            logging.info(msg)


if __name__ == '__main__':
    main()
