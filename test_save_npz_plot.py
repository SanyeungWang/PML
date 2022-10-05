import argparse
import csv
import os

import data as data
import loss
import numpy as np
import torch
from PIL import Image
from config import cfg
from model.resnet_tsne import resnet34
from option import args
from torch.utils.data import dataloader
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description='Age Estimator')
    parser.add_argument('--config', type=str, default='path_of_yml_file',   # please add your yml file path here.
                        help='config file path')
    parser.add_argument("--local_rank", type=int, default=0)  # this distributed variable can only be given by args
    args, unknown = parser.parse_known_args()
    cfg.merge_from_file(args.config)  # merge config file args
    cfg.merge_from_list(unknown)  # merge command args
    cfg.freeze()
    return args


def test():
    # This is for drawing and testing
    args = parse_args()
    print(args)

    val_dataset = data.Data(cfg, mode="val").dataset        # draw train or val feature
    val_loader = dataloader.DataLoader(val_dataset,
                                       batch_size=cfg.train.train_batch_size,
                                       num_workers=cfg.model.nThread,
                                       )

    model = resnet34(pretrained=True)
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 101)
    device = torch.device('cuda')
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    state = torch.load('path_of_pth_file',  # please add your pth file path here.
                       map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
    model.eval()

    prototype = np.zeros([101, 512], dtype=np.float32)
    instance_num = np.zeros([101, 1], dtype=np.float32)
    intra = np.zeros([101, 1], dtype=np.float32)
    inter = np.zeros([101, 101], dtype=np.float32)
    pro = [prototype, instance_num]
    with torch.no_grad():
        for i_iter, batch in enumerate(val_loader):
            # print(batch)
            images, labels, age, name, aa, bb = batch
            images = images.to(device)
            age = age.to(device)
            outputs, pro, intra, inter, z = model(images, age, pro, intra, inter)

            ages = torch.sum(outputs * torch.Tensor([i for i in range(101)]).cuda(), dim=1)
            error = abs(ages - age).cpu().clone().detach().numpy()
            print('current mae {}'.format(np.mean(error)))
            if i_iter == 0:
                temp = error
                temp_name = list(name)
                z_all = z
                age_all = age.cpu().clone().detach().numpy()
            else:
                temp = np.concatenate([temp, error])
                age_all = np.concatenate([age_all, age.cpu().clone().detach().numpy()])
                z_all = np.concatenate([z_all, z])
                temp_name.extend(name)

        np.savez('pth_to_save_feature.npz',
                 feature=z_all, label=age_all)
        mae = np.mean(temp)
        print('final mae {}'.format(mae))
        textfile = open('b.txt', 'w')
        for i, image_name in enumerate(temp_name):
            error_ = temp[i]
            if error_ < 8:
                continue
            textfile.write('{} {}'.format(image_name, error_))
            textfile.write('\n')
        textfile.close()


if __name__ == '__main__':
    test()
