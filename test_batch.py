import torch
import csv
import numpy as np
import os
from option import args
from PIL import Image
from torchvision import transforms
from model.resnet_tsne import resnet34
import data
from config import cfg
import loss
import argparse
from torch.utils.data import dataloader

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Age Estimator')
    parser.add_argument('--config', type=str, default='./configs/morph/exp_margin.yml',
                        help='config file path')
    parser.add_argument("--local_rank", type=int, default=0)  # This distributed variable can only be given by args
    args, unknown = parser.parse_known_args()
    cfg.merge_from_file(args.config)  # merge config file args
    cfg.merge_from_list(unknown)  # merge common args
    cfg.freeze()
    return args


def test():
    args = parse_args()
    print(args)

    val_dataset = data.Data(cfg, mode="train").dataset
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
    state = torch.load('./out/morph/morph_imdb_womargin_Resnet34/2.840/best.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
    model.eval()

    prototype = np.zeros([101, 512], dtype=np.float32)
    instance_num = np.zeros([101, 1], dtype=np.float32)
    intra = np.zeros([101, 1], dtype=np.float32)
    inter = np.zeros([101, 101], dtype=np.float32)
    pro = [prototype, instance_num]
    with torch.no_grad():
        for i_iter, batch in enumerate(val_loader):
            images, labels, age, name = batch
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

        np.savez('./visualization/morph/tail_train_baseline_feature.npz',
                 feature=z_all, label=age_all)
        mae = np.mean(temp)
        print('final mae {}'.format(mae))
        textfile = open('a.txt', 'w')
        for i, image_name in enumerate(temp_name):
            error_ = temp[i]
            if error_ < 8:
                continue
            textfile.write('{} {}'.format(image_name, error_))
            textfile.write('\n')
        textfile.close()


if __name__ == '__main__':
    test()
