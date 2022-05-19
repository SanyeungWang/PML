import os
import csv
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from torch.utils.data.distributed import DistributedSampler
# from utils import list_pictures
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import dataloader


class Data:
    def __init__(self, cfg, mode):
        self.cfg = cfg
        width = cfg.train.width
        height = cfg.train.height
        transform_list = [
            transforms.RandomChoice(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(),
                 transforms.RandomRotation(20),
                 transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                 ]
            ),
            transforms.Resize([height, width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        transform_val = [transforms.RandomHorizontalFlip(p=0),
                         transforms.Resize([height, width]),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ]
        transform = transforms.Compose(transform_list)
        transform_v = transforms.Compose(transform_val)
        self.dataset = Dataset(cfg, transform, mode, transform_v)


class Dataset(dataset.Dataset):
    def __init__(self, cfg, transform, mode, trans_v):
        self.mode = mode
        self.trans_v = trans_v
        if mode == "train":
            self.root = cfg.dataset.train_img
            self.label_path = cfg.dataset.train_label
        elif mode == "val":
            self.root = cfg.dataset.val_img
            self.label_path = cfg.dataset.val_label
        self.transform = transform
        images = []
        label = []
        with open(self.label_path, 'r') as f:
            csv_read = csv.reader(f)
            for line in csv_read:
                images.append(line[0])
                label.append(line[1])
        self.labels = label[1:]  # not include column names
        self.images = images[1:]
        self.loader = default_loader

    def __getitem__(self, index):
        name = self.images[index]
        age = self.labels[index]
        img = self.loader(os.path.join(self.root, name))
        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        age = int(age)
        if self.transform is not None and self.mode == 'train':
            img = self.transform(img)
        if self.trans_v is not None and self.mode == 'val':  # The test set only needs to be flipped horizontally
            img = self.trans_v(img)

        return img, label, age, name

    def __len__(self):
        return len(self.labels)


def decompose(name, img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[i] = img[i]*std[i]+mean[i]
    img1 = img*255
    img1 = img1.numpy().transpose(1, 2, 0)
    img1 = img1.astype(dtype=np.uint8)
    plt.imshow(img1)
    plt.show()
    return img1


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)
