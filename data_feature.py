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
    def __init__(self, args, mode):
        self.args = args
        transform_list = [transforms.ToTensor()]
        transform = transforms.Compose(transform_list)
        self.dataset = Dataset_feature(args, transform, mode)


class Dataset_feature(dataset.Dataset):
    def __init__(self, args, transform, mode):
        if mode == "train":
            self.root = args.train_feature
            self.label_path = args.train_label
        elif mode == "val":
            self.root = args.val_feature
            self.label_path = args.val_label
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
        name = self.images[index][:-4]+'.npy'  # Get the feature corresponding to img
        age = self.labels[index]
        img = np.load(os.path.join(self.root, name))
        img = torch.from_numpy(img)  # numpy->tensor
        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)
        age = int(age)
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


def normal_sampling(mean, label_k, std=1):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)
