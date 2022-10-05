import csv
import os

import numpy as np
import torch
from PIL import Image
from model.resnet import resnet34
from option import args  # change val_img and val_label in option.py
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def preprocess(img):
    img = Image.open(img).convert('RGB')
    imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
    transform_list = [
        transforms.RandomHorizontalFlip(p=0),
        transforms.Resize((args.height, args.width), interpolation=3),  # This value will affect, 5 may better
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    imgs = [transform(i) for i in imgs]
    imgs = [torch.unsqueeze(i, dim=0) for i in imgs]
    return imgs


def test():
    model = resnet34(pretrained=True)
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 101)

    state = torch.load('path_of_pth_file')  # please add your .pth file path here.
    model.load_state_dict(state)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()  # when testing, remember model.eval()
    csv_reader = csv.reader(open(args.val_label, 'r'))
    root_path = args.val_img
    rank = torch.Tensor([i for i in range(101)]).cuda()
    error = 0
    count = 0
    errors = []
    for i in csv_reader:
        if i[0] == 'image_name':
            continue
        name, age = i
        age = torch.IntTensor([int(age)])
        img_path = os.path.join(root_path, name)
        imgs = preprocess(img_path)
        predict_age = 0
        prototype = np.zeros([101, 512], dtype=np.float32)
        instance_num = np.zeros([101, 1], dtype=np.float32)
        intra = np.zeros([101, 1], dtype=np.float32)
        inter = np.zeros([101, 101], dtype=np.float32)
        pro = [prototype, instance_num]
        for img in imgs:
            img = img.to(device)
            age = age.to(device)
            output, pro, intra, inter = model(img, age, pro, intra, inter)
            predict_age += torch.sum(output * rank, dim=1).item() / 2
        print('label:{} \tage:{:.2f}'.format(age, predict_age))
        data = open("newfile1.txt", 'a', encoding="utf-8")
        print('label:{} \tage:{:.2f}'.format(age, predict_age), file=data)

        err = abs(predict_age - age).cpu().clone().detach().numpy()
        error += err
        errors.append(err)
        # print(err)
        count += 1
        # print(error / count)
    # print(errors)
    print('final mae')
    print(error / count)


if __name__ == '__main__':
    test()
