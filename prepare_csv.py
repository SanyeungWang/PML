"""
@File  :prepare_csv.py
@Author:Miles
@Date  :2020/8/24 15:35
@Desc  :Generate train, test, and validation sets csv
"""
import os
import csv
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import os
import re
import cv2


def csv_write(csv_path, data):
    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_name', 'label'])
        writer.writerows(data)


def morph_all():
    data_root = os.listdir('MORPH')
    X = []
    for image_name in data_root:
        if os.path.splitext(image_name)[1] == '.jpg':
            label = os.path.splitext(image_name)[0][-2:]
            temp = [image_name, label]
            X.append(temp)
            print(len(X))
    return X



def imdb_all():
    data_root = os.listdir('IMDB_WIKI')
    X = []
    for image_name in data_root:
        age_end = os.path.splitext(image_name)[0][-4:]
        age_start = re.findall(r'_(.+?)-', os.path.splitext(image_name)[0])
        print(image_name)
        age_start = age_start[0][-4:]
        label = int(age_end) - int(age_start)
        if 16 <= label <= 77:
            temp = [image_name, str(label)]
            X.append(temp)
        print(len(X))
    return X


def imdb_csv():
    kf = KFold(n_splits=5, shuffle=True)
    i = 0
    X = imdb_all()
    for train_index, test_index in kf.split(X):
        X_train = []
        X_test = []
        print('split ' + str(i))
        for j in train_index:
            X_train.append(X[j])
        for j in test_index:
            X_test.append(X[j])
        csv_write('./DATASETS/IMDB_WIKI/IMDB_csv/imdb_train_%s.csv' % (str(i)), X_train)
        csv_write('./DATASETS/IMDB_WIKI/IMDB_csv/imdb_val_%s.csv' % (str(i)), X_test)
        i = i + 1


# csv_write('/raid/hliu_data/miles/DLDL-v2/IMDB_csv/IMDB_all.csv', imdb_all())


def fg_csv():
    for leave in range(82):
        X_train = []
        X_test = []
        leave = str(leave + 1)
        leave_index = leave.zfill(3)
        for fn in os.listdir('./DATASETS/FGNET/FGNET'):
            name = fn
            label = os.path.splitext(name)[0][4:6]
            subject = fn[:3]
            temp = [name, label]
            if subject != leave_index:
                X_train.append(temp)
            else:
                X_test.append(temp)
        csv_write('./DATASETS/FGNET/FGNET_csv/fg_train_%s.csv' % (str(leave)), X_train)
        csv_write('./DATASETS/FGNET/FGNET_csv/fg_val_%s.csv' % (str(leave)), X_test)
    print('ok')


def morph_csv():
    kf = KFold(n_splits=5, shuffle=True)
    i = 0
    X = morph_all()
    for train_index, test_index in kf.split(X):
        X_train = []
        X_test = []
        print('split ' + str(i))
        for j in train_index:
            X_train.append(X[j])
        for j in test_index:
            X_test.append(X[j])
        csv_write('./DATASETS/MORPH/morph_csv/morph_train_%s.csv' % (str(i)), X_train)
        csv_write('./DATASETS/MORPH/morph_csv/morph_val_%s.csv' % (str(i)), X_test)
        i = i + 1


def mv(a):
    target = 'IMDB-WIKI'
    small_size_imdb = './DATASETS/IMDB_WIKI/small_size_imdb'
    paths = os.listdir(target)
    for i, path in enumerate(paths[a:a+50000]):
        image_path = os.path.join(target, path)
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_height = img.shape[0]
        img_width = img.shape[1]
        if img_height <= 100 or img_width <= 100:
            cmd = 'mv {} {}'.format(image_path, small_size_imdb)
            os.system(cmd)
            print(cmd)


def chalearn_csv():
    path = './DATASETS/ChaLearn/ChaLearn15_csv/train.csv'
    with open(path, 'r') as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            line.split(';')


