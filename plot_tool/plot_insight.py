"""
@File  :plot_insight.py
@Author:Miles
@Date  :2020/11/109:31
@Desc  :
"""
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from numpy import *
import csv
import math
import matplotlib.ticker as ticker
from pandas import Series
from scipy import stats


class TwoNomal():
    def __init__(self, mu1, mu2, sigma1, sigma2):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2

    def doubledensity(self, x):
        mu1 = self.mu1
        sigma1 = self.sigma1
        mu2 = self.mu2
        sigma2 = self.sigma1
        N1 = np.sqrt(2 * np.pi * np.power(sigma1, 2))
        fac1 = np.power(x - mu1, 2) / np.power(sigma1, 2)
        density1 = np.exp(-fac1 / 2) / N1

        N2 = np.sqrt(2 * np.pi * np.power(sigma2, 2))
        fac2 = np.power(x - mu2, 2) / np.power(sigma2, 2)
        density2 = np.exp(-fac2 / 2) / N2
        # print(density1,density2)
        density = 0.5 * density2 + 0.5 * density1
        return density


# N2 = TwoNomal(14, 18, 1, 2)
# # Create arithmetic progression as X
# X = np.arange(-20, 120, 0.05)
# # print(X)
# Y = N2.doubledensity(X)
# # print(Y)
# plt.plot(X, Y, 'b-', linewidth=3)
#
# plt.show()
def annotate_max(x, y, index_a=0):
    x_index = np.argsort(-y)
    x1 = x_index[0]

    x_index2 = np.argsort(abs(x - 14))
    x2 = x_index2[0]

    y_target = round(y[x1], 2)
    x_target = round(x[x1], 2)
    plt.plot(x_target, y_target, 'x')
    show = '({}, {})'.format(x_target, y_target)
    plt.annotate(show, xy=(x_target, y_target), xytext=(x_target + 1, y_target), color='blue')
    plt.plot()

    y_target = round(y[x2], 2)
    x_target = round(x[x2], 2)
    plt.plot(x_target, y_target, 'ks')
    show = '({}, {})'.format(x_target, y_target)
    plt.annotate(show, xy=(x_target, y_target), xytext=(x_target + 1, y_target), color='blue')
    plt.plot()


def double_gaussian():
    comp1 = np.random.normal(14, 2.3, size=200)  # N(0,1)
    comp2 = np.random.normal(18, 1.7, size=200)  # (10,4)
    data = np.concatenate([comp1, comp2])
    x = np.linspace(3, 30, 200)
    y = stats.gaussian_kde(data)(x)
    plt.grid(color="k", linestyle=":")
    plt.xlabel('Age')
    plt.ylabel('Probability')
    annotate_max(x, y)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
    # values = Series(np.concatenate([comp1, comp2]))
    plt.rcParams['savefig.dpi'] = 300  # picture pixel
    plt.rcParams['figure.dpi'] = 300  # Resolution
    # values.hist(bins=100,alpha=0.3,color='g',normed=True)
    # values.plot(kind='kde', style='k--', colormap='Reds')
    # plt.grid(color="k", linestyle=":")
    plt.plot(x, y, 'r', linewidth=1)
    plt.show()


def sorted_csv():
    csv_cha = '../DATASETS/ChaLearn/ChaLearn15_csv/train.csv'
    with open(csv_cha, 'r') as f:
        data = csv.reader(f)
        sortedlist = sorted(data, key=lambda x: int(x[1]))

    with open('../DATASETS/ChaLearn/ChaLearn15_csv/trian_sorted_.csv', 'w') as f:
        filewriter = csv.writer(f, delimiter=',')
        for row in sortedlist:
            filewriter.writerow(row)


def annotate(index_a, x, y):
    # x_index = np.where(x == index_a)
    # x_index = x_index[0][0]
    x_index = 28
    y_target = round(y[x_index], 2)
    plt.plot(index_a, y_target, 'ks')
    show = '({}, {})'.format(index_a, y_target)
    plt.annotate(show, xy=(index_a, y_target), xytext=(index_a + 1, y_target,), color='blue')
    plt.plot()


def plt_distribution():
    csv_morph = '../DATASETS/MORPH/morph_csv/morph_all.csv'
    csv_fg = '../DATASETS/FGNET/FGNET_csv/train_all.csv'
    csv_cha = '../DATASETS/ChaLearn/ChaLearn15_csv/chalearn_all.csv'
    with open(csv_morph, 'r') as f:
        images = []
        label = []
        temp = []
        csv_reader = csv.reader(f)
        for line in csv_reader:
            images.append(line[0])
            label.append(line[1])
        labels = label[1:]
        for ms in labels:
            ms = ms.zfill(2)
            temp.append(ms)
        labels = np.array(temp)

        key = np.unique(labels)
        result = {}
        x = np.array([])
        y = np.array([])
        for k in key:
            mask = (labels == k)
            arr_new = labels[mask]
            v = arr_new.size
            result[k] = v
            y = np.append(y, v)
        for j in key:
            j = j.zfill(2)
            x = np.append(x, int(j))
        plt.rcParams['savefig.dpi'] = 300  # picture pixel
        plt.rcParams['figure.dpi'] = 300  # Resolution
        plt.plot(x, y, color="r", linestyle="-", linewidth=2)
        # plt.grid(color="k", linestyle=":")
        plt.locator_params('y', nbins=25)
        plt.ylabel('Number of images per age')
        plt.ylim(ymin=0)
        plt.xlim(xmin=15)
        plt.xlabel('Age')
        plt.title('Morph II', fontsize=12, color='black')
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(3))
        # annotate(14, x, y)
        # annotate(18, x, y)
        plt.show()
plt_distribution()
def plt_annote():
    plt.rcParams['savefig.dpi'] = 300  # picture pixel
    plt.rcParams['figure.dpi'] = 300  # Resolution
    index = [20, 24, 28, 32, 36, 40, 54, 57, 60, 63, 66, 69]
    plt.scatter(0, 0, color=plt.cm.Set1(0), label='Age 20')
    plt.scatter(0, 0, color=plt.cm.Set1(1), label='Age 24')
    plt.scatter(0, 0, color=plt.cm.Set1(2), label='Age 28')
    plt.scatter(0, 0, color=plt.cm.Set1(3), label='Age 32')
    plt.scatter(0, 0, color=plt.cm.Set1(4), label='Age 36')
    plt.scatter(0, 0, color=plt.cm.Set1(5), label='Age 40')

    # plt.scatter(0, 0, color=plt.cm.Set1(6), label='Age 54')
    # plt.scatter(0, 0, color=plt.cm.Set1(7), label='Age 57')
    # plt.scatter(0, 0, color=plt.cm.Set1(8), label='Age 60')
    # plt.scatter(0, 0, color=plt.cm.Set2(0), label='Age 63')
    # plt.scatter(0, 0, color=plt.cm.Set2(1), label='Age 66')
    # plt.scatter(0, 0, color=plt.cm.Set2(2), label='Age 69')

    plt.legend(loc='upper left', fontsize='18', markerscale=2)
    plt.savefig('2.png')
    plt.show()
plt_annote()

def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def plot_gaussion():
    mu = 14.0
    sigma = 0.2
    plt.rcParams['savefig.dpi'] = 300  # picture pixel
    plt.rcParams['figure.dpi'] = 300  # Resolution
    x1 = np.linspace(0, 101, 200)
    y1 = normal_distribution(x1, mu, sigma)
    y1 = y1 / np.linalg.norm(y1)
    plt.xlim(xmax=90)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(4))
    plt.grid(color="k", linestyle=":")
    plt.xlabel('Age')
    plt.ylabel('Probability')
    annotate(14, x1, y1)
    # plt.title('(a)', fontsize=14, color='black')
    plt.plot(x1, y1, 'r', linewidth=1)
    plt.show()
