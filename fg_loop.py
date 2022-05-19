"""
@File  :fg_loop.py
@Author:Miles
@Date  :2020/10/239:46
@Desc  : 82 train_val pairs in fgnet
"""
import os

def loop_five(start):
    process_path = './train.py'
    env = '/home/*/data/anaconda3/envs/*/bin/python'
    for i in range(start, start+5):
        i = i+1
        index = str(i)
        train_label = './DATASETS/FGNET/FGNET_csv/fg_train_{}.csv'.format(index)
        val_label = './DATASETS/FGNET/FGNET_csv/fg_val_{}.csv'.format(index)
        dataset_name = 'fg_{}_margin'.format(index)

        cmd = '{} {} --train_label {} --val_label {} --dataset_name {}'.format(
            env, process_path, train_label, val_label, dataset_name)
        os.system(cmd)


if __name__ == '__main__':
    loop_five(2)
