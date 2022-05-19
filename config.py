"""
@File  :config.py
@Author:Miles
@Date  :2020/10/2322:27
@Desc  :config file
"""
import argparse
from os.path import basename, join, splitext
from yacs.config import CfgNode

cfg = CfgNode(dict(
    log=dict(
        output_dir='./out/',
        log_dir='./log/',
        print_freq=1
    ),
    dataset=dict(
        train_img='./DATASETS/MORPH/MORPH',
        train_label='./DATASETS/MORPH/morph_csv/bak/morph_train_1.csv',
        val_img='./DATASETS/MORPH/MORPH',
        val_label='./DATASETS/MORPH/morph_csv/bak/morph_val_1.csv',
    ),
    train=dict(
        lr=3e-4,
        optimizer='ADAM',
        multiplier=16,
        warmup_epoch=50,
        weight_decay=0.1,
        SGD_params=dict(
            momentum=0.9,
            dampening=0,
            nesterov='store_true',
        ),
        ADAM_params=dict(
            beta1=0.9,
            beta2=0.999,
            amsgrad='store_true',
            epsilon=1e-8
        ),
        gamma=0.8,
        reset='store_true',
        epochs=300,
        train_batch_size=800,
        val_batch_size=800,
        height=224,
        width=224
    ),
    distributed=dict(
        rank=0,
        word_szie=6,
        init_method='env://'
    ),
    model=dict(
        dataset_name='morph',
        resume=False,                # restart or continue training
        model_name='Resnet34',
        nThread=8,
        margin=False,
        pretrained=False,
        finetune_factor=0.1,
        pretrained_path='',
    )
))



