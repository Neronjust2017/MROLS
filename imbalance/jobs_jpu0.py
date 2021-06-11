# -*- coding: utf-8 -*-
# @Time    : 2021/6/7 13:23
# @Author  : yuhuawei
# @FileName: jpu0.py
# @Mail: yuhuawei2020@gmail.com
import os

py_file = [
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.005 --seed 1 --cuda 0',
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.005 --seed 2 --cuda 0',
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.005 --seed 3 --cuda 0',

    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.2 --seed 1 --cuda 0',
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.2 --seed 2 --cuda 0',
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 0.2 --seed 3 --cuda 0',

    ]

for f in py_file:
    os.system('python ' + f)