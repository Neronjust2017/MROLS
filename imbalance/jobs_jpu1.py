# -*- coding: utf-8 -*-
# @Time    : 2021/6/7 13:23
# @Author  : yuhuawei
# @FileName: jpu0.py
# @Mail: yuhuawei2020@gmail.com
import os

py_file = [
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 1 --seed 1 --cuda 1',
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 1 --seed 2 --cuda 1',
    'BaseModel.py --dataset cifar10 --num_classes 10 --num_meta 100 --imb_factor 1 --seed 3 --cuda 1',

    ]

for f in py_file:
    os.system('python ' + f)