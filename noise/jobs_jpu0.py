# -*- coding: utf-8 -*-
# @Time    : 2021/6/7 13:23
# @Author  : yuhuawei
# @FileName: jpu0.py
# @Mail: yuhuawei2020@gmail.com
import os

py_file = [
    'BaseModel.py --dataset cifar100 --corruption_prob 0.4 --corruption_type unif --model WideResNet --epochs 40 --seed 1 --cuda 0',
    'BaseModel.py --dataset cifar100 --corruption_prob 0.6 --corruption_type unif --model WideResNet --epochs 40 --seed 1 --cuda 0',
    'BaseModel.py --dataset cifar100 --corruption_prob 0.2 --corruption_type flip2 --model ResNet --epochs 60 --seed 1 --cuda 0',
    'BaseModel.py --dataset cifar100 --corruption_prob 0.4 --corruption_type flip2 --model ResNet --epochs 60 --seed 1 --cuda 0',

    'BaseModel.py --dataset cifar100 --corruption_prob 0.4 --corruption_type unif --model WideResNet --epochs 40 --seed 3 --cuda 0',
    'BaseModel.py --dataset cifar100 --corruption_prob 0.6 --corruption_type unif --model WideResNet --epochs 40 --seed 3 --cuda 0',
    ]

for f in py_file:
    os.system('python ' + f)