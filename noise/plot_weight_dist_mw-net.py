# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np

from wideresnet import WideResNet, VNet
from resnet import ResNet32,VNet
from load_corrupted_data import CIFAR10, CIFAR100

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=102)
parser.add_argument('--prefetch', type=int, default=6, help='Pre-fetching threads.')
parser.set_defaults(augment=True)
parser.add_argument('--cuda', type=str, default='', help='cuda visible device')
parser.add_argument('--model', type=str, default='ResNet', help='model ResNet or WideResNet')
parser.set_defaults(augment=True)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

torch.manual_seed(args.seed)
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

print()
print(args)

# model and vnet path
if args.dataset == 'cifar10':
    results_path = './results/cifar10/'
elif args.dataset == 'cifar100':
    results_path = './results/cifar100/'
else:
    print('error')
    exit(1)

model_path = results_path + 'models/model_state_corruptionProb{}_corruptionType{}_seed{}_last.pth'.format(args.corruption_prob, args.corruption_type, args.seed)
vnet_path = results_path + 'models/vnet_state_corruptionProb{}_corruptionType{}_seed{}_last.pth'.format(args.corruption_prob, args.corruption_type, args.seed)
weight_path = results_path + 'models/weight_corruptionProb{}_corruptionType{}_seed{}.npy'.format(args.corruption_prob, args.corruption_type, args.seed)

train_labels = results_path + 'train_labels/targets_corruptionProb{}_corruptionType{}.npy'.format(args.corruption_prob, args.corruption_type)


def build_dataset():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, train_labels=train_labels)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed, train_labels=train_labels)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True, train_labels=train_labels)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, train_labels=train_labels)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed, train_labels=train_labels)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True, train_labels=train_labels)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader


def build_model(type='ResNet'):
    if type == 'ResNet':
        model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    elif type == 'WideResNet':
        model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       args.widen_factor, dropRate=args.droprate)
    else:
        print('error')
        exit(1)
    # weights_init(model)

    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.params()])))

    if torch.cuda.is_available():
        model.cuda()
        # torch.backends.cudnn.benchmark = True

    return model


def plot_distribution(weights_clean, weights_noise, corruptionProb, corruptionType, i):

    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.figure(figsize=(6, 6.5))

    default_bins = 60
    bins = int(default_bins * (max(np.concatenate((weights_noise, weights_clean))) - min(
        np.concatenate((weights_noise, weights_clean)))))
    print(bins)

    bins = 6
    plt.hist(weights_noise, bins=bins, facecolor="blue", edgecolor="blue", alpha=1.0, rwidth=0.4, label="noisy")
    plt.hist(weights_clean, bins=bins, facecolor="red", edgecolor="red", alpha=1.0, rwidth=0.4,label="clean")

    x_major_locator = MultipleLocator(0.1)
    # y_major_locator = MultipleLocator(500)
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为0.1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)  # 把x轴的主刻度设置为500的倍数

    plt.xlim([0,1])
    # 显示横轴标签
    plt.xlabel("Weight", fontsize=10)
    # 显示纵轴标签
    plt.ylabel("Numbers", fontsize=10)
    # 显示图标题
    plt.grid(alpha=0.8,linestyle=':')

    corruptionProb = int(corruptionProb * 100)
    # plt.title("CIFAR-10_{}% {} noise li={:4f}".format(corruptionProb, corruptionType, li))

    plt.legend(loc='upper center', fontsize=15)

    plot_path = results_path + 'plots/weights_distribution_corruptionProb{}_corruptionType{}_seed{}_epoch{}.png'.format(
        args.corruption_prob, args.corruption_type, args.seed, i)

    plt.savefig(plot_path)
    plt.show()
    plt.close()

train_loader, train_meta_loader, test_loader = build_dataset()

# create model
model = build_model(args.model)
if torch.cuda.is_available():
    vnet = VNet(1, 100, 1).cuda()
else:
    vnet = VNet(1, 100, 1)

def main():

    # load train_labels
    train_labels_c = np.load(train_labels)
    train_labels_n = np.array(train_loader.dataset.train_labels_0)

    weights_record = np.load(weight_path)

    for i in [59]:
        weights_clean = []
        weights_noise = []

        for j in range(train_labels_c.shape[0]):
            if train_labels_c[j] == train_labels_n[j]:
                weights_clean.append(weights_record[j][i])
            else:
                weights_noise.append(weights_record[j][i])

        weights_clean = np.array(weights_clean)
        weights_noise = np.array(weights_noise)

        plot_distribution(weights_clean, weights_noise, args.corruption_prob, args.corruption_type, i)

    print(2)

if __name__ == '__main__':
    main()
