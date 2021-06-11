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
import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np
import copy

from torch.distributions import Categorical
from wideresnet import WideResNet, VNet
from resnet import ResNet32,VNet, VCNN, VCNN2
from load_corrupted_data import CIFAR10_2, CIFAR100, make_dirs

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=40, type=int,
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
parser.add_argument('--np_seed', type=int, default=202002)
parser.add_argument('--prefetch', type=int, default=6, help='Pre-fetching threads.')
parser.add_argument('--cuda', type=str, default=' ', help='cuda visible device')
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

make_dirs(results_path)

model_path = results_path + 'models/model_state_with_vnet_corruptionProb{}_corruptionType{}_seed{}.pth'.format(args.corruption_prob, args.corruption_type, args.seed)
txt = results_path +  'results/result_with_vnet_corruptionProb{}_corruptionType{}_seed{}.txt'.format(args.corruption_prob, args.corruption_type, args.seed)
train_labels = results_path + 'train_labels/no_meta/targets_corruptionProb{}_corruptionType{}.npy'.format(args.corruption_prob, args.corruption_type)

# vnet trained with MW net
vnet_path_ls = results_path + 'models/vnet_state_corruptionProb{}_corruptionType{}_seed{}.pth'.format(args.corruption_prob, args.corruption_type, args.seed)

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

        train_data = CIFAR10_2(
            root='../data', train=True, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed, train_labels=train_labels)
        test_data = CIFAR10_2(root='../data', train=False, transform=test_transform, download=True, train_labels=train_labels)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, train_labels=train_labels)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed, train_labels=train_labels)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True, train_labels=train_labels)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, test_loader

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate_resnet(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 40)) * (0.1 ** int(epochs >= 50)))  # For ResNet32
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_wideresnet(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 36)) * (0.1 ** int(epochs >= 38)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy

def train(train_loader, model, vnet,optimizer_model,epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0
    indexes = []
    losses2save = []
    # entropy2save = []
    # losses2save_intensified = []
    # losses2save_intensified_1 = []
    num_batches = 0

    for batch_idx, (inputs, targets, index) in enumerate(train_loader):
        indexes.append(index)
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        ### to get entropy
        # entropy = Categorical(probs=outputs).entropy()
        # entropy = torch.reshape(entropy, (len(entropy),))
        # tmp = entropy.cpu().detach().numpy()
        # entropy2save.append(tmp)

        cost_w = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]
        cost_tmp = cost_w.cpu().detach().numpy()
        losses2save.append(cost_tmp)

        with torch.no_grad():
            w = vnet(cost_v)

        loss = torch.sum(cost_v * w) / len(cost_v)
        # loss = torch.sum(cost_w)/len(cost_w)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()

        num_batches += 1
        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, 0))
    # return np.concatenate(indexes), np.concatenate(losses2save), np.concatenate(entropy2save), np.concatenate(losses2save_intensified)
    return np.concatenate(indexes), np.concatenate(losses2save), train_loss / num_batches, meta_loss / num_batches

train_loader, test_loader = build_dataset()
# create model
model = build_model(args.model)


if torch.cuda.is_available():
    vnet = VNet(1, 100, 1).cuda()
else:
    vnet = VNet(1, 100, 1)

if use_cuda:
    vnet.load_state_dict(torch.load(vnet_path_ls))
else:
    vnet.load_state_dict(torch.load(vnet_path_ls, map_location=torch.device('cpu')))


if args.dataset == 'cifar10':
    num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100

optimizer_model = torch.optim.SGD(model.params(), args.lr,
                              momentum=args.momentum, nesterov=args.nesterov,
                              weight_decay=args.weight_decay)

if args.model == 'ResNet':
    adjust_learning_rate = adjust_learning_rate_resnet
elif args.model == 'WideResNet':
    adjust_learning_rate = adjust_learning_rate_wideresnet
else:
    print('error')
    exit(1)

def main():
    best_acc = 0

    loss_recorder = np.zeros((len(train_loader.dataset), args.epochs))
    # entropy_recorder = np.zeros((len(train_loader.dataset), args.epochs))
    # loss_recorder_intensified = np.zeros((len(train_loader.dataset), args.epochs))
    # loss_recorder_intensified_1 = np.zeros((len(train_loader.dataset), args.epochs))

    train_loss_recorder = np.zeros((args.epochs, ))
    meta_loss_recorder = np.zeros((args.epochs, ))
    test_acc_recorder = np.zeros((args.epochs, ))

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        indexes, losses, train_loss, meta_loss = train(train_loader,model, vnet,optimizer_model,epoch)
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)

        loss_recorder[indexes, epoch] = losses
        # entropy_recorder[indexes, epoch] = entropy
        # loss_recorder_intensified[indexes, epoch] = losses_intensified
        # loss_recorder_intensified_1[indexes, epoch] = losses_intensified_1
        train_loss_recorder[epoch] = train_loss
        meta_loss_recorder[epoch] = meta_loss
        test_acc_recorder[epoch] = test_acc

    print('best accuracy:', best_acc)

    with open(txt, "w") as f:
            f.write("best accuracy: {}".format(best_acc))

if __name__ == '__main__':
    main()
