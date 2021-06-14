import os
import time
import argparse
# import random
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as tbc
import torchvision
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from data_utils import build_dataset, get_img_num_per_cls, make_dirs
from resnet import ResNet32, VNet, VCNN
np.random.seed(202002)

# parse arguments
parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=float, default=0.1)
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--cuda', type=str, default='0', help='cuda visible device')
args = parser.parse_args()
print(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

# model and vnet path
if args.dataset == 'cifar10':
    results_path = './results/cifar10/'
elif args.dataset == 'cifar100':
    results_path = './results/cifar100/'
else:
    print('error')
    exit(1)

make_dirs(results_path)

# model and vnet path
loss_history_path = results_path + 'loss_sequence/losses_imbFactor{}_average3.npy'.format(args.imb_factor)
model_path = results_path + 'models/model_state_loss_sequence_imb_factor{}_seed{}.pth'.format(args.imb_factor, args.seed)
vnet_path = results_path + 'models/vnet_state_loss_sequence_imb_factor{}_seed{}.pth'.format(args.imb_factor, args.seed)
weight_path = results_path + 'models/weight_loss_sequence_imb_factor{}_seed{}.npy'.format(args.imb_factor, args.seed)
txt = results_path +  'results/result_loss_sequence_imb_factor{}_seed{}.txt'.format(args.imb_factor, args.seed)

model_path_mid = results_path + 'models/model_state_loss_sequence_imb_factor{}_seed{}_mid.pth'.format(args.imb_factor, args.seed)
vnet_path_mid = results_path + 'models/vnet_state_loss_sequence_imb_factor{}_seed{}_mid.pth'.format(args.imb_factor, args.seed)

model_path_last = results_path + 'models/model_state_loss_sequence_imb_factor{}_seed{}_last.pth'.format(args.imb_factor, args.seed)
vnet_path_last = results_path + 'models/vnet_state_loss_sequence_imb_factor{}_seed{}_last.pth'.format(args.imb_factor, args.seed)

kwargs = {'num_workers': 6, 'pin_memory': True}
use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

train_data_meta,train_data,test_dataset = build_dataset(args.dataset,args.num_meta)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

# make imbalanced data
# np.random.seed(args.seed)
torch.manual_seed(args.seed)
tbc.deterministic = True
tbc.benchmark = False
classe_labels = range(args.num_classes)

data_list = {}

for j in range(args.num_classes):
    data_list[j] = [i for i, label in enumerate(train_loader.dataset.targets) if label == j]

img_num_list = get_img_num_per_cls(args.dataset,args.imb_factor,args.num_meta*args.num_classes)
print(img_num_list)
print(sum(img_num_list))
im_data = {}
idx_to_del = []
for cls_idx, img_id_list in data_list.items():
    np.random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    im_data[cls_idx] = img_id_list[img_num:]
    idx_to_del.extend(img_id_list[img_num:])

print(len(idx_to_del))

imbalanced_train_dataset = copy.deepcopy(train_data)
imbalanced_train_dataset.targets = np.delete(train_loader.dataset.targets, idx_to_del, axis=0)
imbalanced_train_dataset.data = np.delete(train_loader.dataset.data, idx_to_del, axis=0)
imbalanced_train_loader = torch.utils.data.DataLoader(
    imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


validation_loader = torch.utils.data.DataLoader(
    train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

best_prec1 = 0

# early stopping
# early_stopping = EarlyStopping(patience=15, verbose=True, model_path=model_path, vnet_path=vnet_path)

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = build_model()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    # vnet = VNet(1, 100, 1).cuda()
    vnet = VCNN(16, 1, ksize=3).cuda()
    loss_history = np.load(loss_history_path)
    loss_history = np.expand_dims(loss_history, 1)
    loss_history = torch.from_numpy(loss_history)
    loss_history = to_var(loss_history)
    loss_history = loss_history.to(dtype=torch.float)

    optimizer_c = torch.optim.SGD(vnet.params(), 1e-5,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    # learning rate scheduler
    # lr_scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, factor=0.1,patience=10, mode='max')
    # lr_scheduler_c = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, factor=0.1,patience=10, mode='max')

    # cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    loss_recorder = np.zeros((len(imbalanced_train_dataset), args.epochs))
    weight_recorder = np.zeros((len(imbalanced_train_dataset), args.epochs))
    for epoch in range(args.epochs):

        if epoch == args.epochs / 2:
            torch.save(model.state_dict(), model_path_mid)
            torch.save(vnet.state_dict(), vnet_path_mid)

        adjust_learning_rate(optimizer_a, epoch + 1)

        indexes, weights = train(imbalanced_train_loader, validation_loader,model,vnet, optimizer_a, optimizer_c,epoch, loss_history)
        weight_recorder[indexes, epoch] = weights

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion, epoch)

        # learning rate scheduler
        # lr_scheduler_a.step(prec1, epoch)
        # lr_scheduler_c.step(prec1, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            torch.save(model.state_dict(), model_path)
            torch.save(vnet.state_dict(), vnet_path)

        # early_stopping
        # early_stopping(-prec1, model, vnet)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    print('Best accuracy: ', best_prec1)
    torch.save(model.state_dict(), model_path_last)
    torch.save(vnet.state_dict(), vnet_path_last)

    # np.save('losses.npy', loss_recorder)
    np.save(weight_path, weight_recorder)
    with open(txt, "w") as f:
        f.write("best accuracy: {}".format(best_prec1))

def train(train_loader, validation_loader, model, vnet,optimizer_a,optimizer_c,epoch, loss_history):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    indexes = []
    weights2save = []
    model.train()


    for i, (input, target, index) in enumerate(train_loader):
        indexes.append(index)

        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        meta_model = build_model()

        meta_model.load_state_dict(model.state_dict())

        y_f_hat = meta_model(input_var)
        cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))

        v_lambda = vnet(loss_history[index])

        norm_c = torch.sum(v_lambda)

        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda

        l_f_meta = torch.sum(cost_v * v_lambda_norm)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        # print(grads)
        meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        input_validation, target_validation, index2 = next(iter(validation_loader))
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation, requires_grad=False)
        y_g_hat = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        # l_g_meta.backward(retain_graph=True)
        prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]

        optimizer_c.zero_grad()
        l_g_meta.backward()
        # print(vnet.linear1.weight.grad)
        optimizer_c.step()

        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(loss_history[index])
        norm_v = torch.sum(w_new)

        ### to save weights
        weight = torch.reshape(w_new, (len(w_new),))
        weight = weight.cpu().detach().numpy()
        weights2save.append(weight)

        if norm_v != 0:
            w_v = w_new / norm_v
        else:
            w_v = w_new

        l_f = torch.sum(cost_v * w_v)

        losses.update(l_f.item(), input.size(0))
        meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))

    return np.concatenate(indexes), np.concatenate(weights2save)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard

    return top1.avg


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model.cuda()
        # torch.backends.cudnn.benchmark = True


    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    # lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



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


if __name__ == '__main__':
    main()




