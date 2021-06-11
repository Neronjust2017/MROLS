import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaConv1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv1d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    MetaBatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(MetaModule):
    def __init__(self, num_classes, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.Sigmoid()
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class VCNN(MetaModule):
    def __init__(self, hidden1, output, ksize=5):
        super(VCNN, self).__init__()
        padding = (ksize - 1) // 2
        self.conv1 = MetaConv1d(1, 16, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv2 = MetaConv1d(16, 16, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv3 = MetaConv1d(16, 32, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv4 = MetaConv1d(32, 32, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.linear1 = MetaLinear(32*25, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 32*25)
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class VCNN_1Dense(MetaModule):
    def __init__(self, ksize=5):
        super(VCNN_1Dense, self).__init__()
        padding = (ksize - 1) // 2
        self.conv1 = MetaConv1d(1, 16, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv2 = MetaConv1d(16, 16, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv3 = MetaConv1d(16, 32, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv4 = MetaConv1d(32, 32, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.linear1 = MetaLinear(32*25, 1)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 32*25)
        out = self.linear1(x)
        return torch.sigmoid(out)

class VCNN_1Dense_2(MetaModule):
    def __init__(self, ksize=5):
        super(VCNN_1Dense_2, self).__init__()
        padding = (ksize - 1) // 2
        self.conv1 = MetaConv1d(1, 16, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv2 = MetaConv1d(16, 16, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.linear1 = MetaLinear(16*50, 1)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 32*25)
        out = self.linear1(x)
        return torch.sigmoid(out)

class VCNN_1Dense_5(MetaModule):
    def __init__(self, ksize=5):
        super(VCNN_1Dense_5, self).__init__()
        padding = (ksize - 1) // 2
        self.conv1 = MetaConv1d(1, 8, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv2 = MetaConv1d(8, 8, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv3 = MetaConv1d(8, 8, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.conv4 = MetaConv1d(8, 8, kernel_size=ksize, stride=1, padding=padding, bias=False)
        self.linear1 = MetaLinear(8*25, 1)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 8*25)
        out = self.linear1(x)
        return torch.sigmoid(out)

class VCNN3(MetaModule):
    def __init__(self, hidden1, output):
        super(VCNN3, self).__init__()
        self.conv1 = MetaConv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = MetaBatchNorm1d(16)
        self.bn2 = MetaBatchNorm1d(16)
        self.bn3 = MetaBatchNorm1d(32)
        self.bn4 = MetaBatchNorm1d(32)
        self.conv2 = MetaConv1d(16, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = MetaConv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = MetaConv1d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.linear1 = MetaLinear(32*25, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 32*25)
        x = self.linear1(x)
        # x = self.linear2(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class VCNN2(MetaModule):
    def __init__(self, hidden1, output, input=1):
        super(VCNN2, self).__init__()
        self.conv1 = MetaConv1d(input, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = MetaConv1d(16, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = MetaConv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = MetaConv1d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.linear1 = MetaLinear(32*25, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = x.view(-1, 32*25)
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class VCNN_1layer(MetaModule):
    def __init__(self, hidden1, output):
        super(VCNN_1layer, self).__init__()
        self.conv1 = MetaConv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.linear1 = MetaLinear(16*50, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = x.view(-1, 16*50)
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class BasicBlock_1d(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1d, self).__init__()
        self.conv1 = MetaConv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm1d(planes)
        self.conv2 = MetaConv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2], (0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MetaConv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    MetaBatchNorm1d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MetaConv1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv1d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class ResNet32_1d(MetaModule):
    def __init__(self, num_classes, block=BasicBlock_1d, num_blocks=[5, 5, 5]):
        super(ResNet32_1d, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm1d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool1d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return torch.sigmoid(out)


class VCNN_BCE(MetaModule):
    def __init__(self, hidden1, output):
        super(VCNN_BCE, self).__init__()
        self.conv1 = MetaConv1d(10, 10, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = MetaConv1d(10, 10, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = MetaConv1d(10, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = MetaConv1d(10, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear1 = MetaLinear(10*50, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = F.max_pool1d(x, 2)

        x = x.view(-1, 10*50)
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class VCNN_BCE_2(MetaModule):
    def __init__(self, hidden1, output):
        super(VCNN_BCE_2, self).__init__()
        self.conv1 = MetaConv1d(10, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = MetaConv1d(10, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = MetaConv1d(10, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = MetaConv1d(10, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear1 = MetaLinear(10*50, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = F.max_pool1d(x, 2)

        x = x.view(-1, 10*50)
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


if __name__ == '__main__':
    # a = torch.randn((16, 10, 100))
    # a = to_var(a)
    # vcnn = VCNN_BCE(16, 1)
    # b = vcnn(a)
    # print('get output')
    from torch.distributions import Categorical
    a = torch.tensor([0.1, 0.65, 0.15, 0.1])
    b = torch.tensor([0.1, 0.1, 0.7, 0.1])
    a_label = torch.tensor([1])
    b_label = torch.tensor([2])
    a_entropy = Categorical(probs=a).entropy()
    print(a_entropy)
    b_entropy = Categorical(probs=b).entropy()
    print(b_entropy)
    a_ce = torch.nn.functional.cross_entropy(a.view(1, 4), a_label)
    b_ce = torch.nn.functional.cross_entropy(b.view(1, 4), b_label)
    print('done')


