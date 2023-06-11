import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import vgg
import thop

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--dir_data', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=11,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--ratio', type=float, default=0.1,
                    help='prune ratio for each layer')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg.vgg(dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# vgg11的cfg:[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
# 预先设定新的cfg，根据设定的cfg生成剪枝模型
# cfg = [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]

# cfg = [int(64*args.ratio), 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
# cfg = [64, 'M', int(128*args.ratio), 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
# cfg = [64, 'M', 128, 'M', int(256*args.ratio), 256, 'M', 512, 512, 'M', 512, 512]
# cfg = [64, 'M', 128, 'M', 256, int(256*args.ratio), 'M', 512, 512, 'M', 512, 512]
# cfg = [64, 'M', 128, 'M', 256, 256, 'M', int(512*args.ratio), 512, 'M', 512, 512]
# cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, int(512*args.ratio), 'M', 512, 512]
# cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', int(512*args.ratio), 512]
cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, int(512*args.ratio)]


# 存放生成的剪枝掩膜，掩膜中保留通道的对应值为1，剪枝通道的对应值为0
cfg_mask = []
layer_id = 0
# 遍历模型

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        # 若当前卷积层的输出通道数等于cfg中设定的输出通道数，则当前卷积层全部保留，剪枝掩膜为全1
        if out_channels == cfg[layer_id]:
            cfg_mask.append(torch.ones(out_channels))
            layer_id += 1
            continue
        # 计算当前卷积层每一个卷积核的L1范数
        weight_copy = m.weight.data.abs().clone()
        weight_copy = weight_copy.cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
        # 对每一个卷积核的L1范数排序，得到排序结果的索引
        arg_max = np.argsort(L1_norm)
        # 根据cfg中对应层所需保留的输出通道数n和排序结果，从大到小取前n个输出通道的索引
        arg_max_rev = arg_max[:cfg[layer_id]]
        assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
        # mask为当前层输出通道的掩膜
        mask = torch.zeros(out_channels)
        mask[arg_max_rev.tolist()] = 1
        cfg_mask.append(mask)
        layer_id += 1
    elif isinstance(m, nn.MaxPool2d):
        layer_id += 1

# 根据上述新定义的cfg构建新模型
newmodel = vgg.vgg(dataset=args.dataset, depth=args.depth)
if args.cuda:
    newmodel.cuda()

# start_mask为输入通道的掩膜，初始化为第一个卷积层输入通道的掩膜
start_mask = torch.ones(3)
layer_id_in_cfg = 0
# end_mask为输出通道的掩膜，初始化为第一个卷积层的输出通道的掩膜
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    # 将model中需要留下的BN层权重移植到newmodel上
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        # 根据索引移植权重
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        # 当前BN层权重移植结束，输入和输出通道的掩膜进行迭代更新
        layer_id_in_cfg += 1
        # 下一层的输入通道掩膜等于当前层的输出通道掩膜
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            # end_mask变为下一层的掩膜
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        # idx0是从start_mask中得到当前卷积层m0需要保留输入通道的索引
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        # idx1是从end_mask中得到当前卷积层m0需要保留输出通道的索引
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        # 根据索引移植权重
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        # 第一个全连接层的输入节点根据上一层的剪枝结果，进行剪枝
        if layer_id_in_cfg == len(cfg_mask):
            # idx1是通过cfg_mask[-1]得到的当前全连接层保留下来的输入节点索引
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
            layer_id_in_cfg += 1
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
    elif isinstance(m0, nn.BatchNorm1d):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()


filename1 = f'pruned_layer8_{args.ratio}.pth.tar'
torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, filename1))
print(newmodel)
model = newmodel


filename2 = f"prune_layer8_{args.ratio}.txt"
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
with open(os.path.join(args.save, filename2), "w") as fp:
    fp.write("Number of parameters: \n" + str(num_parameters) + "\n")

