'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from matplotlib import pyplot as plt
import copy

#import models
from utils import progress_bar
from normal import *
from PGD import *
from wideresnet import WideResNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--gpu',
                    default=None,
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--model',
                    default="wideresnet",
                    type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--loss',
                    default="CE",
                    type=str,
                    help='CE, l2, l4, softmaxl2, softmaxl4')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decay', default=5e-4, type=float, help='decay rate')
parser.add_argument('--epoch',
                    default=200,
                    type=int,
                    help='total epochs to run')
#pgd train
parser.add_argument('--pgd',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--iters', default=7, type=int, help='num of iters')
parser.add_argument('--eps', default=0.031, type=float, help='eps')
parser.add_argument('--norm', default=10, type=float, help='norm')

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=2,
                                         pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

print('==> Building model pgd..')
if args.model == 'wideresnet':
    net = WideResNet(depth=28, num_classes=10, widen_factor=2)
elif args.model == 'wideresnet28':
    net = WideResNet(depth=28, num_classes=10, widen_factor=10)
else:
    net = models.__dict__[args.model]()
net = torch.nn.Sequential(NormalizeLayer(), net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(
        './PGDtrain/checkpoint'), 'Error: no checkpoint directory found!'
    ckpname = ('./PGDtrain/checkpoint/' + args.loss + '_' + args.model + '_' +
               args.name + '.pth')
    checkpoint = torch.load(ckpname)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.decay)


class myloss():
    def __init__(self, norm):
        self.norm = norm

    def __call__(self, outputs, targets):
        mask = torch.ones_like(outputs) * (-1)
        mask[range(len(targets)), targets] = 10
        return ((outputs - mask)**2).mean()


if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'l2':
    criterion = myloss(norm=2)


def test(epoch):
    global best_acc
    net.eval()
    total = torch.zeros(10).int()
    correct = torch.zeros(10).int()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.size(0)
            total_batch = torch.zeros(batch_size, 10).int().to(device)
            for i in range(200):
                inputs_noise = torch.clamp(inputs +
                                           torch.randn_like(inputs) * 1,
                                           min=0,
                                           max=1)
                outputs = net(inputs_noise)

                _, predicted = outputs.max(1)
                total_batch[range(batch_size), predicted] += 1
            
            _, predicted = total_batch.max(1)
            correct_batch = predicted.eq(targets).int()
            for j in range(batch_size):
                lable = targets[j]
                correct[lable] += correct_batch[j]
                total[lable] += 1

            print(batch_idx)
            print((correct.float() / total.float()))


print(args)
test(1)
