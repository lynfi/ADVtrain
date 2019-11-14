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
from Classifier import *
from collections import OrderedDict

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
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay', default=5e-4, type=float, help='decay rate')
parser.add_argument('--epoch',
                    default=200,
                    type=int,
                    help='total epochs to run')
#pgd train
parser.add_argument('--pgd',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--iters', default=20, type=int, help='num of iters')
parser.add_argument('--eps', default=0.031, type=float, help='eps')
parser.add_argument('--norm', default=10, type=float, help='norm')
parser.add_argument('--numsample',
                    default=256,
                    type=int,
                    help='num of random sample')

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
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=2)

feature = WideResNet(22, 10, 2)
classifier = Classifier(widen_factor=2, num_classes=10, fixup=False)
feature = nn.Sequential(NormalizeLayer(), feature)
net = nn.Sequential(
    OrderedDict([('feature', feature), ('classifier', classifier)]))
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True
for p in net.parameters():
    p.requires_grad = False

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(
        './PGDtrain/checkpoint'), 'Error: no checkpoint directory found!'
    ckpname = ('./PGDtrain/checkpoint/' + args.loss + '_' + args.name + '.pth')
    checkpoint = torch.load(ckpname)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

detector = Detect()
optimizer = optim.SGD(detector.parameters(),
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.decay)

criterion = nn.CrossEntropyLoss()


#collect data
def datacollection():
    net.eval()
    total = 0
    total_p = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        print(batch_idx)
        if batch_idx >= 20:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        for i in range(9):
            f = torch.zeros_like(net(inputs))
            for j in range(args.numsample):
                inputs_noise = inputs + torch.randn_like(inputs) * 0.02
                outputs = net(inputs_noise)
                f += outputs
            f /= args.numsample
            if total == 0:
                L = f
                targets_L = torch.ones_like(targets).long()
                targets_true = targets.clone()
            else:
                L = torch.cat([L, f], 0)
                targets_L = torch.cat(
                    [targets_L, torch.ones_like(targets).long()], 0)
                targets_true = torch.cat([targets_true, targets.clone()], 0)
            total += f.size(0)

        invpgd = PGD(net, args.eps, 4 / 255, args.iters, args.norm, criterion,
                     device)
        for i in range(10):
            adv_targets = (torch.ones_like(targets) * i).long()
            inputs_p = inputs[1 - (adv_targets == targets)]
            adv_targets = adv_targets[1 - (adv_targets == targets)]
            adv_inputs = invpgd(inputs_p,
                                adv_targets,
                                randinit=True,
                                inverse=-1)
            f = torch.zeros_like(net(adv_inputs))
            for j in range(args.numsample):
                inputs_noise = adv_inputs + torch.randn_like(adv_inputs) * 0.02
                outputs = net(inputs_noise)
                f += outputs
            f /= args.numsample
            L = torch.cat([L, f], 0)
            targets_L = torch.cat(
                [targets_L, torch.zeros_like(adv_targets).long()], 0)
            targets_true = torch.cat([targets_true, targets.clone()], 0)
            total_p += f.size(0)

        print(total, total_p)
        print(L.size(0), targets_L.size(0))
        print('saving data')
        state = {
            'L': L,
            'target_L': targets_L,
            'targets_true': targets_true,
        }

        idx = str(int(batch_idx / 10))
        ckpname = ('./PGDtrain/checkpoint/' + args.name + 'data' + idx +
                   '.pth')
        torch.save(state, ckpname)
        if batch_idx % 10 == 9:
            total = 0
            total_p = 0
    return L, targets_L


# Training detector
def train(epoch, L, targets_L):
    detector.train()
    total_sample = L.size(0)
    r = torch.randperm(total_sample)
    L = L[r]
    targets_L = targets_L[r]
    batch_size = 180
    iters = int(total_sample / batch_size)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx in range(iters):
        start_idx = batch_idx * batch_size
        end_idx = batch_idx * batch_size + batch_size
        inputs = L[range(start_idx, end_idx)]
        targets = targets_L[range(start_idx, end_idx)]

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = detector(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, iters, 'Loss: %.2f | Acc: %.1f%% (%d/%d)' %
            (train_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / (batch_idx + 1), 100. * correct / total)


def detect(inputs):
    return (detector(net(inputs)))


def test(epoch):
    test_loss = 0
    correct = 0
    total = 0
    detector.eval()
    for p in detector.parameters():
        p.requires_grad = False

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx < 20:
            pass
        inputs, targets = inputs.to(device), targets.to(device)

        f = net(inputs)
        f_bar = torch.zeros_like(f)
        for i in range(args.numsample):
            inputs_noise = inputs + torch.randn_like(inputs) * 0.1
            outputs = net(inputs_noise)
            f_bar += outputs
        f_bar /= args.numsample

        pgd = PGD(net, args.eps, 4 / 255, args.iters, args.norm, criterion,
                  device)
        adv_inputs = pgd(inputs.clone(), targets, randinit=True)
        f_p = net(adv_inputs)
        f_pbar = torch.zeros_like(f_p)
        for i in range(args.numsample):
            inputs_noise = adv_inputs + torch.randn_like(adv_inputs) * 0.1
            outputs = net(inputs_noise)
            f_pbar += outputs
        f_pbar /= args.numsample

        L = torch.cat([f, f_bar], 1)
        L_p = torch.cat([f_p, f_pbar], 1)

        inputs_logit = torch.cat([L, L_p], 0)
        targets_lable = torch.cat([targets, targets + 10], 0)
        outputs = model(inputs_logit)
        loss = criterion(outputs, targets_lable)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets_lable.size(0)
        correct += predicted.eq(targets_lable).sum().item()

        progress_bar(
            batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (test_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


print(args)
logit, adv_logit, targets_L, targets_L_p = datacollection()
for epoch in range(start_epoch, args.epoch):
    adjust_learning_rate(optimizer, epoch)
    train(epoch, logit, adv_logit, targets_L, targets_L_p)
test(0)