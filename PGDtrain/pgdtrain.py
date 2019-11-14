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
parser.add_argument('--loss',
                    default="CE",
                    type=str,
                    help='CE, l2, l4, softmaxl2, softmaxl4')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decay', default=2e-4, type=float, help='decay rate')
parser.add_argument('--epoch',
                    default=200,
                    type=int,
                    help='total epochs to run')
#pgd train
parser.add_argument('--pgd',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--lowpass',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--fre', default=10, type=int, help='num of iters')
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
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
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



print('==> Building model pgd..')

feature = WideResNet(22, 10, 2)
classifier = Classifier(widen_factor=2, num_classes=10, fixup=False)
feature = nn.Sequential(NormalizeLayer(), feature)
net = nn.Sequential(
    OrderedDict([('feature', feature), ('classifier', classifier)]))
if args.lowpass:
    net = Lowpass(net, fre=args.fre)
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

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

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.decay)

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.pgd:
            net_cp = copy.deepcopy(net)
            for p in net_cp.parameters():
                p.requires_grad = False
            pgd = PGD(net_cp, args.eps, 2 / 255, args.iters, args.norm,
                      criterion, device)
            adv_inputs = pgd(inputs, targets, randinit=True)
            outputs = net(adv_inputs)
        else:
            outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(trainloader), 'Loss: %.2f | Acc: %.1f%% (%d/%d)' %
            (train_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / (batch_idx + 1), 100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if epoch % 5 == 0:
            net_cp = copy.deepcopy(net)
            for p in net_cp.parameters():
                p.requires_grad = False
            pgd = PGD(net_cp, args.eps, 4 / 255, 20, args.norm, criterion,
                      device)
            adv_inputs = pgd(inputs, targets, randinit=True)
            outputs = net(adv_inputs)
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    #if acc > best_acc:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('./PGDtrain/checkpoint'):
        os.mkdir('./PGDtrain/checkpoint')
    ckpname = ('./PGDtrain/checkpoint/' + args.loss + '_' + args.name + '.pth')
    torch.save(state, ckpname)
    best_acc = acc
    return (test_loss / (batch_idx + 1), acc)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


test(0)
trainloss, trainacc, loss, testacc = [], [], [], []
print(args)
for epoch in range(start_epoch, args.epoch):
    adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc = train(epoch)
    trainloss.append(train_loss)
    trainacc.append(train_acc)

    test_loss, test_acc = test(epoch)
    loss.append(test_loss)
    testacc.append(test_acc)

plt.plot(trainloss, label='Training loss')
plt.plot(loss, label='Validation loss')
plt.legend(frameon=False)
pltname = ('./PGDtrain/plots/' + args.loss + '_' + args.name + '.png')
plt.savefig(pltname)