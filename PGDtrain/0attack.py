'''0 attack'''
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
from getmodel import *
from Classifier import *
from collections import OrderedDict
import torch_dct as dct

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
#pgd train
parser.add_argument('--eps', default=0.031, type=float, help='eps')

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
batchsize = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batchsize,
                                         shuffle=False,
                                         num_workers=2,
                                         pin_memory=True)
print('==> Building model pgd..')

net = getmodel(22, 2, device)
net = resumenet(net, '0')  # contains net.eval()

if device == 'cuda':
    cudnn.benchmark = True


def add_mask(idx, mask):
    for i in range(len(idx)):
        m = torch.zeros_like(mask)
        m[:, :, :idx[i] + 1, :idx[i] + 1] = 1
        m[:, :, :idx[i], :idx[i]] = 0
        mask += m
    return mask


def cri(outputs, lables):
    o = outputs[0].clone()
    o1 = o[lables]
    o[lables] = -100
    o2 = o.max()
    return o1 - o2


#Init Setting
fre = 32
H = 1.
steps = 500
stepsize = 0.2
n = 4  #num of fre we select
lam = 0.  #control the distance
u = torch.ones(fre) * H  #value function
T = torch.zeros(fre)  #num of trails
num_queries = 1
attack_success = 0
num_sample = 0
total_norm = 0

avg_queries = []
avg_norm = []


#target attack
def untarget(init=False):
    global u, T, num_queries, attack_success, num_sample, total_norm, avg_norm, avg_queries
    print('Init:', init)
    for batch_idx, (images, lables) in enumerate(testloader):
        if (init and batch_idx >= 200):
            break
        if (not init and batch_idx < 200):
            continue
        images, lables = images.to(device), lables.to(device)
        outputs = net(images)
        _, predicted = outputs.max(1)
        if predicted != lables:
            continue
        num_queries += 1
        num_sample += 1
        trail = 0  #num of trail for this sample
        if not init:
            u_bar = u + (3 * torch.log(T.sum()) / 2 / (T + 0.01))**0.5
        adv = images.clone()  #adv images
        while trail <= (fre / n - 1):
            trail += 1
            trail_queries = 0  #num of query for this trail
            if init:
                idx = torch.randperm(fre)[:n]
            else:
                _, idx = torch.topk(u_bar, fre)
                idx = idx[(trail - 1) * n:trail * n]

            stay = 0  #if stay>100 then change the fre we select
            loss = cri(outputs, lables) - lam * (adv - images).norm()

            for i in range(steps):
                noise = torch.randn_like(images)
                mask = torch.zeros_like(images)
                mask = add_mask(idx, mask)
                g = dct.idct_2d(noise * mask)
                g = g / g.norm() * stepsize  #the gradient
                outputs_p = net((adv + g).clamp(0, 1))  #positive direction
                num_queries += 1
                trail_queries += 1
                loss_p = cri(outputs_p,
                             lables) - lam * (adv + g - images).norm()
                outputs_n = net((adv - g).clamp(0, 1))  #negative direction
                num_queries += 1
                trail_queries += 1
                loss_n = cri(outputs_n,
                             lables) - lam * (adv - g - images).norm()

                if (loss_n > loss or loss_p > loss):
                    stay = 0
                    if loss_p >= loss_n:
                        loss = loss_p
                        adv = (adv + g).clamp(0, 1)
                        _, predicted = outputs_p.max(1)
                        if predicted != lables:
                            attack_success += 1
                            break
                    elif loss_p < loss_n:
                        loss = loss_n
                        adv = (adv - g).clamp(0, 1)
                        _, predicted = outputs_n.max(1)
                        if predicted != lables:
                            attack_success += 1
                            break
                else:
                    stay += 1
                    if stay > 100:
                        i = steps - 1
                        break

            reward = torch.tensor((1 - trail_queries / 600.) * H).clamp(0., H)
            if (i < 499 and trail > 1):
                reward += .8
                reward.clamp_(0., H)
            u[idx] = (u[idx] * T[idx] + reward) / (T[idx] + 1)
            #print(u, trail_queries)
            T[idx] += 1
            if i < 499:
                total_norm += (adv - images).norm()
                avg_queries.append(1. * num_queries / num_sample)
                avg_norm.append(1. * total_norm / num_sample)
                progress_bar(
                    batch_idx, len(testloader),
                    'avg_queries: %.2f | success: %.2f%% | avg_norm: %.2f' %
                    (1. * num_queries / num_sample, 100. * attack_success /
                     num_sample, 1. * total_norm / num_sample))
                break


#target attack
def test(init=False):
    global u, T, num_queries, attack_success, num_sample, total_norm, avg_norm, avg_queries
    print('Init:', init)
    for batch_idx, (images, lables_true) in enumerate(testloader):
        if (init and batch_idx >= 200):
            break
        if (not init and batch_idx < 200):
            continue
        images, lables_true = images.to(device), lables_true.to(device)
        outputs = net(images)
        _, predicted = outputs.max(1)
        lables = torch.randint(10, [1]).to(device)
        while (lables == predicted or lables == lables_true):
            lables = torch.randint(10, [1]).to(device)
        num_queries += 1
        num_sample += 1
        trail = 0  #num of trail for this sample
        if not init:
            u_bar = u + (3 * torch.log(T.sum()) / 2 / (T + 0.01))**0.5
        adv = images.clone()  #adv images
        while trail <= (fre / n - 1):
            trail += 1
            trail_queries = 0  #num of query for this trail
            if init:
                idx = torch.randperm(fre)[:n]
            else:
                _, idx = torch.topk(u_bar, fre)
                idx = idx[(trail - 1) * n:trail * n]

            stay = 0  #if stay>100 then change the fre we select
            loss = cri(outputs, lables) - lam * (adv - images).norm()

            for i in range(steps):
                noise = torch.randn_like(images)
                mask = torch.zeros_like(images)
                mask = add_mask(idx, mask)
                g = dct.idct_2d(noise * mask)
                g = g / g.norm() * stepsize  #the gradient
                outputs_p = net((adv + g).clamp(0, 1))  #positive direction
                num_queries += 1
                trail_queries += 1
                loss_p = cri(outputs_p,
                             lables) - lam * (adv + g - images).norm()
                outputs_n = net((adv - g).clamp(0, 1))  #negative direction
                num_queries += 1
                trail_queries += 1
                loss_n = cri(outputs_n,
                             lables) - lam * (adv - g - images).norm()
                #print(loss, loss_p, loss_n)
                if (loss_n > loss or loss_p > loss):
                    stay = 0
                    if loss_p >= loss_n:
                        loss = loss_p
                        adv = (adv + g).clamp(0, 1)
                        _, predicted = outputs_p.max(1)
                        if predicted == lables:
                            attack_success += 1
                            break
                    elif loss_p < loss_n:
                        loss = loss_n
                        adv = (adv - g).clamp(0, 1)
                        _, predicted = outputs_n.max(1)
                        if predicted == lables:
                            attack_success += 1
                            break
                else:
                    stay += 1
                    if stay > 100:
                        i = steps - 1
                        break

            reward = torch.tensor((1 - trail_queries / 600.) * H).clamp(0., H)
            if (i < 499 and trail > 1):
                reward += .8
                reward.clamp_(0., H)
            u[idx] = (u[idx] * T[idx] + reward) / (T[idx] + 1)
            #print(u, trail_queries, reward)
            T[idx] += 1
            if i < 499:
                total_norm += (adv - images).norm()
                avg_queries.append(1. * num_queries / num_sample)
                avg_norm.append(1. * total_norm / num_sample)
                progress_bar(
                    batch_idx, len(testloader),
                    'avg_queries: %.2f | success: %.2f%% | avg_norm: %.2f' %
                    (1. * num_queries / num_sample, 100. * attack_success /
                     num_sample, 1. * total_norm / num_sample))
                break


test(init=True)
test(init=False)

plt.plot(avg_queries, label='avg_queries')
plt.legend(frameon=False)
pltname = ('./PGDtrain/plots/avg_queries.png')
plt.savefig(pltname)

plt.plot(avg_norm, label='avg_norm')
plt.legend(frameon=False)
pltname = ('./PGDtrain/plots/avg_norm.png')
plt.savefig(pltname)