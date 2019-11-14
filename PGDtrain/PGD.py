import torch
import os
import torch.optim as optim


class Attack():
    def __init__(self, model, eps, alpha, iters, norm, criterion, device):
        self.model = model.eval()
        self.eps = eps
        self.iters = iters
        self.norm = norm
        self.device = device
        self.alpha = alpha
        self.criterion = criterion


class PGD(Attack):
    def __call__(self, images, labels, randinit=False, inverse=1):
        images = images.to(self.device).detach().clone()
        labels = labels.to(self.device).detach().clone()
        ori_images = images.detach().clone()

        if ((randinit) and (self.norm == 10) and (self.iters > 0)):
            images += (torch.rand_like(images) - .5) * self.eps * 2
            images = torch.clamp(images, min=0, max=1)
        if ((randinit) and (self.norm < 10) and (self.iters > 0)):
            images += torch.randn_like(images) * 0.001
            images = torch.clamp(images, min=0, max=1)

        for i in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)
            self.model.zero_grad()
            loss = self.criterion(outputs, labels) * inverse
            loss.backward()
            #print(loss)
            with torch.no_grad():
                if self.norm == 10:
                    adv_images = images + self.alpha * images.grad.sign()
                    eta = torch.clamp(adv_images - ori_images,
                                      min=-self.eps,
                                      max=self.eps)
                    images = torch.clamp(ori_images + eta, min=0,
                                         max=1).detach_()
                else:
                    adv_images = images + self.alpha * images.grad / images.grad.view(
                        images.shape[0], -1).norm(self.norm, dim=1).view(
                            -1, 1, 1, 1)
                    eta = adv_images - ori_images
                    mask = eta.view(eta.shape[0], -1).norm(self.norm,
                                                           dim=1) <= self.eps
                    scale = eta.view(eta.shape[0], -1).norm(self.norm, dim=1)
                    scale[mask] = self.eps
                    eta *= self.eps / scale.view(-1, 1, 1, 1)
                    images = torch.clamp(ori_images + eta, min=0,
                                         max=1).detach_()
        adv_images = images
        return adv_images


class PGDadam(Attack):
    def __call__(self, images, labels, randinit=False, inverse=1):
        images = images.to(self.device).detach().clone()
        labels = labels.to(self.device).detach().clone()
        ori_images = images.detach().clone()

        if ((randinit) and (self.norm == 10) and (self.iters > 0)):
            images += (torch.rand_like(images) - .5) * self.eps * 2
            images = torch.clamp(images, min=0, max=1)
        if ((randinit) and (self.norm < 10) and (self.iters > 0)):
            images += torch.randn_like(images) * 0.001
            images = torch.clamp(images, min=0, max=1)

        images.requires_grad = True
        optimizer = optim.Adam([images], lr=self.alpha)
        for i in range(self.iters):
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = (-1) * self.criterion(outputs, labels) * inverse
            #print(loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if self.norm == 10:
                    eta = torch.clamp(images.detach().clone() - ori_images,
                                      min=-self.eps,
                                      max=self.eps)
                    images.data = torch.clamp(ori_images + eta, min=0, max=1)
                else:
                    eta = images.detach().clone() - ori_images
                    mask = eta.view(eta.shape[0], -1).norm(self.norm,
                                                           dim=1) <= self.eps
                    scale = eta.view(eta.shape[0], -1).norm(self.norm, dim=1)
                    scale[mask] = self.eps
                    eta *= self.eps / scale.view(-1, 1, 1, 1)
                    images.data = torch.clamp(ori_images + eta, min=0, max=1)
        return images.detach().clone()