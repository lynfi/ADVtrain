"""
    some classifier
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct


class Detect(nn.Module):
    def __init__(self, model, num_classes=10):
        super(Detect, self).__init__()
        self.fc = nn.Sequential(nn.Linear(num_classes, 5 * num_classes),
                                nn.PReLU(), nn.Dropout(0.2),
                                nn.Linear(num_classes * 5, 5 * num_classes),
                                nn.PReLU(), nn.Dropout(0.2),
                                nn.Linear(num_classes * 5, 2))

    def forward(self, x):
        return self.fc(x)


class Lowpass(nn.Module):
    def __init__(self, model, fre=15, widen_factor=1):
        super(Lowpass, self).__init__()
        self.fre = fre
        self.model = model

    def forward(self, x):
        X = dct.dct_2d(x)
        mask = torch.zeros_like(X)
        mask[:, :, 0:self.fre, 0:self.fre] = 1
        out = dct.idct_2d(X * mask)
        out = self.model(out)
        return out


class sumEnsemble(nn.Module):
    def __init__(self, model_S, model_T, num_classes=10):
        super(sumEnsemble, self).__init__()
        self.model_S = model_S
        self.model_T = model_T
        self.fc = nn.Sequential(
            nn.Linear(num_classes * 2, 2 * num_classes),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_classes * 2, 2 * num_classes),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_classes * 2, num_classes),
        )

    def forward(self, x):
        x1 = self.model_S(x)
        x2 = self.model_T(x)
        out = torch.cat((x1, x2), dim=1)
        return self.fc(out)


class Ensemble(nn.Module):
    def __init__(self, model_S, model_T, widen_factor=2, num_classes=10):
        super(Ensemble, self).__init__()
        self.model_S = model_S
        self.model_T = model_T
        self.widen_factor = widen_factor
        self.relu = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(64 * widen_factor * 2, 32 * widen_factor),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(32 * widen_factor, num_classes),
        )

    def forward(self, x):
        x1 = self.model_S(x)
        x2 = self.model_T(x)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x1 = F.avg_pool2d(x1, 8)
        x2 = F.avg_pool2d(x2, 8)
        x1 = x1.view(-1, 64 * self.widen_factor)
        x2 = x2.view(-1, 64 * self.widen_factor)
        out = torch.cat((x1, x2), dim=1)
        return self.fc(out)


class Classifier(nn.Module):
    def __init__(self, widen_factor=2, num_classes=10, fixup=False):
        super(Classifier, self).__init__()
        self.widen_factor = widen_factor
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * widen_factor, num_classes)
        self.fc.bias.data.zero_()
        if fixup:
            self.fc.weight.data.zero_()

    def forward(self, x):
        out = self.relu(x)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, 64 * self.widen_factor)
        return self.fc(out)
