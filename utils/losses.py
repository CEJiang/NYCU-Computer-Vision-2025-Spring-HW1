import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        return ((1 - p) ** self.gamma * logp).mean()


class SmoothFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, smoothing=0.1, alpha=0.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.alpha = alpha
        self.weight = weight

    def forward(self, input, target):
        ce = nn.functional.cross_entropy(
            input, target, reduction='none', weight=self.weight)
        logp = -ce
        p = torch.exp(logp)
        focal = ((1 - p) ** self.gamma) * ce
        smoothed = ce * (1 - self.smoothing) + self.smoothing / input.size(1)
        return self.alpha * focal.mean() + (1 - self.alpha) * smoothed.mean()
