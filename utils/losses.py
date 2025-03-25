"""losses.py - Implements FocalLoss and SmoothFocalLoss for classification tasks."""

from torch import nn
import torch


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and focusing on hard examples.

    Args:
        gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma.
        weight (Tensor, optional): A manual rescaling weight given to each class.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        """
        Computes the Focal Loss.

        Args:
            logits (Tensor): Raw output from the model [batch_size, num_classes].
            targets (Tensor): Ground truth labels [batch_size].

        Returns:
            Tensor: Scalar loss value.
        """
        logp = self.ce(logits, targets)
        p = torch.exp(-logp)
        return ((1 - p) ** self.gamma * logp).mean()


class SmoothFocalLoss(nn.Module):
    """
    Combines Focal Loss with Label Smoothing for stable training.

    Args:
        gamma (float): Focal Loss gamma parameter.
        smoothing (float): Label smoothing factor.
        alpha (float): Weight for combining focal and smoothed loss.
        weight (Tensor, optional): Class weights.
    """

    def __init__(self, gamma=1.5, smoothing=0.1, alpha=0.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.alpha = alpha
        self.weight = weight

    def forward(self, logits, targets):
        """
        Computes the combined Focal and Smoothed Loss.

        Args:
            logits (Tensor): Raw model output [batch_size, num_classes].
            targets (Tensor): Ground truth labels [batch_size].

        Returns:
            Tensor: Scalar loss value.
        """
        ce = nn.functional.cross_entropy(
            logits, targets, reduction='none', weight=self.weight)
        logp = -ce
        p = torch.exp(logp)
        focal = ((1 - p) ** self.gamma) * ce
        smoothed = ce * (1 - self.smoothing) + self.smoothing / logits.size(1)
        return self.alpha * focal.mean() + (1 - self.alpha) * smoothed.mean()
