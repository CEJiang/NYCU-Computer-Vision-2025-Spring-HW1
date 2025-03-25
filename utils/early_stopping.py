"""Implements simple early stopping based on validation loss."""

import numpy as np


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    Attributes:
        patience (int): How many epochs to wait after last time validation loss improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        best_loss (float): Best recorded validation loss.
        counter (int): Counts epochs with no improvement.
        early_stop (bool): Flag to indicate whether to stop training.
    """

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if validation loss has improved.
        If not, increase counter and check for early stopping.

        Args:
            val_loss (float): Current epoch's validation loss.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                self.early_stop = True
