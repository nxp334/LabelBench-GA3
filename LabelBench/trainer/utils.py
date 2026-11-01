# Modified version of https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py.
import copy
import math
from typing import Optional

import numpy as np
import torch


def effective_max_epoch(max_epoch: int, num_labeled: int, train_batch_size: int,
                        min_steps: Optional[int] = None) -> int:
    """Number of epochs that gives at least min_steps optimizer steps, and never less than max_epoch."""
    if min_steps is None:
        return max_epoch
    # The labeled loader only drops the last batch if there is a full batch.
    steps_per_epoch = num_labeled // train_batch_size if num_labeled >= train_batch_size else 1
    return max(max_epoch, math.ceil(min_steps / steps_per_epoch))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0):
        """
        :param int patience: How long to wait after last time validation loss improved.
        :param bool verbose: If True, prints a message for each validation loss improvement. Default: False.
        :param float delta: Minimum change in the monitored quantity to qualify as an improvement. Default: 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.train_loss_min = np.inf
        self.delta = delta
        self.best_state_dict = None

        if self.patience is None:
            print("EarlyStopping is disabled.")

    def __call__(self, val_loss, model=None, train_loss=None):
        if self.patience is None:
            pass
        else:
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, train_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, train_loss, model) 
                self.counter = 0

    def save_checkpoint(self, val_loss, train_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print()
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). ')
            if train_loss is not None:
                print(f'Train loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}). ')

        self.best_state_dict = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
        if train_loss is not None:
            self.train_loss_min = train_loss
