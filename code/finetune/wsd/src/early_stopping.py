

import os

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf

    def __call__(self, score, model, args):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, args)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, args)
            self.counter = 0

    def save_checkpoint(self, score, model, args):
        """Saves model when validation loss decreases or accuracy/f1 increases."""
        if self.verbose:
            print(f'{args.tuning_metric} imporoved ({self.val_score_min:.6f} ----> {score:.6f}). Saving model ....')
        model.save_pretrained(args.model_dir)
        torch.save(args, os.path.join(args.model_dir, "training_args.bin"))
        self.val_score_min = score