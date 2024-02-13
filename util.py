import importlib
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold

from typing import Optional
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F

from dataloaders.base_dataset import TissueDataset

import torch

def find_dataset_using_name(dataset_name):
    """Import the module "dataloaders/[dataset_name].py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of TissueDataset,
    and it is case-insensitive.
    """
    dataset_filename = "dataloaders." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'Dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, TissueDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of TissueDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def separate_data(graph_list, seed, n_folds, fold_idx):
    assert 0 <= fold_idx and fold_idx < n_folds, "fold_idx must be from 0 to 9."
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True)

    groups = []
    labels = []
    for info in graph_list:
        info = info.replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        patient_id = file_name.rsplit('-', 1)[0]
        groups.append(patient_id)
        labels.append(label)
    idx_list = []
    for idx in sgkf.split(np.zeros(len(labels)), labels, groups=groups):
        idx_list.append(idx)

    train_val_idx, test_idx = idx_list[fold_idx]

    train_val_graph_list = [graph_list[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]

    test_graph_list = [graph_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_val_graph_list, test_graph_list, train_val_labels

def read_file(file_name):
    with open(file_name, 'r') as f:
        records = list(f)

    return records

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              args.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <args.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch) / float(args.n_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], verbose=True)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0, verbose=True)
    elif args.lr_policy == 'cosine_wr':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, verbose=True)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler

def downsample_image(slide, downsampling_factor, mode="numpy"):
    """Downsample an Openslide at a factor.
    Takes an OpenSlide SVS object and downsamples the original resolution
    (level 0) by the requested downsampling factor, using the most convenient
    image level. Returns an RGB numpy array or PIL image.
    Args:
        slide: An OpenSlide object.
        downsampling_factor: Power of 2 to downsample the slide.
        mode: String, either "numpy" or "PIL" to define the output type.
    Returns:
        img: An RGB numpy array or PIL image, depending on the mode,
            at the requested downsampling_factor.
        best_downsampling_level: The level determined by OpenSlide to perform the downsampling.
    """

    # Get the best level to quickly downsample the image
    # Add a pseudofactor of 0.1 to ensure getting the next best level
    # (i.e. if 16x is chosen, avoid getting 4x instead of 16x)
    best_downsampling_level = slide.get_best_level_for_downsample(downsampling_factor + 0.1)

    # Get the image at the requested scale
    target_size = slide.level_dimensions[best_downsampling_level]
    img = slide.read_region((0, 0), best_downsampling_level, target_size)
    
    # target_size = tuple([int(x//downsampling_factor) for x in slide.dimensions])
    # img = svs_native_levelimg.resize(target_size)

    # By default, return a numpy array as RGB, otherwise, return PIL image
    if mode == "numpy":
        # Remove the alpha channel
        img = np.array(img.convert("RGB"))

    return img, best_downsampling_level, target_size

class BinaryCrossEntropyLoss(nn.Module):
    """BCE with optional one-hot from dense targets, label smoothing, thresholding
    from https://github.com/rwightman/pytorch-image-models/blob/a520da9b49/timm/loss/binary_cross_entropy.py

    The label smoothing is done as in `torch.nn.CrossEntropyLoss`.
    In other words, the formula from https://arxiv.org/abs/1512.00567 is strictly followed
    even if input targets samples are sparse, unlike in timm.

    Important: Inputs are assumed to be logits. Targets can be either dense or sparse, and in the latter
    they should not be in logit space.
    """

    def __init__(
        self,
        smoothing=0.0,
        target_threshold: Optional[float] = None,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(BinaryCrossEntropyLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]

        num_classes = x.shape[-1]

        # convert dense to sparse
        if target.shape != x.shape:
            target = F.one_hot(target, num_classes=num_classes).to(dtype=x.dtype)

        # apply smoothing if needed
        if self.smoothing > 0.0:
            # just like in `torch.nn.CrossEntropyLoss`
            target = target * (1 - self.smoothing) + self.smoothing / num_classes

        # Make target 0, or 1 if threshold set
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)

        return F.binary_cross_entropy_with_logits(
            x, target, self.weight, pos_weight=self.pos_weight, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        result = f"reduction={self.reduction}, "
        if self.smoothing > 0:
            result += f"smoothing={self.smoothing}, "
        if self.target_threshold is not None:
            result += f"target_threshold={self.target_threshold}, "
        if self.weight is not None:
            result += f"weight={self.weight.shape}, "
        if self.pos_weight is not None:
            result += f"pos_weight={self.pos_weight.shape}, "
        result = result[:-2]
        return result