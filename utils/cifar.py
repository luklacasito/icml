"""CIFAR-10 preprocessing shared by the original MLP sweep notebooks."""

import numpy as np
import torch
from torchvision import datasets


def load_cifar10(train_size=None, test_size=None, seed=0, *, root, device):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    train_set = datasets.CIFAR10(root / "data", train=True, download=True)
    test_set = datasets.CIFAR10(root / "data", train=False, download=True)

    rng = np.random.RandomState(seed)

    if train_size is None or train_size >= len(train_set.data):
        tr_idx = np.arange(len(train_set.data))
    else:
        tr_idx = rng.choice(len(train_set.data), train_size, replace=False)

    if test_size is None or test_size >= len(test_set.data):
        te_idx = np.arange(len(test_set.data))
    else:
        te_idx = rng.choice(len(test_set.data), test_size, replace=False)

    print(f"Train: {len(tr_idx)} samples, Test: {len(te_idx)} samples")

    x_tr = torch.from_numpy(train_set.data[tr_idx]).permute(0, 3, 1, 2).float() / 255
    y_tr = torch.tensor(np.array(train_set.targets)[tr_idx])
    x_te = torch.from_numpy(test_set.data[te_idx]).permute(0, 3, 1, 2).float() / 255
    y_te = torch.tensor(np.array(test_set.targets)[te_idx])

    if device == "cuda":
        x_tr, y_tr = x_tr.cuda(), y_tr.cuda()
        x_te, y_te = x_te.cuda(), y_te.cuda()
        mean, std = mean.cuda(), std.cuda()

    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    return (x_tr, y_tr), (x_te, y_te)
