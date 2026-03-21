"""Shared training utilities.

Provides a single implementation of the inner training loop, evaluation,
and CIFAR data loading used across all experiment notebooks.

Usage
-----
    from utils.training import iterate_batches, train_epoch, evaluate, load_cifar

    train_data, test_data = load_cifar(10, device)

    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_data, opt, crit, bs,
                                      grad_clip=1.0, amp_dtype=torch.bfloat16)
        te_loss, te_acc = evaluate(model, test_data, crit, bs,
                                   amp_dtype=torch.bfloat16)
"""

import contextlib

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets


# ---------------------------------------------------------------------------
# Batch iteration
# ---------------------------------------------------------------------------

def iterate_batches(x: torch.Tensor, y: torch.Tensor, bs: int,
                    shuffle: bool = True):
    """Yield (x_batch, y_batch) mini-batches from GPU-resident tensors."""
    n = x.size(0)
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for start in range(0, n, bs):
        b = idx[start:start + bs]
        yield x[b], y[b]


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, data: tuple, opt: torch.optim.Optimizer,
                crit: nn.Module, bs: int,
                grad_clip: float | None = None,
                amp_dtype: torch.dtype | None = None) -> tuple[float, float]:
    """Run one training epoch.

    Parameters
    ----------
    model     : the network (must already be on the correct device)
    data      : (x, y) tensors, GPU-resident
    opt       : optimiser
    crit      : loss function
    bs        : mini-batch size
    grad_clip : if not None, clip gradient norm to this value (use 1.0 for ViT)
    amp_dtype : dtype for automatic mixed precision, e.g. torch.bfloat16.
                Pass None to disable AMP (default).

    Returns
    -------
    (mean_loss, accuracy_percent)
    """
    model.train()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0

    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
               if amp_dtype is not None and x.device.type == "cuda"
               else contextlib.nullcontext())

    for xb, yb in iterate_batches(x, y, bs):
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(xb)
            loss = crit(out, yb)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)

    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model: nn.Module, data: tuple, crit: nn.Module, bs: int,
             amp_dtype: torch.dtype | None = None) -> tuple[float, float]:
    """Evaluate model on data.

    Parameters
    ----------
    model     : the network
    data      : (x, y) tensors, GPU-resident
    crit      : loss function
    bs        : mini-batch size
    amp_dtype : dtype for AMP; None to disable.

    Returns
    -------
    (mean_loss, accuracy_percent)
    """
    model.eval()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0

    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
               if amp_dtype is not None and x.device.type == "cuda"
               else contextlib.nullcontext())

    for xb, yb in iterate_batches(x, y, bs, shuffle=False):
        with amp_ctx:
            out = model(xb)
            loss = crit(out, yb)
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)

    return loss_sum / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Dataset-specific normalisation constants
_CIFAR_STATS = {
    10:  (torch.tensor([0.4914, 0.4822, 0.4465]),
          torch.tensor([0.2470, 0.2435, 0.2616])),
    100: (torch.tensor([0.5071, 0.4867, 0.4408]),
          torch.tensor([0.2675, 0.2565, 0.2761])),
}

_CIFAR_CLASS = {10: datasets.CIFAR10, 100: datasets.CIFAR100}


def load_cifar(num_classes: int, device, train_size: int | None = None,
               test_size: int | None = None,
               seed: int = 0) -> tuple[tuple, tuple]:
    """Load CIFAR-10 or CIFAR-100, normalise, and move to *device*.

    Parameters
    ----------
    num_classes : 10 or 100
    device      : torch device string or object, e.g. 'cuda' or 'cpu'
    train_size  : number of training samples to use; None = full dataset
    test_size   : number of test samples to use; None = full dataset
    seed        : RNG seed for subsampling

    Returns
    -------
    (x_train, y_train), (x_test, y_test)  — GPU tensors if device is CUDA
    """
    if num_classes not in _CIFAR_STATS:
        raise ValueError(f"num_classes must be 10 or 100, got {num_classes}")

    mean, std = _CIFAR_STATS[num_classes]
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)

    DataClass = _CIFAR_CLASS[num_classes]
    train_ds = DataClass("./data", train=True,  download=True)
    test_ds  = DataClass("./data", train=False, download=True)

    rng = np.random.RandomState(seed)
    tr_idx = (rng.choice(len(train_ds.data), train_size, replace=False)
              if train_size and train_size < len(train_ds.data)
              else np.arange(len(train_ds.data)))
    te_idx = (rng.choice(len(test_ds.data), test_size, replace=False)
              if test_size and test_size < len(test_ds.data)
              else np.arange(len(test_ds.data)))

    x_tr = torch.from_numpy(train_ds.data[tr_idx]).permute(0, 3, 1, 2).float().div_(255)
    y_tr = torch.tensor(np.array(train_ds.targets)[tr_idx])
    x_te = torch.from_numpy(test_ds.data[te_idx]).permute(0, 3, 1, 2).float().div_(255)
    y_te = torch.tensor(np.array(test_ds.targets)[te_idx])

    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        x_tr, y_tr, x_te, y_te = x_tr.cuda(), y_tr.cuda(), x_te.cuda(), y_te.cuda()
        mean, std = mean.cuda(), std.cuda()

    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    print(f"CIFAR-{num_classes} — train: {x_tr.shape[0]:,}  test: {x_te.shape[0]:,}")
    return (x_tr, y_tr), (x_te, y_te)
