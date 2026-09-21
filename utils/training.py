"""Batch training, evaluation, and CIFAR loading for the experiment notebooks."""

import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets


def iterate_batches(x: torch.Tensor, y: torch.Tensor, bs: int, shuffle: bool = True):
    """Yield paired batches on the tensors' device, including the final partial batch."""
    n = x.size(0)
    idx = (
        torch.randperm(n, device=x.device)
        if shuffle
        else torch.arange(n, device=x.device)
    )
    for start in range(0, n, bs):
        b = idx[start : start + bs]
        yield x[b], y[b]


def train_epoch(
    model: nn.Module,
    data: tuple,
    opt: torch.optim.Optimizer,
    crit: nn.Module,
    bs: int,
    grad_clip: float | None = None,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    """Train once over the data and return mean loss and accuracy in percent.

    The criterion must average over its batch. Model and data must share a
    device. Set grad_clip to clip the gradient norm, and amp_dtype to enable
    mixed precision on CUDA; CPU computation remains full precision.
    """
    model.train()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None and x.device.type == "cuda"
        else contextlib.nullcontext()
    )

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
def evaluate(
    model: nn.Module,
    data: tuple,
    crit: nn.Module,
    bs: int,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    """Return sample-weighted mean loss and accuracy in percent, without gradients."""
    model.eval()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None and x.device.type == "cuda"
        else contextlib.nullcontext()
    )

    for xb, yb in iterate_batches(x, y, bs, shuffle=False):
        with amp_ctx:
            out = model(xb)
            loss = crit(out, yb)
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)

    return loss_sum / total, 100.0 * correct / total


_CIFAR_STATS = {
    10: (
        torch.tensor([0.4914, 0.4822, 0.4465]),
        torch.tensor([0.2470, 0.2435, 0.2616]),
    ),
    100: (
        torch.tensor([0.5071, 0.4867, 0.4408]),
        torch.tensor([0.2675, 0.2565, 0.2761]),
    ),
}

_CIFAR_CLASS = {10: datasets.CIFAR10, 100: datasets.CIFAR100}


def load_cifar(
    num_classes: int,
    device,
    train_size: int | None = None,
    test_size: int | None = None,
    seed: int = 0,
) -> tuple[tuple, tuple]:
    """Load normalized CIFAR tensors on CPU or CUDA, cached in the repo's data/.

    Return (x_train, y_train), (x_test, y_test). Optional sample counts select
    subsets without replacement using a local NumPy RNG seeded by seed.
    """
    if num_classes not in _CIFAR_STATS:
        raise ValueError(f"num_classes must be 10 or 100, got {num_classes}")

    mean, std = _CIFAR_STATS[num_classes]
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)

    DataClass = _CIFAR_CLASS[num_classes]
    data_root = Path(__file__).resolve().parents[1] / "data"
    train_ds = DataClass(data_root, train=True, download=True)
    test_ds = DataClass(data_root, train=False, download=True)

    rng = np.random.RandomState(seed)
    tr_idx = (
        rng.choice(len(train_ds.data), train_size, replace=False)
        if train_size and train_size < len(train_ds.data)
        else np.arange(len(train_ds.data))
    )
    te_idx = (
        rng.choice(len(test_ds.data), test_size, replace=False)
        if test_size and test_size < len(test_ds.data)
        else np.arange(len(test_ds.data))
    )

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
