"""Unmodified definitions extracted from notebooks/mlp_dropout_sweep.ipynb.

See provenance.json for the source and AST hashes.
"""

import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

USE_AMP = False
AMP_DTYPE = torch.float16


def get_dropout_schedule(schedule_type, depth, h_bar, h_max=None):
    """
    Generate layer-wise dropout rates.
    Nonzero schedules preserve the mean dropout budget h_bar.
    """
    if schedule_type == "none":
        return [0.0] * depth

    if schedule_type == "constant":
        return [h_bar] * depth

    if schedule_type == "reverse_step":
        # Dropout concentrated in first half (early), mean = h_bar
        if h_max is None:
            h_max = 2 * h_bar
        f = h_bar / h_max
        n_drop = max(1, int(np.ceil(f * depth)))
        h_adj = h_bar * depth / n_drop
        return [h_adj] * n_drop + [0.0] * (depth - n_drop)

    if schedule_type == "big_step":
        # Dropout concentrated in first 1/3 of layers, mean = h_bar
        f = 1 / 3
        n_drop = max(1, int(np.ceil(f * depth)))
        h_adj = h_bar * depth / n_drop
        return [h_adj] * n_drop + [0.0] * (depth - n_drop)

    raise ValueError(f"Unknown schedule: {schedule_type}")


class CriticalReLUNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        h_layers,
        sigma_w_sq=2.0,
        sigma_b_sq=0.0,
    ):
        super().__init__()
        self.depth = len(h_layers)
        self.sigma_w_sq = sigma_w_sq
        self.sigma_b_sq = sigma_b_sq

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.dropouts = nn.ModuleList([nn.Dropout(p=h) for h in h_layers])
        self._init_critical()

    def _init_critical(self):
        for layer in self.layers:
            fan_in = layer.weight.shape[1]
            std_w = np.sqrt(self.sigma_w_sq / fan_in)
            nn.init.normal_(layer.weight, mean=0.0, std=std_w)
            if self.sigma_b_sq > 0:
                nn.init.normal_(layer.bias, mean=0.0, std=np.sqrt(self.sigma_b_sq))
            else:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            x = self.dropouts[i](x)
        return self.layers[-1](x)


def autocast_context():
    if not USE_AMP:
        return contextlib.nullcontext()
    return autocast(enabled=True, dtype=AMP_DTYPE)


def iterate_batches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    idx = (
        torch.randperm(n, device=x.device)
        if shuffle
        else torch.arange(n, device=x.device)
    )
    for i in range(0, n, batch_size):
        yield x[idx[i : i + batch_size]], y[idx[i : i + batch_size]]


def train_epoch(model, data, optimizer, criterion, batch_size):
    model.train()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in iterate_batches(x, y, batch_size):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            out = model(xb)
            loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)
    return loss_sum / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, data, criterion, batch_size):
    model.eval()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in iterate_batches(x, y, batch_size, shuffle=False):
        with autocast_context():
            out = model(xb)
            loss = criterion(out, yb)
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)
    return loss_sum / total, 100 * correct / total
