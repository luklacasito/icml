"""Small checks for the paper's schedules and sample-weighted metrics."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from utils import (
    evaluate,
    get_dropout_schedule,
    iterate_batches,
    load_cifar,
    train_epoch,
)
from utils import training


@pytest.mark.parametrize("depth", [1, 5, 12])
@pytest.mark.parametrize("name", ["constant", "linear", "reverse_linear", "step", "reverse_step"])
def test_matched_schedules_preserve_mean(name, depth):
    rates = get_dropout_schedule(name, depth, h_bar=0.1, h_max=0.27)
    assert len(rates) == depth
    assert np.mean(rates) == pytest.approx(0.1)
    assert all(0 <= rate < 1 for rate in rates)


def test_schedule_direction_and_budget_controls():
    for early, late in [("reverse_linear", "linear"), ("reverse_step", "step")]:
        front = get_dropout_schedule(early, 12, 0.1)
        back = get_dropout_schedule(late, 12, 0.1)
        assert front == pytest.approx(back[::-1])
        assert front[0] > front[-1]

    for name, mean in [("none", 0), ("double", 0.2), ("triple", 0.3)]:
        assert np.mean(get_dropout_schedule(name, 12, 0.1)) == pytest.approx(mean)

    # Big step retains the paper's integer rounding, not a rescaled budget.
    assert np.mean(get_dropout_schedule("big_step", 12, 0.1)) == pytest.approx(0.1)
    assert get_dropout_schedule("big_step", 5, 0.1) == pytest.approx([0.3, 0, 0, 0, 0])


@pytest.mark.parametrize("name", ["step", "reverse_step"])
@pytest.mark.parametrize("h_max", [None, 0, 0.2])
def test_zero_dropout_step_is_a_no_dropout_schedule(name, h_max):
    assert get_dropout_schedule(name, 6, 0, h_max) == [0.0] * 6


def test_batches_keep_pairs_and_include_remainder():
    x = torch.arange(5).unsqueeze(1)
    y = torch.arange(5) + 10
    batches = list(iterate_batches(x, y, bs=2, shuffle=False))
    assert [len(xb) for xb, _ in batches] == [2, 2, 1]
    assert torch.equal(torch.cat([xb for xb, _ in batches]), x)

    batches = list(iterate_batches(x, y, bs=2))
    shuffled_x = torch.cat([xb for xb, _ in batches]).squeeze(1)
    shuffled_y = torch.cat([yb for _, yb in batches])
    assert torch.equal(shuffled_x.sort().values, x.squeeze(1))
    assert torch.equal(shuffled_y, shuffled_x + 10)


def test_evaluation_is_independent_of_batch_size():
    logits = torch.tensor([[5.0, 0.0], [0.0, 2.0], [1.0, 3.0], [4.0, 1.0], [0.0, 6.0]])
    targets = torch.tensor([0, 1, 0, 1, 0])
    criterion = nn.CrossEntropyLoss()
    model = nn.Dropout(p=0.9)
    expected_loss = criterion(logits, targets).item()
    for batch_size in (2, 5):
        model.train()
        loss, accuracy = evaluate(model, (logits, targets), criterion, batch_size)
        assert loss == pytest.approx(expected_loss)
        assert accuracy == pytest.approx(40.0)
        assert not model.training


def test_training_updates_model_and_reduces_loss():
    x = torch.ones(5, 2)
    y = torch.ones(5, dtype=torch.long)
    model = nn.Linear(2, 2)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    before, _ = evaluate(model, (x, y), criterion, bs=2)
    loss, accuracy = train_epoch(model, (x, y), optimizer, criterion, bs=2, grad_clip=1.0)
    assert model.training
    assert np.isfinite(loss)
    assert 0 <= accuracy <= 100
    after, final_accuracy = evaluate(model, (x, y), criterion, bs=2)
    assert after < before
    assert final_accuracy == 100


def test_cifar_path_and_seeded_subsets(monkeypatch, tmp_path):
    calls = []

    class FakeCIFAR:
        def __init__(self, root, train, download):
            calls.append((Path(root), train, download))
            self.data = np.broadcast_to(
                np.arange(6, dtype=np.uint8)[:, None, None, None], (6, 2, 2, 3)
            ).copy()
            self.targets = list(range(6))

    monkeypatch.setitem(training._CIFAR_CLASS, 10, FakeCIFAR)
    monkeypatch.chdir(tmp_path)
    train, test = load_cifar(10, "cpu", train_size=4, test_size=3, seed=7)
    repo_data = Path(training.__file__).resolve().parents[1] / "data"
    assert calls == [(repo_data, True, True), (repo_data, False, True)]

    rng = np.random.RandomState(7)
    train_indices = rng.choice(6, 4, replace=False)
    test_indices = rng.choice(6, 3, replace=False)
    assert train[1].tolist() == train_indices.tolist()
    assert test[1].tolist() == test_indices.tolist()
    assert train[0].shape == (4, 3, 2, 2)
    assert train[0].device.type == "cpu"
    expected_red = (torch.tensor(train_indices).float() / 255 - 0.4914) / 0.2470
    torch.testing.assert_close(train[0][:, 0, 0, 0], expected_red)

    # The meta device exercises device routing without requiring GPU hardware.
    train, test = load_cifar(10, "meta", train_size=4, test_size=3, seed=7)
    assert all(tensor.device.type == "meta" for tensor in (*train, *test))

    with pytest.raises(ValueError, match="exceeds"):
        load_cifar(10, "cpu", train_size=7)
