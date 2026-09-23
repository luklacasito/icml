"""Historical benchmark optimization and validation-checkpoint selection.

Extracted without changes to the optimization loop; both test endpoints are
reported only after training. See README.md for the execution provenance.
"""

from __future__ import annotations
import contextlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Literal
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DatasetName = Literal["fi2010", "tiny_imagenet", "speech_commands", "openml_jannis"]


@dataclass(frozen=True)
class DatasetBundle:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    split_hash: str
    dataset: DatasetName
    split_protocol: str
    test_subset_hash: str
    test_subset_protocol: str
    test_subset_seed: int | None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 75
    batch_size: int = 75
    learning_rate: float = 1e-4
    lr_floor_ratio: float = 1e-3
    weight_decay: float = 1e-7
    gradient_clip_norm: float | None = None
    seed: int = 0
    stochastic_seed: int | None = None
    evaluate_test: bool = True
    # When true, the primary test endpoint is the minimum-validation-loss
    # checkpoint.  Benchmark confirmation additionally records the fixed final
    # epoch as a preregistered secondary endpoint; neither test metric is used
    # for model or hyperparameter selection. The benchmark runner sets this
    # flag explicitly to true.
    restore_best_validation: bool = False
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multiplicative_cosine_factor(
    epoch: int,
    *,
    epochs: int,
    floor_ratio: float,
) -> float:
    progress = min(max(epoch, 0), epochs) / epochs
    return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_multiplicative_cosine_scheduler(
    optimizer,
    *,
    epochs: int,
    floor_ratio: float = 1e-3,
):
    # LambdaLR is stepped after each epoch.  Using epochs - 1 as the decay
    # horizon makes the multiplier used by the final training epoch equal to
    # floor_ratio, while the first epoch still uses the original group LRs.
    decay_steps = max(epochs - 1, 1)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: multiplicative_cosine_factor(
            epoch, epochs=decay_steps, floor_ratio=floor_ratio
        ),
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(dataset: TensorDataset, batch_size: int, *, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss_sum += float(criterion(logits, targets))
        correct += int((logits.argmax(dim=1) == targets).sum())
        count += targets.numel()
    return loss_sum / count, correct / count


def train_model(
    model: nn.Module,
    optimizer,
    bundle: DatasetBundle,
    config: TrainingConfig,
    *,
    return_best_state: bool = False,
    return_final_state: bool = False,
) -> dict:
    """Fit on training data, select by validation loss, then evaluate test endpoints."""

    # Model initialization happens before this function.  The optional
    # stochastic seed gives dropout an independent, deterministic stream while
    # the explicit loader generator continues to pair minibatch order by seed.
    seed_everything(config.seed if config.stochastic_seed is None else config.stochastic_seed)
    device = _resolve_device(config.device)
    model.to(device)
    train_loader = _loader(bundle.train, config.batch_size, shuffle=True, seed=config.seed)
    validation_loader = _loader(
        bundle.validation, config.batch_size, shuffle=False, seed=config.seed
    )
    test_loader = (
        _loader(bundle.test, config.batch_size, shuffle=False, seed=config.seed)
        if config.evaluate_test
        else None
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = make_multiplicative_cosine_scheduler(
        optimizer,
        epochs=config.epochs,
        floor_ratio=config.lr_floor_ratio,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "lr_multiplier": [],
        "optimizer_group_lrs": [],
        "optimizer_steps": [],
    }
    optimizer_steps = 0
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.cuda.amp.autocast(dtype=torch.bfloat16)
                if use_amp and torch.cuda.is_bf16_supported()
                else torch.cuda.amp.autocast(enabled=use_amp)
            )
            with autocast if use_amp else contextlib.nullcontext():
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if config.gradient_clip_norm is not None:
                # Unscale before clipping so the configured norm has the same
                # meaning under CUDA AMP as in the original ViT recipe.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * targets.numel()
            correct += int((logits.detach().argmax(dim=1) == targets).sum())
            count += targets.numel()

        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        if config.restore_best_validation and validation_loss < best_validation_loss:
            # Strict `<` keeps the first minimum, matching `np.argmin` on the
            # recorded validation curve, so `selected_epoch` and the epoch the
            # test set is evaluated at cannot disagree.
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in model.state_dict().items()
            }
        history["train_loss"].append(loss_sum / count)
        history["train_accuracy"].append(correct / count)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        multiplier = multiplicative_cosine_factor(
            epoch,
            epochs=max(config.epochs - 1, 1),
            floor_ratio=config.lr_floor_ratio,
        )
        history["lr_multiplier"].append(multiplier)
        history["optimizer_group_lrs"].append(
            [float(group["lr"]) for group in optimizer.param_groups]
        )
        history["optimizer_steps"].append(optimizer_steps)
        scheduler.step()

    # Snapshot before restoring the selected checkpoint. Copying tensors does
    # not consume RNG or change the optimization trajectory.
    final_state = (
        {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()}
        if return_final_state
        else None
    )
    final_epoch_test_loss, final_epoch_test_accuracy = None, None
    if test_loader is not None and config.restore_best_validation:
        # The model still holds the fixed-final-epoch weights.  Reading the test
        # set here is reporting only: training and checkpoint selection have
        # already finished, and neither endpoint feeds a decision.
        final_epoch_test_loss, final_epoch_test_accuracy = evaluate(model, test_loader, device)

    if config.restore_best_validation:
        test_epoch = best_epoch
        test_protocol = "best_validation_epoch_single_evaluation_v1"
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        test_epoch = config.epochs - 1
        test_protocol = "final_epoch_single_evaluation_v1"
    if test_loader is None:
        test_loss, test_accuracy = None, None
    elif config.restore_best_validation and test_epoch == config.epochs - 1:
        # Both preregistered endpoints coincide, so the fixed-final pass above
        # is also the selected-checkpoint evaluation.
        test_loss, test_accuracy = final_epoch_test_loss, final_epoch_test_accuracy
    elif config.evaluate_test:
        test_loss, test_accuracy = evaluate(model, test_loader, device)
    else:
        test_loss, test_accuracy = None, None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    examples_seen = len(bundle.train) * config.epochs
    # Standard dense-training estimate: forward + backward ~= 6 FLOPs/parameter/example.
    estimated_flops = 6 * parameter_count * examples_seen
    result = {
        "training_config": asdict(config),
        "dataset": bundle.dataset,
        "split_hash": bundle.split_hash,
        "history": {key: np.asarray(value) for key, value in history.items()},
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "final_epoch_test_loss": final_epoch_test_loss,
        "final_epoch_test_accuracy": final_epoch_test_accuracy,
        "final_epoch_test_epoch": config.epochs - 1,
        "final_epoch_test_protocol": "fixed_final_epoch_single_evaluation_v1",
        "test_epoch": test_epoch,
        "test_protocol": test_protocol,
        "optimizer_steps": optimizer_steps,
        "parameter_count": parameter_count,
        "estimated_training_flops": estimated_flops,
        "examples_seen": examples_seen,
        "device": str(device),
    }
    if return_best_state:
        if not config.restore_best_validation:
            raise ValueError("return_best_state requires restore_best_validation=True")
        if best_state is None:  # pragma: no cover - positive epochs guarantee this
            raise RuntimeError("No best-validation checkpoint was captured")
        result["best_state_dict"] = best_state
    if return_final_state:
        result["final_state_dict"] = final_state
    return result
