"""Shared epoch and seed loops for the interactive experiment notebooks."""

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

METRICS = ("train_acc", "test_acc", "train_loss", "test_loss")


def fit_history(
    model,
    train_data,
    test_data,
    optimizer,
    criterion,
    batch_size,
    progress,
    *,
    train,
    evaluate,
    scheduler=None,
    evaluate_every=1,
    on_epoch=None,
):
    """Record every epoch, carrying test values forward between evaluations."""
    history = {key: [] for key in METRICS}
    best = 0.0
    for epoch in progress:
        train_loss, train_acc = train(model, train_data, optimizer, criterion, batch_size)
        if scheduler is not None:
            scheduler.step()
        if epoch % evaluate_every == 0 or epoch == len(progress) - 1:
            test_loss, test_acc = evaluate(model, test_data, criterion, batch_size)
        else:
            test_loss, test_acc = history["test_loss"][-1], history["test_acc"][-1]
        best = max(best, test_acc)
        if on_epoch is not None:
            on_epoch(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/best_acc": best,
                    "lr": optimizer.param_groups[0]["lr"],
                    "gen_gap": train_acc - test_acc,
                }
            )
        for key, value in zip(METRICS, (train_acc, test_acc, train_loss, test_loss)):
            history[key].append(value)
        progress.set_postfix(
            tr_L=f"{train_loss:.3f}",
            tr_A=f"{train_acc:.1f}%",
            te_L=f"{test_loss:.3f}",
            te_A=f"{test_acc:.1f}%",
            best=f"{best:.1f}%",
        )
    return history


def run_simulations(
    model_factory,
    train_data,
    test_data,
    n_sims,
    label,
    *,
    epochs,
    batch_size,
    learning_rate,
    lr_min,
    train,
    evaluate,
):
    """Run the MLP sweep's original seed sequence and AdamW recipe."""
    histories = []
    for sim in range(n_sims):
        torch.manual_seed(42 + sim)
        np.random.seed(42 + sim)
        model = model_factory()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-7)
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
            if lr_min is not None
            else None
        )
        criterion = nn.CrossEntropyLoss()
        progress = tqdm(range(epochs), desc=f"{label} sim {sim + 1}/{n_sims}", leave=True)
        histories.append(
            fit_history(
                model,
                train_data,
                test_data,
                optimizer,
                criterion,
                batch_size,
                progress,
                train=train,
                evaluate=evaluate,
                scheduler=scheduler,
            )
        )
    return {key: np.array([history[key] for history in histories]) for key in METRICS}
