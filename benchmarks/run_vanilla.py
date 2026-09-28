"""Plan, reproduce, and smoke-test the standalone FI-2010 vanilla MLP recipe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .vanilla import (
    ARCHITECTURES,
    LEARNING_RATES,
    PROFILES,
    VanillaMLP,
    digest,
    fingerprint,
    load_data,
    prepare_segments,
    trial,
)


def source_hash():
    """Fingerprint the two public files that determine this independent recipe."""
    value = hashlib.sha256()
    for name in ("vanilla.py", "run_vanilla.py"):
        path = Path(__file__).with_name(name)
        value.update(name.encode() + b"\0" + path.read_bytes())
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng(model, shuffle):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "shuffle": shuffle.get_state(),
        "dropout": [module.rng_state() for module in model.dropouts],
    }


def restore_rng(state, model, shuffle):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    shuffle.set_state(state["shuffle"])
    for module, value in zip(model.dropouts, state["dropout"], strict=True):
        module.restore_rng(value)


def metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    f1_denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(
        2 * confusion.diagonal(),
        f1_denominator,
        out=np.zeros(3),
        where=f1_denominator != 0,
    )
    correct_probability = np.clip(probabilities[np.arange(len(labels)), labels], 1e-300, 1)
    return {
        "n": len(labels),
        "logloss": float(-np.log(correct_probability).mean()),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1.mean()),
        "brier": float(((probabilities - np.eye(3)[labels]) ** 2).sum(axis=1).mean()),
        "confusion": confusion.tolist(),
        "per_class_f1": f1.tolist(),
    }


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, return_predictions=False):
    model.eval()
    labels, probabilities, indexes = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for inputs, targets, batch_indexes in loader:
        logits = model(inputs.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        labels.append(targets.numpy())
        probabilities.append(logits.double().softmax(-1).cpu().numpy())
        if return_predictions:
            indexes.append(batch_indexes.numpy())
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    result = metrics(labels, probabilities)
    if not return_predictions:
        return result, None
    return result, {
        "labels": labels,
        "probabilities": probabilities,
        "indexes": np.concatenate(indexes),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, dataset, optimizer, shuffle, batch_size, device, clip_norm):
    model.train()
    loss_sum = 0.0
    example_count = 0
    gradient_norm_sum = 0.0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=shuffle,
    )
    synchronize(device)
    started = time.monotonic()
    for inputs, targets, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs.to(device))
        loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        loss_sum += float(loss.detach()) * len(targets)
        example_count += len(targets)
        gradient_norm_sum += float(gradient_norm)
    synchronize(device)
    elapsed = time.monotonic() - started
    return {
        "dropout_on_logloss": loss_sum / example_count,
        "n": example_count,
        "seconds": elapsed,
        "examples_per_second": example_count / elapsed,
        "last_grad_norm": float(gradient_norm),
        "mean_grad_norm": gradient_norm_sum / len(loader),
        "optimizer_steps": len(loader),
    }


@torch.no_grad()
def activation_probe(model, dataset, device):
    model.eval()
    sample_indexes = np.linspace(0, len(dataset) - 1, 8, dtype=int)
    inputs = torch.stack([dataset[int(index)][0] for index in sample_indexes]).to(device)
    variances = []
    handles = [
        dropout.register_forward_pre_hook(
            lambda _module, values: variances.append(float(values[0].var(unbiased=False)))
        )
        for dropout in model.dropouts
    ]
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return variances


def _save_test_predictions(path, dataset, predictions):
    underlying = dataset.dataset if isinstance(dataset, Subset) else dataset
    predictions["day_stock_window_start_label_endpoint"] = underlying.example_ids(
        predictions["indexes"]
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **predictions)
    temporary.replace(path)


def initialize_training(spec, device):
    """Seed the independent streams, then build the model, optimizer and shuffle RNG."""
    streams = spec["randomization"]
    random.seed(streams["initialization_seed"])
    np.random.seed(streams["initialization_seed"])
    torch.manual_seed(streams["initialization_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(streams["initialization_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = VanillaMLP(
        input_dim=spec["input_shape"][0] * spec["input_shape"][1],
        width=spec["width"],
        depth=spec["depth"],
        probabilities=spec["dropout_probabilities"],
        dropout_seed=streams["dropout_seed"],
        output_dim=spec["output_dim"],
        sigma_w_sq=spec["initialization"]["sigma_w_sq"],
        sigma_b_sq=spec["initialization"]["sigma_b_sq"],
    ).to(device)
    config = spec["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    shuffle = torch.Generator().manual_seed(streams["minibatch_seed"])
    return model, optimizer, shuffle


def run(spec, data_root, output, device="cuda", stop_after_epoch=None, check_source=True):
    """Run one trial. ``stop_after_epoch`` exists only to test exact resume."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(spec)
    if check_source and source_hash() != spec["source_sha256"]:
        raise ValueError("Source differs from immutable plan")
    if digest(Path(data_root) / "manifest.json") != spec["data_sha256"]:
        raise ValueError("Dataset manifest differs from immutable plan")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = output / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["fingerprint"] != identity or result["status"] != "complete":
                raise ValueError("Output contains a different trial")
            return result
        spec_path = output / "spec.json"
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError("Output directory contains a different trial spec")
        atomic_json(spec_path, spec)
        if spec["stage"] not in {
            "canary",
            "lr_search",
            "confirmation",
            "test_fixture",
        }:
            raise ValueError("Unknown stage")

        include_test = spec["stage"] == "confirmation"
        datasets, data_info = load_data(
            data_root, include_test=include_test, sequence=spec["input_shape"][0]
        )
        if "canary_examples" in spec:
            if spec["stage"] not in {"canary", "test_fixture"}:
                raise ValueError("Subsampling is forbidden in scientific runs")
            limit = spec["canary_examples"]
            datasets = {
                name: Subset(
                    dataset,
                    np.linspace(0, len(dataset) - 1, min(limit, len(dataset)), dtype=int).tolist(),
                )
                for name, dataset in datasets.items()
            }

        model, optimizer, shuffle = initialize_training(spec, device)
        config = spec["training"]
        runtime = {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "device_type": device.type,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        atomic_json(output / "runtime.json", {**runtime, "data": data_info})

        # Restore optimizer and RNG state together so a resumed fit follows the same path.
        latest_path = output / "latest.pt"
        start_epoch, history, best = 0, [], None
        if latest_path.exists():
            state = torch.load(latest_path, map_location="cpu", weights_only=False)
            if state["fingerprint"] != identity or state["runtime"] != runtime:
                raise ValueError("Checkpoint spec or runtime differs from this trial")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"], model, shuffle)
            start_epoch = state["epoch"]
            history = state["history"]
            best = state["best"]

        for epoch in range(start_epoch, config["epochs"]):
            phase = epoch / max(config["epochs"] - 1, 1)
            multiplier = config["cosine_floor"] + (1 - config["cosine_floor"]) * 0.5 * (
                1 + math.cos(math.pi * phase)
            )
            learning_rate = config["learning_rate"] * multiplier
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            epoch_started = time.monotonic()
            training = train_epoch(
                model,
                datasets["train"],
                optimizer,
                shuffle,
                config["batch_size"],
                device,
                config["clip_norm"],
            )
            validation, _ = evaluate(model, datasets["validation"], config["batch_size"], device)
            record = {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "training": training,
                "validation": validation,
                "epoch_seconds": time.monotonic() - epoch_started,
            }
            history.append(record)
            if best is None or validation["logloss"] < best["validation"]["logloss"]:
                best = {
                    "epoch": epoch + 1,
                    "validation": validation,
                    "model": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                }
            atomic_checkpoint(
                latest_path,
                {
                    "fingerprint": identity,
                    "runtime": runtime,
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": capture_rng(model, shuffle),
                    "best": best,
                    "history": history,
                },
            )
            atomic_json(output / "history.json", history)
            print(
                json.dumps(
                    {
                        "trial": identity[:12],
                        "epoch": epoch + 1,
                        "training": training["dropout_on_logloss"],
                        "validation": validation["logloss"],
                        "seconds": record["epoch_seconds"],
                    }
                ),
                flush=True,
            )
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                return {"status": "checkpointed", "epoch": epoch + 1}

        # Select by validation loss before evaluating the held-out test data.
        model.load_state_dict(best["model"])
        final = {}
        for split in ("train", "validation"):
            final[split], _ = evaluate(model, datasets[split], config["batch_size"], device)
        if include_test:
            final["test"], predictions = evaluate(
                model,
                datasets["test"],
                config["batch_size"],
                device,
                return_predictions=True,
            )
            _save_test_predictions(output / "test_predictions.npz", datasets["test"], predictions)
        atomic_checkpoint(
            output / "best.pt",
            {"fingerprint": identity, "epoch": best["epoch"], "model": best["model"]},
        )
        result = {
            "status": "complete",
            "fingerprint": identity,
            "spec": spec,
            "runtime": runtime,
            "best_epoch": best["epoch"],
            "metrics": final,
            "data": data_info,
            "test_evaluated": include_test,
            "activation_variance_at_best": activation_probe(model, datasets["train"], device),
            "epoch_seconds": [record["epoch_seconds"] for record in history],
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
        atomic_json(result_path, result)
        latest_path.unlink()
        return result


def write_immutable(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Will not replace a different immutable plan: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_plans(data_hash, code_hash):
    canaries = [
        trial("canary", depth, width, "uniform", 3e-4, 0, 1, data_hash, code_hash)
        for depth, width in ARCHITECTURES
    ]
    lr_trials = [
        trial(
            "lr_search",
            depth,
            width,
            "uniform",
            learning_rate,
            seed,
            50,
            data_hash,
            code_hash,
        )
        for depth, width in ARCHITECTURES
        for learning_rate in LEARNING_RATES
        for seed in range(3)
    ]
    shared = {
        "study": "fi2010_vanilla_mlp_v1",
        "data_sha256": data_hash,
        "source_sha256": code_hash,
        "test_policy": "test loaded and evaluated only in confirmation",
    }
    return {
        "canary": {**shared, "stage": "canary", "trials": canaries},
        "lr_search": {
            **shared,
            "stage": "lr_search",
            "trials": lr_trials,
            "selection": {
                "criterion": "lowest mean best-validation logloss per depth-width cell",
                "tie_break": "lower learning rate",
                "confirmation_seeds": [100, 101, 102, 103, 104],
                "confirmation_epochs": 100,
            },
        },
    }


def load_result(root, index, spec):
    matches = list(root.glob(f"{index:03d}_{fingerprint(spec)[:12]}/result.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one result for LR trial {index}; found {len(matches)}")
    result = json.loads(matches[0].read_text())
    if result["status"] != "complete" or result["fingerprint"] != fingerprint(spec):
        raise ValueError(f"Invalid completion record for LR trial {index}")
    if result["spec"] != spec or result["test_evaluated"]:
        raise ValueError(f"Spec mismatch or premature test access in LR trial {index}")
    if len(result["epoch_seconds"]) != spec["training"]["epochs"]:
        raise ValueError(f"Incomplete epoch history in LR trial {index}")
    return result


def select_rates(plan, results):
    values = defaultdict(list)
    for index, spec in enumerate(plan["trials"]):
        result = load_result(results, index, spec)
        cell = (spec["depth"], spec["width"])
        values[cell, spec["training"]["learning_rate"]].append(
            result["metrics"]["validation"]["logloss"]
        )
    selected = {}
    for cell in sorted({key[0] for key in values}):
        candidates = []
        for (candidate_cell, learning_rate), observations in values.items():
            if candidate_cell == cell:
                if len(observations) != 3:
                    raise ValueError(f"Cell {cell}, lr {learning_rate} lacks three seeds")
                candidates.append((statistics.mean(observations), learning_rate))
        selected[cell] = min(candidates)[1]
    return selected, values


def make_confirmation(plan, selected):
    example = plan["trials"][0]
    trials = [
        trial(
            "confirmation",
            depth,
            width,
            profile,
            selected[(depth, width)],
            seed,
            plan["selection"]["confirmation_epochs"],
            example["data_sha256"],
            example["source_sha256"],
        )
        for depth, width in sorted(selected)
        for profile in PROFILES
        for seed in plan["selection"]["confirmation_seeds"]
    ]
    return {
        "study": plan["study"],
        "stage": "confirmation",
        "data_sha256": plan["data_sha256"],
        "source_sha256": plan["source_sha256"],
        "test_policy": plan["test_policy"],
        "selected_learning_rates": {
            f"depth{depth}_width{width}": rate for (depth, width), rate in selected.items()
        },
        "selection_source": "complete validated lr_search plan",
        "trials": trials,
    }


def make_smoke_data(root):
    """Write a tiny synthetic fixture, never a scientific benchmark cache."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    entries = []
    for day in range(6, 11):
        path = root / f"day{day}.npz"
        np.savez(
            path,
            x=rng.normal(size=(30, 40)).astype(np.float32),
            y=rng.integers(0, 3, size=(30, 1), dtype=np.int64),
        )
        entries.append({"day": day, "stock": 1, "file": path.name, "sha256": digest(path)})
    atomic_json(
        root / "manifest.json",
        {
            "schema": "fi2010_segments_v1",
            "synthetic_smoke_only": True,
            "tail_purge_representations": 10,
            "segments": entries,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="audit and convert FI-2010 decimal-precision releases"
    )
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser(
        "plan", help="write immutable 48-run validation-only LR screen and canaries"
    )
    plan.add_argument("--data-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    select = commands.add_parser(
        "select",
        help="select rates from the complete LR screen; write 140 confirmations",
    )
    select.add_argument("--plan", type=Path, required=True)
    select.add_argument("--results", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run", help="execute one immutable planned trial")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    execute.add_argument("--data-root", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    execute.add_argument("--threads", type=int, default=2)
    smoke = commands.add_parser("smoke", help="run two epochs on a tiny synthetic CPU fixture")
    smoke.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_segments(args.raw, args.output)
    elif args.command == "plan":
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((args.data_root / "manifest.json").read_text())
        if manifest.get("synthetic_smoke_only"):
            parser.error("Synthetic smoke data cannot create a scientific plan")
        plans = make_plans(digest(args.data_root / "manifest.json"), source_hash())
        for name, document in plans.items():
            path = args.output / f"{name}.json"
            write_immutable(path, document)
            print(f"{path}: {len(document['trials'])} trials")
    elif args.command == "select":
        document = json.loads(args.plan.read_text())
        if document["stage"] != "lr_search" or len(document["trials"]) != 48:
            parser.error("Expected the complete 48-trial LR-search plan")
        rates, _ = select_rates(document, args.results)
        confirmation = make_confirmation(document, rates)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_immutable(args.output, confirmation)
        print(json.dumps(confirmation["selected_learning_rates"], sort_keys=True))
        print(f"{args.output}: {len(confirmation['trials'])} trials")
    elif args.command == "run":
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        document = json.loads(args.plan.read_text())
        if not 0 <= args.index < len(document["trials"]):
            parser.error("--index is outside the plan")
        spec = document["trials"][args.index]
        output = args.output / f"{args.index:03d}_{fingerprint(spec)[:12]}"
        run(spec, args.data_root, output, args.device)
    elif args.command == "smoke":
        torch.set_num_threads(1)
        data_root = args.output / "synthetic-data"
        make_smoke_data(data_root)
        spec = trial(
            "test_fixture",
            6,
            8,
            "linear_early",
            1e-4,
            0,
            2,
            digest(data_root / "manifest.json"),
            source_hash(),
        )
        spec["input_shape"] = [8, 40]
        spec["training"]["batch_size"] = 8
        spec["canary_examples"] = 8
        result = run(spec, data_root, args.output / "synthetic-run", "cpu")
        print(
            json.dumps(
                {
                    "synthetic_smoke_only": True,
                    "status": result["status"],
                    "test_evaluated": result["test_evaluated"],
                }
            )
        )


if __name__ == "__main__":
    main()
