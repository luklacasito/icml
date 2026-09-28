"""Extend two frozen notebook comparisons without retuning or validation selection.

These original studies evaluate the test set every epoch. Their minimum test
loss is retrospective; it is not a validation-selected estimate. Each completed
fit saves its final weights, all curves, dataset fingerprints and source hashes.
Interrupted fits restart from their seed; completed, verified fits are skipped.

The ReLU notebook loader emits non-contiguous NCHW tensors, incompatible with
the model's ``view(batch, -1)`` in current PyTorch. We make those inputs contiguous
without changing their NCHW feature order. This recorded layout fix and the new
runtime mean these extensions are not claimed to be byte-identical old replays.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from .original import relu, vit
from utils import schedules as vit_schedules
from utils import training as vit_training

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = Path(__file__).resolve().parent / "original"
RECIPES = json.loads((SUPPORT / "recipes.json").read_text())
SCHEMA_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance():
    paths = [
        Path(__file__).resolve(),
        *sorted(SUPPORT.glob("*.py")),
        *sorted(SUPPORT.glob("*.json")),
        ROOT / "utils" / "training.py",
        ROOT / "utils" / "schedules.py",
    ]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    return {
        "files": hashes,
        "sha256": hashlib.sha256(canonical(hashes).encode()).hexdigest(),
    }


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_json(path, value):
    atomic_write(path, lambda stream: stream.write((canonical(value) + "\n").encode()))


@contextlib.contextmanager
def fit_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns this fit: {directory}") from error
        yield


def specification(study, arm, seed, *, smoke=False):
    if study not in RECIPES or arm not in ("uniform", "frontloaded") or seed < 0:
        raise ValueError("Expected study relu|vit, arm uniform|frontloaded and seed >= 0")
    recipe = dict(RECIPES[study]["recipe"])
    if smoke:
        recipe.update(epochs=2, batch_size=3, train_size=6, test_size=4)
    schedule = (
        "constant" if arm == "uniform" else ("big_step" if study == "relu" else "reverse_step")
    )
    get_schedule = (
        relu.get_dropout_schedule if study == "relu" else vit_schedules.get_dropout_schedule
    )
    recipe["dropout_probabilities"] = get_schedule(schedule, recipe["depth"], 0.1, 0.2)
    return {
        "schema_version": SCHEMA_VERSION,
        "study": study,
        "arm": arm,
        "seed": seed,
        "schedule": schedule,
        "recipe": recipe,
        "smoke": smoke,
        "protocol": "original_no_validation_fixed_recipe_extension_v1",
        "accuracy_unit": "percent",
        "selection": "none; test minima are retrospective",
        "layout_adapter": "contiguous_NCHW" if study == "relu" else "none",
    }


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(canonical({"shape": array.shape, "dtype": str(array.dtype)}).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def subset_indices(train_count, test_count, train_size, test_size, seed=0):
    rng = np.random.RandomState(seed)
    train = (
        rng.choice(train_count, train_size, replace=False)
        if train_size < train_count
        else np.arange(train_count)
    )
    test = (
        rng.choice(test_count, test_size, replace=False)
        if test_size < test_count
        else np.arange(test_count)
    )
    return train, test


def prepare_inputs(images, targets, indices, device, study):
    """Match notebook normalization; explicitly repair only the MLP layout."""
    x = torch.from_numpy(images[indices]).permute(0, 3, 1, 2).float()
    x = x / 255 if study == "relu" else x.div_(255)
    y = torch.tensor(np.asarray(targets)[indices])
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    x, y, mean, std = (value.to(device) for value in (x, y, mean, std))
    x = (x - mean) / std
    if study == "relu":
        x = x.contiguous()
    return x, y


def load_data(spec, data_root, device, expected_cache_sha256=None):
    if spec["smoke"]:
        rng = np.random.RandomState(20260921)
        train_images = rng.randint(0, 256, size=(6, 32, 32, 3), dtype=np.uint8)
        test_images = rng.randint(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
        train_targets, test_targets = np.arange(6) % 10, np.arange(4) % 10
        cache = {"synthetic": "seed_20260921_uint8_6_train_4_test"}
    else:
        from torchvision.datasets import CIFAR10

        # CIFAR10 verifies the official extracted-file MD5 hashes on loading.
        train_set = CIFAR10(data_root, train=True, download=True)
        test_set = CIFAR10(data_root, train=False, download=True)
        train_images, train_targets = train_set.data, train_set.targets
        test_images, test_targets = test_set.data, test_set.targets
        base = Path(data_root) / train_set.base_folder
        filenames = sorted({name for name, _ in train_set.train_list + train_set.test_list})
        filenames.append(train_set.meta["filename"])
        cache = {name: sha256_file(base / name) for name in filenames}
    cache_sha = hashlib.sha256(canonical(cache).encode()).hexdigest()
    if expected_cache_sha256 and cache_sha != expected_cache_sha256:
        raise RuntimeError(
            f"CIFAR cache hash mismatch: expected {expected_cache_sha256}, got {cache_sha}"
        )
    recipe = spec["recipe"]
    tr, te = subset_indices(
        len(train_images),
        len(test_images),
        recipe["train_size"],
        recipe["test_size"],
        recipe["split_seed"],
    )
    provenance = {
        "cache_files_sha256": cache,
        "cache_sha256": cache_sha,
        "train_indices_sha256": array_hash(tr),
        "test_indices_sha256": array_hash(te),
        "train_size": len(tr),
        "test_size": len(te),
        "split_seed": recipe["split_seed"],
        "sampling": "RandomState(seed).choice train then test; without replacement; full sets in original order",
        "train_labels_sha256": array_hash(np.asarray(train_targets)[tr]),
        "test_labels_sha256": array_hash(np.asarray(test_targets)[te]),
    }
    provenance["sha256"] = hashlib.sha256(canonical(provenance).encode()).hexdigest()
    return (
        prepare_inputs(train_images, train_targets, tr, device, spec["study"]),
        prepare_inputs(test_images, test_targets, te, device, spec["study"]),
        provenance,
    )


def build_model(spec):
    recipe = spec["recipe"]
    if spec["study"] == "relu":
        return relu.CriticalReLUNet(
            3072,
            recipe["width"],
            10,
            recipe["dropout_probabilities"],
            sigma_w_sq=recipe["sigma_w_sq"],
            sigma_b_sq=recipe["sigma_b_sq"],
        )
    return vit.ViT(
        recipe["depth"],
        recipe["embedding_dim"],
        recipe["heads"],
        recipe["mlp_ratio"],
        recipe["dropout_probabilities"],
        ablation_mode="both",
    )


def resolve_device(value):
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu or cuda devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; refusing an unnoticed CPU run"
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def verified_result(directory, spec, source, data):
    path = directory / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    for name, value in (("spec", spec), ("source", source), ("data", data)):
        if result.get(name) != value:
            raise RuntimeError(f"Existing result has a different {name}: {path}")
    epochs = spec["recipe"]["epochs"]
    if result.get("status") != "complete" or any(
        len(result["curves"].get(key, [])) != epochs
        for key in (
            "train_loss",
            "test_loss",
            "train_accuracy_percent",
            "test_accuracy_percent",
        )
    ):
        raise RuntimeError(f"Existing completion is incomplete: {path}")
    checkpoint = directory / result["final_checkpoint"]["file"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != result["final_checkpoint"]["sha256"]:
        raise RuntimeError(f"Final checkpoint is missing or changed: {checkpoint}")
    return result


def run_trial(
    study,
    arm,
    seed,
    data_root,
    run_root,
    device,
    *,
    smoke=False,
    expected_cache_sha256=None,
):
    spec = specification(study, arm, seed, smoke=smoke)
    device = resolve_device(device)
    directory = (
        Path(run_root) / ("original-smoke" if smoke else "original") / study / arm / f"seed-{seed}"
    )
    source = source_provenance()
    with fit_lock(directory):
        train_data, test_data, data = load_data(spec, data_root, device, expected_cache_sha256)
        existing = verified_result(directory, spec, source, data)
        if existing is not None:
            print(
                canonical(
                    {
                        "status": "skipped_verified_complete",
                        "result": str(directory / "result.json"),
                    }
                ),
                flush=True,
            )
            return existing
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.reset_peak_memory_stats(device)
        amp_dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        relu.USE_AMP, relu.AMP_DTYPE = device.type == "cuda", amp_dtype
        runtime = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "amp_dtype": str(amp_dtype) if device.type == "cuda" else None,
        }
        # Exactly the notebooks' order: reset both RNGs, construct model on CPU,
        # move to device, construct optimizer and train. No separate loader RNG.
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(spec).to(device)
        recipe = spec["recipe"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=recipe["learning_rate"],
            weight_decay=recipe["weight_decay"],
        )
        lr_floor = 1e-7 if study == "relu" else 5e-6
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=recipe["epochs"], eta_min=lr_floor
        )
        criterion = torch.nn.CrossEntropyLoss()
        curves = {
            key: []
            for key in (
                "train_loss",
                "test_loss",
                "train_accuracy_percent",
                "test_accuracy_percent",
                "learning_rate_after_step",
            )
        }
        start = time.monotonic()
        for epoch in range(recipe["epochs"]):
            if study == "relu":
                train_loss, train_accuracy = relu.train_epoch(
                    model, train_data, optimizer, criterion, recipe["batch_size"]
                )
            else:
                train_loss, train_accuracy = vit_training.train_epoch(
                    model,
                    train_data,
                    optimizer,
                    criterion,
                    recipe["batch_size"],
                    grad_clip=1.0,
                    amp_dtype=amp_dtype,
                )
            scheduler.step()
            if study == "relu":
                test_loss, test_accuracy = relu.evaluate(
                    model, test_data, criterion, recipe["batch_size"]
                )
            else:
                test_loss, test_accuracy = vit_training.evaluate(
                    model,
                    test_data,
                    criterion,
                    recipe["batch_size"],
                    amp_dtype=amp_dtype,
                )
            values = (
                train_loss,
                test_loss,
                train_accuracy,
                test_accuracy,
                optimizer.param_groups[0]["lr"],
            )
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(
                    f"Nonfinite metric at epoch {epoch}; not marking this fit complete"
                )
            for key, value in zip(curves, values):
                curves[key].append(value)
            progress = {
                "status": "running",
                "spec": spec,
                "source": source,
                "data": data,
                "runtime": runtime,
                "last_epoch_zero_based": epoch,
                "curves": curves,
                "elapsed_seconds": time.monotonic() - start,
                "interruption_policy": "Restart unfinished fits from epoch zero with the same seed",
            }
            atomic_json(directory / "progress.json", progress)
            print(
                canonical(
                    {
                        "study": study,
                        "arm": arm,
                        "seed": seed,
                        "epoch": epoch + 1,
                        "epochs": recipe["epochs"],
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "test_accuracy_percent": test_accuracy,
                    }
                ),
                flush=True,
            )
        minimum_epoch = int(np.argmin(curves["test_loss"]))
        final_epoch = recipe["epochs"] - 1
        checkpoint = directory / "final.pt"
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        atomic_write(
            checkpoint,
            lambda stream: torch.save(
                {
                    "model_state_dict": state,
                    "spec": spec,
                    "source_sha256": source["sha256"],
                    "data_sha256": data["sha256"],
                    "epoch_zero_based": final_epoch,
                    "endpoint": "final_epoch",
                },
                stream,
            ),
        )
        result = {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "spec": spec,
            "source": source,
            "data": data,
            "runtime": runtime,
            "curves": curves,
            "elapsed_seconds": time.monotonic() - start,
            "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "metrics": {
                "final_epoch_test_loss": curves["test_loss"][-1],
                "final_epoch_test_accuracy_percent": curves["test_accuracy_percent"][-1],
                "final_epoch_zero_based": final_epoch,
                "retrospective_min_test_loss": curves["test_loss"][minimum_epoch],
                "retrospective_min_test_epoch_zero_based": minimum_epoch,
                "test_accuracy_percent_at_retrospective_min_test_loss": curves[
                    "test_accuracy_percent"
                ][minimum_epoch],
                "retrospective_max_test_accuracy_percent": max(curves["test_accuracy_percent"]),
            },
            "final_checkpoint": {"file": "final.pt", "sha256": sha256_file(checkpoint)},
            "interpretation": "No validation set. Min-test and max-accuracy endpoints are retrospective; do not label them validation-selected.",
            "pooling": "Separate extension cohort; verify compatibility before combining with historical notebook results.",
        }
        atomic_json(directory / "result.json", result)
        atomic_json(directory / "progress.json", {"status": "complete", "result": "result.json"})
        print(
            canonical({"status": "complete", "result": str(directory / "result.json")}),
            flush=True,
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--study", choices=sorted(RECIPES), required=True)
    run.add_argument("--arm", choices=("uniform", "frontloaded"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument(
        "--cache-sha256",
        help="Expected aggregate SHA256 of the official CIFAR batch/meta files",
    )
    smoke_parser = sub.add_parser("smoke")
    for command in (run, smoke_parser):
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--device", default="cuda" if command is run else "cpu")
    args = parser.parse_args()
    if args.command == "run":
        run_trial(
            args.study,
            args.arm,
            args.seed,
            args.data_root,
            args.run_root,
            args.device,
            expected_cache_sha256=args.cache_sha256,
        )
    else:
        torch.set_num_threads(min(2, torch.get_num_threads()))
        for study in RECIPES:
            for arm in ("uniform", "frontloaded"):
                result = run_trial(study, arm, 900001, None, args.run_root, args.device, smoke=True)
                repeated = run_trial(
                    study, arm, 900001, None, args.run_root, args.device, smoke=True
                )
                assert result == repeated
        print(
            "Original model smoke and verified-completion restart checks passed.",
            flush=True,
        )


if __name__ == "__main__":
    main()
