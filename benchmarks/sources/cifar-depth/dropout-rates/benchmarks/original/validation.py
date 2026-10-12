"""Select on validation loss; record test curves only as training diagnostics."""

import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from benchmarks import run_original as base
from benchmarks.original import relu

CURVES = ("train_loss", "validation_loss", "train_accuracy_percent",
          "validation_accuracy_percent", "learning_rate_after_step",
          "test_loss", "test_accuracy_percent")


def validation_indices(labels, train_indices, validation_size, seed):
    """Sample equally from classes 0--9 after excluding the original training rows."""
    labels, train_indices = np.asarray(labels), np.asarray(train_indices)
    if labels.ndim != 1 or validation_size <= 0 or validation_size % 10:
        raise ValueError("Validation size must be positive and divisible by ten")
    if (train_indices.ndim != 1 or not np.issubdtype(train_indices.dtype, np.integer)
            or np.any(train_indices < 0) or np.any(train_indices >= len(labels))
            or len(np.unique(train_indices)) != len(train_indices)):
        raise ValueError("Training indices must be unique valid integer rows")
    available = np.ones(len(labels), dtype=bool)
    available[train_indices] = False
    rng = np.random.RandomState(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(available & (labels == label)), validation_size // 10,
                   replace=False) for label in range(10)
    ])
    if len(np.unique(selected)) != validation_size or np.intersect1d(selected, train_indices).size:
        raise RuntimeError("Validation rows must be unique and disjoint from training")
    return selected


def load_validation(spec, data_root, device, original_data):
    """Return validation tensors and provenance without changing the train/test split."""
    recipe = spec["recipe"]
    seed, size = recipe["validation_split_seed"], recipe["validation_size"]
    if spec["smoke"]:
        if size != 10:
            raise ValueError("Synthetic validation requires ten examples")
        images = np.random.RandomState(seed).randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
        labels = np.arange(10)
        train_indices = np.array([], dtype=np.int64)
        old_rng = np.random.RandomState(20260921)
        old_images = np.concatenate([old_rng.randint(0, 256, (n, 32, 32, 3), dtype=np.uint8)
                                     for n in (6, 4)])
        if {base.array_hash(x) for x in images} & {base.array_hash(x) for x in old_images}:
            raise RuntimeError("Synthetic validation overlaps train/test images")
        source = "Independent synthetic images; classes 0--9, one example each"
    else:
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(data_root, train=True, download=False)
        images, labels = dataset.data, np.asarray(dataset.targets)
        train_indices, test_indices = base.subset_indices(
            len(images), 10000, recipe["train_size"], recipe["test_size"], recipe["split_seed"])
        if (base.array_hash(train_indices) != original_data["train_indices_sha256"]
                or base.array_hash(test_indices) != original_data["test_indices_sha256"]
                or base.array_hash(labels[train_indices]) != original_data["train_labels_sha256"]):
            raise RuntimeError("Validation construction does not match the original split")
        source = "Official CIFAR-10 training pool, excluding the original training subset"
    indices = validation_indices(labels, train_indices, size, seed)
    counts = np.bincount(labels[indices], minlength=10).tolist()
    if counts != [size // 10] * 10:
        raise RuntimeError("Validation classes must be balanced")
    provenance = {
        "validation_source": source, "validation_size": size,
        "validation_split_seed": seed, "validation_class_counts": counts,
        "validation_sampling": "RandomState(seed).choice per class 0--9, without replacement",
        "validation_train_disjoint": True,
        "validation_indices_sha256": base.array_hash(indices),
        "validation_labels_sha256": base.array_hash(labels[indices]),
        "validation_images_sha256": base.array_hash(images[indices]),
    }
    provenance["validation_sha256"] = hashlib.sha256(base.canonical(provenance).encode()).hexdigest()
    return base.prepare_inputs(images, labels, indices, device, spec["study"]), provenance


def evaluate(model, data, criterion, batch_size, device):
    """Evaluation must leave both dropout and minibatch random streams untouched."""
    cpu_state = torch.get_rng_state()
    shuffle_state = relu.SHUFFLE_GENERATOR.get_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    metrics = relu.evaluate(model, data, criterion, batch_size)
    if (not torch.equal(cpu_state, torch.get_rng_state())
            or not torch.equal(shuffle_state, relu.SHUFFLE_GENERATOR.get_state())
            or (cuda_state is not None and not torch.equal(cuda_state, torch.cuda.get_rng_state(device)))):
        raise RuntimeError("Evaluation consumed a training random stream")
    if not all(math.isfinite(value) for value in metrics):
        raise RuntimeError("Nonfinite evaluation metric")
    return metrics


def verified_result(directory, spec, source, data):
    path = directory / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    for key, expected in (("spec", spec), ("source", source), ("data", data)):
        if result.get(key) != expected:
            raise RuntimeError(f"Existing result has a different {key}: {path}")
    epochs, curves = spec["recipe"]["epochs"], result["curves"]
    if result.get("status") != "complete" or any(
            len(curves.get(key, [])) != epochs or not all(math.isfinite(v) for v in curves[key])
            for key in CURVES):
        raise RuntimeError(f"Existing completion is incomplete: {path}")
    metrics = result["metrics"]
    best_epoch = int(np.argmin(curves["validation_loss"]))
    if (metrics["best_validation_epoch_zero_based"] != best_epoch
            or metrics["minimum_validation_loss"] != curves["validation_loss"][best_epoch]
            or metrics["final_epoch_zero_based"] != epochs - 1
            or not all(math.isfinite(metrics[key]) for key in (
                "validation_selected_test_loss", "validation_selected_test_accuracy_percent",
                "final_epoch_test_loss", "final_epoch_test_accuracy_percent"))):
        raise RuntimeError(f"Existing endpoints are inconsistent: {path}")
    for endpoint, epoch in (("validation_selected", best_epoch), ("final_epoch", epochs - 1)):
        for metric in ("loss", "accuracy_percent"):
            if not math.isclose(metrics[f"{endpoint}_test_{metric}"], curves[f"test_{metric}"][epoch],
                                rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError(f"Existing test endpoint differs from its curve: {path}")
    for name in ("best", "final"):
        checkpoint = result[f"{name}_checkpoint"]
        if (checkpoint["file"] != f"{name}.pt"
                or base.sha256_file(directory / f"{name}.pt") != checkpoint["sha256"]):
            raise RuntimeError(f"Missing or changed {name} checkpoint: {directory}")
    return result


def save_checkpoint(path, model, spec, source, data, epoch, endpoint):
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {"model_state_dict": state, "spec": spec, "source_sha256": source["sha256"],
               "data_sha256": data["sha256"], "epoch_zero_based": epoch, "endpoint": endpoint}
    base.atomic_write(path, lambda stream: torch.save(payload, stream))


def run_trial(study, arm, seed, data_root, run_root, device, *, smoke=False,
              expected_cache_sha256=None, spec=None):
    if spec is None or study != "relu":
        raise ValueError("An explicit ReLU validation-study specification is required")
    if tuple(spec[key] for key in ("study", "arm", "seed", "smoke")) != (study, arm, seed, smoke):
        raise ValueError("Specification does not match the requested run")
    device = base.resolve_device(device)
    directory = Path(run_root) / ("original-smoke" if smoke else "original") / study / arm / f"seed-{seed}"
    source = base.source_provenance()
    with base.fit_lock(directory):
        train_data, test_data, data = base.load_data(spec, data_root, device, expected_cache_sha256)
        validation_data, validation_provenance = load_validation(spec, data_root, device, data)
        data = {**data, "original_train_test_sha256": data["sha256"], **validation_provenance}
        del data["sha256"]
        data["sha256"] = hashlib.sha256(base.canonical(data).encode()).hexdigest()
        existing = verified_result(directory, spec, source, data)
        if existing is not None:
            print(base.canonical({"status": "skipped_verified_complete", "result": str(directory / "result.json")}), flush=True)
            return existing
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.reset_peak_memory_stats(device)
        amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        relu.USE_AMP, relu.AMP_DTYPE = device.type == "cuda", amp_dtype
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "amp_dtype": str(amp_dtype) if device.type == "cuda" else None}
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = base.build_model(spec)
        runtime["initial_state_sha256"] = base.initial_state_hash(model)
        model = model.to(device)
        relu.SHUFFLE_GENERATOR = torch.Generator(device=device).manual_seed(seed)
        recipe = spec["recipe"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"], eta_min=1e-7)
        criterion = torch.nn.CrossEntropyLoss()
        curves, best_loss, best_epoch = {key: [] for key in CURVES}, math.inf, None
        start = time.monotonic()
        for epoch in range(recipe["epochs"]):
            train_loss, train_accuracy = relu.train_epoch(model, train_data, optimizer, criterion, recipe["batch_size"])
            scheduler.step()
            validation_loss, validation_accuracy = evaluate(model, validation_data, criterion, recipe["batch_size"], device)
            test_loss, test_accuracy = evaluate(model, test_data, criterion, recipe["batch_size"], device)
            values = (train_loss, validation_loss, train_accuracy, validation_accuracy,
                      optimizer.param_groups[0]["lr"], test_loss, test_accuracy)
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"Nonfinite metric at epoch {epoch}")
            for key, value in zip(CURVES, values):
                curves[key].append(value)
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                save_checkpoint(directory / "best.pt", model, spec, source, data, epoch, "minimum_validation_loss")
            base.atomic_json(directory / "progress.json", {
                "status": "running", "spec": spec, "source": source, "data": data, "runtime": runtime,
                "last_epoch_zero_based": epoch, "curves": curves,
                "minimum_validation_loss": best_loss, "best_validation_epoch_zero_based": best_epoch,
                "elapsed_seconds": time.monotonic() - start,
                "interruption_policy": "Restart unfinished fits from epoch zero with the same seed"})
            print(base.canonical({"study": study, "arm": arm, "seed": seed, "epoch": epoch + 1,
                                  "epochs": recipe["epochs"], "train_loss": train_loss,
                                  "validation_loss": validation_loss}), flush=True)
        final_epoch = recipe["epochs"] - 1
        save_checkpoint(directory / "final.pt", model, spec, source, data, final_epoch, "final_epoch")
        final_loss, final_accuracy = curves["test_loss"][-1], curves["test_accuracy_percent"][-1]
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        selected_loss, selected_accuracy = evaluate(model, test_data, criterion, recipe["batch_size"], device)
        for actual, expected in ((selected_loss, curves["test_loss"][best_epoch]),
                                 (selected_accuracy, curves["test_accuracy_percent"][best_epoch])):
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError("Restored validation checkpoint differs from its recorded test endpoint")
        result = {
            "status": "complete", "schema_version": base.SCHEMA_VERSION, "spec": spec,
            "source": source, "data": data, "runtime": runtime, "curves": curves,
            "elapsed_seconds": time.monotonic() - start,
            "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "metrics": {"minimum_validation_loss": best_loss, "best_validation_epoch_zero_based": best_epoch,
                        "validation_selected_test_loss": selected_loss,
                        "validation_selected_test_accuracy_percent": selected_accuracy,
                        "final_epoch_test_loss": final_loss, "final_epoch_test_accuracy_percent": final_accuracy,
                        "final_epoch_zero_based": final_epoch},
            "best_checkpoint": {"file": "best.pt", "sha256": base.sha256_file(directory / "best.pt")},
            "final_checkpoint": {"file": "final.pt", "sha256": base.sha256_file(directory / "final.pt")},
            "interpretation": "First minimum-validation-loss checkpoint; per-epoch test curves are diagnostics, never selection criteria. Training curves use dropout-on minibatches as weights change; validation and test use dropout-off evaluation.",
        }
        base.atomic_json(directory / "result.json", result)
        base.atomic_json(directory / "progress.json", {"status": "complete", "result": "result.json"})
        print(base.canonical({"status": "complete", "result": str(directory / "result.json")}), flush=True)
        return result
