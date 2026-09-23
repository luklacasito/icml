"""Run frozen retained benchmark arms and save both test endpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import BENCHMARK_SPECS, cache_path, load_benchmark_bundle
from .models import build_model
from .training import (
    DatasetBundle,
    TrainingConfig,
    evaluate,
    seed_everything,
    train_model,
)

PROTOCOLS = Path(__file__).with_name("protocols.json")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(spec: dict) -> str:
    return hashlib.sha256(
        canonical_json({k: v for k, v in spec.items() if k != "seed"}).encode()
    ).hexdigest()[:20]


def seed_streams(spec: dict) -> dict:
    """Historical named streams: JSON numeric types are part of the protocol."""

    def derived(stream, payload):
        digest = hashlib.sha256(f"{stream}\0{canonical_json(payload)}".encode()).digest()
        return int.from_bytes(digest[:4], "big")

    pair_payload = dict(spec)
    for suffix in ("_early", "_late"):
        if spec["profile_id"].endswith(suffix):
            pair_payload["profile_id"] = spec["profile_id"][: -len(suffix)]
            break
    pair_key = hashlib.sha256(canonical_json(pair_payload).encode()).hexdigest()[:20]
    base = {"base_seed": spec["seed"]}
    return {
        "scheme": "sha256_named_streams_v1",
        "base_seed": spec["seed"],
        "initialization_seed": derived("initialization", base),
        "minibatch_seed": derived("minibatch", base),
        "dropout_seed": derived("dropout_common_random_numbers", {"pair_key": pair_key}),
        "dropout_crn_group": pair_key,
    }


def read_protocols(path: Path = PROTOCOLS) -> dict:
    payload = json.loads(path.read_text())
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported protocol schema")
    return payload["protocols"]


def validate_arm(cohort: dict, arm: dict) -> None:
    spec = arm["spec"]
    if config_hash(spec) != arm["historical_config_hash"]:
        raise ValueError("Frozen configuration hash differs from the historical record")
    if spec["dataset"] not in BENCHMARK_SPECS or spec["dataset"] != cohort["dataset"]:
        raise ValueError("Dataset differs from the retained cohort")
    if spec["model_kind"] != cohort["model_kind"] or spec["parameterization"] != "sp":
        raise ValueError("Unsupported or mismatched model configuration")
    if spec["stage"] != "confirm" or spec["evaluate_test"] is not True:
        raise ValueError("This runner accepts frozen confirmation arms only")
    p = np.asarray(arm["p_layers"], dtype=float)
    if p.shape != (spec["depth"],) or not np.isfinite(p).all() or np.any((p < 0) | (p >= 1)):
        raise ValueError("Invalid dropout vector")
    if not np.isclose(p.mean(), spec["mean_dropout"], rtol=0, atol=1e-12):
        raise ValueError("Dropout vector does not spend the specified budget")
    if p.max() > spec["max_dropout"] + 1e-12:
        raise ValueError("Dropout vector exceeds its specified cap")
    for key in ("train_size", "validation_size", "test_size", "split_seed"):
        if spec[key] != cohort["split"][key]:
            raise ValueError(f"Inconsistent cohort split field: {key}")


def verify_cache_payload(path: Path) -> str:
    """Recompute the historical array digest with bounded decompression memory.

    Split hashes include this digest. Checking actual bytes prevents an edited
    cache from passing merely by retaining its old payload_sha256 metadata.
    """
    from numpy.lib import format as npy

    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for key in ("features_mlp", "features_sequence", "labels"):
            with archive.open(f"{key}.npy") as handle:
                version = npy.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = npy.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = npy.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"Unsupported NPY version: {version}")
                if fortran or dtype.hasobject:
                    raise ValueError("Cache arrays must be row-major numeric arrays")
                expected = int(np.prod(shape)) * dtype.itemsize
                size = 0
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                if size != expected:
                    raise ValueError(f"Wrong byte count for {key}")
                digest.update(str(dtype).encode())
                digest.update(str(shape).encode())
    actual = digest.hexdigest()
    with np.load(path, allow_pickle=False) as payload:
        if actual != str(payload["payload_sha256"]):
            raise ValueError("Cache array bytes do not match payload_sha256")
    return actual


def source_provenance() -> dict:
    directory = Path(__file__).parent
    return {
        "execution_files": {p.name: sha256_file(p) for p in sorted(directory.glob("*.py"))},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
    }


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fit(
    spec: dict,
    p_layers: list[float],
    bundle: DatasetBundle,
    directory: Path,
    *,
    device: str,
    provenance: dict,
) -> dict:
    """Train one immutable output directory; never overwrite a prior run."""
    directory.mkdir(parents=True, exist_ok=False)
    randomization = seed_streams(spec)
    manifest = {
        "spec": spec,
        "p_layers": p_layers,
        "randomization": randomization,
        "split_hash": bundle.split_hash,
        "split_protocol": bundle.split_protocol,
        "test_subset_hash": bundle.test_subset_hash,
        "provenance": provenance,
    }
    write_json(directory / "manifest.json", manifest)
    seed_everything(randomization["initialization_seed"])
    model = build_model(spec, p_layers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    config = TrainingConfig(
        **{
            key: spec[key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "lr_floor_ratio",
                "weight_decay",
                "gradient_clip_norm",
                "evaluate_test",
            )
        },
        seed=randomization["minibatch_seed"],
        stochastic_seed=randomization["dropout_seed"],
        restore_best_validation=True,
        device=device,
    )
    start = time.perf_counter()
    trained = train_model(
        model,
        optimizer,
        bundle,
        config,
        return_best_state=True,
        return_final_state=True,
    )
    history = {key: value.tolist() for key, value in trained.pop("history").items()}
    if any(
        not np.isfinite(history[key]).all()
        for key in (
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        )
    ):
        raise RuntimeError("Nonfinite training curve; run has no valid complete result")
    selected = int(np.argmin(history["validation_loss"]))
    if trained["test_epoch"] != selected:
        raise RuntimeError("Test checkpoint differs from the first validation minimum")
    test = {
        "best_validation_checkpoint": {
            "epoch": selected,
            "loss": trained["final_test_loss"],
            "accuracy": trained["final_test_accuracy"],
        },
        "fixed_final_epoch": {
            "epoch": spec["epochs"] - 1,
            "loss": trained["final_epoch_test_loss"],
            "accuracy": trained["final_epoch_test_accuracy"],
        },
    }
    for filename, state_key, endpoint in (
        ("best.pt", "best_state_dict", "best_validation_checkpoint"),
        ("final.pt", "final_state_dict", "fixed_final_epoch"),
    ):
        torch.save(
            {
                "model_state_dict": trained.pop(state_key),
                "spec": spec,
                "p_layers": p_layers,
                "epoch": test[endpoint]["epoch"],
                "split_hash": bundle.split_hash,
            },
            directory / filename,
        )
    result = {
        "status": "complete",
        "selected_epoch": selected,
        "test": test,
        "history": history,
        "training": trained,
        "duration_seconds": time.perf_counter() - start,
        "checkpoint_sha256": {
            name: sha256_file(directory / name) for name in ("best.pt", "final.pt")
        },
    }
    write_json(directory / "result.json", result)
    return result


def synthetic_fixture():
    """Opposed labels force best and final checkpoints to differ."""
    spec = {
        "stage": "confirm",
        "dataset": "fi2010",
        "model_kind": "mlp",
        "profile_id": "uniform",
        "mean_dropout": 0.1,
        "learning_rate": 0.01,
        "seed": 9182,
        "max_dropout": 0.20,
        "depth": 2,
        "width": 8,
        "heads": 8,
        "mlp_ratio": 4.0,
        "activation": "relu",
        "sigma_w_sq": 1.98,
        "sigma_b_sq": 0.0,
        "epochs": 4,
        "batch_size": 8,
        "weight_decay": 0.0,
        "lr_floor_ratio": 1e-3,
        "gradient_clip_norm": None,
        "train_size": 24,
        "validation_size": 12,
        "test_size": 12,
        "split_seed": 20260812,
        "evaluate_test": True,
        "budget_space": "dropout_probability",
        "parameterization": "sp",
        "cohort_id": "public-synthetic-endpoint-smoke-v1",
    }
    dimension = BENCHMARK_SPECS["fi2010"].mlp_input_dim
    train = TensorDataset(torch.zeros(24, dimension), torch.zeros(24, dtype=torch.long))
    validation = TensorDataset(torch.zeros(12, dimension), torch.ones(12, dtype=torch.long))
    test = TensorDataset(torch.zeros(12, dimension), torch.ones(12, dtype=torch.long))
    digest = hashlib.sha256(b"synthetic-zero-input-opposed-labels-v1").hexdigest()
    bundle = DatasetBundle(
        train,
        validation,
        test,
        digest,
        "fi2010",
        "synthetic_v1",
        digest,
        "synthetic_v1",
        0,
    )
    return spec, bundle


def smoke(directory: Path) -> dict:
    spec, bundle = synthetic_fixture()
    result = fit(
        spec,
        [0.1, 0.1],
        bundle,
        directory,
        device="cpu",
        provenance={**source_provenance(), "synthetic": True},
    )
    if result["selected_epoch"] != 0 or not np.all(
        np.diff(result["history"]["validation_loss"]) > 0
    ):
        raise AssertionError("Fixture must select epoch zero while validation loss rises")
    states = []
    for name, endpoint in (
        ("best.pt", "best_validation_checkpoint"),
        ("final.pt", "fixed_final_epoch"),
    ):
        saved = torch.load(directory / name, map_location="cpu", weights_only=True)
        model = build_model(saved["spec"], saved["p_layers"])
        model.load_state_dict(saved["model_state_dict"])
        loss, accuracy = evaluate(model, DataLoader(bundle.test, batch_size=8), torch.device("cpu"))
        expected = result["test"][endpoint]
        np.testing.assert_allclose(
            [loss, accuracy], [expected["loss"], expected["accuracy"]], rtol=1e-6
        )
        states.append(saved["model_state_dict"])
    if all(torch.equal(states[0][key], states[1][key]) for key in states[0]):
        raise AssertionError("Smoke checkpoints must differ")
    report = {
        "synthetic": True,
        "status": "complete",
        "selected_epoch": 0,
        "final_epoch": spec["epochs"] - 1,
        "saved_endpoints_reproduced": True,
    }
    write_json(directory / "smoke.json", report)
    return report


def run(args):
    protocols = read_protocols(args.protocols)
    if args.cohort not in protocols:
        raise ValueError(f"Unknown cohort {args.cohort!r}; use the list command")
    cohort = protocols[args.cohort]
    if cohort.get("runner", "benchmark") != "benchmark":
        raise ValueError("This cohort requires its separately documented runner")
    arms = []
    for name in args.profile:
        if name == "frontloaded":
            comparison = cohort.get("primary_comparison", {})
            name = comparison.get("profile")
            if name not in {"step_early", "big_step", "linear_early"}:
                raise ValueError(
                    "This cohort has no selected frontloaded arm; use an explicit profile ID"
                )
        if name not in cohort["arms"]:
            raise ValueError(f"Unavailable profile {name!r}; available: {list(cohort['arms'])}")
        arm = cohort["arms"][name]
        validate_arm(cohort, arm)
        arms.append((name, arm))
    if len({name for name, _ in arms}) != len(arms):
        raise ValueError("Select each profile only once")
    if args.seed is not None and (
        any(seed < 0 for seed in args.seed) or len(set(args.seed)) != len(args.seed)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if args.describe:
        print(
            json.dumps(
                {"cohort": args.cohort, "split": cohort["split"], "arms": dict(arms)},
                indent=2,
            )
        )
        return
    if args.data_root is None or args.output_dir is None:
        raise ValueError("Training requires --data-root and --output-dir")
    cache = cache_path(cohort["dataset"], args.data_root)
    print(
        json.dumps({"event": "verifying_data", "dataset": cohort["dataset"]}),
        flush=True,
    )
    payload_hash = verify_cache_payload(cache)
    cache_hash = sha256_file(cache)
    if args.cache_sha256 and cache_hash != args.cache_sha256:
        raise ValueError("Cache file differs from --cache-sha256")
    spec = arms[0][1]["spec"]
    split = cohort["split"]
    bundle = load_benchmark_bundle(
        spec["dataset"],
        root=args.data_root,
        view="mlp" if spec["model_kind"] == "mlp" else "sequence",
        **{key: spec[key] for key in ("train_size", "validation_size", "test_size", "split_seed")},
        tiny_standardization=split["standardization"],
    )
    if bundle.split_hash != split["split_hash"] or bundle.split_protocol != split["split_protocol"]:
        raise ValueError(
            "Prepared data do not reproduce this cohort's frozen split; refusing to train"
        )
    provenance = {
        **source_provenance(),
        "protocols_sha256": sha256_file(args.protocols),
        "cache_sha256": cache_hash,
        "payload_sha256": payload_hash,
        "standardization": split["standardization"],
        "cohort": args.cohort,
    }
    for name, arm in arms:
        for seed in args.seed if args.seed is not None else arm["seeds"]:
            spec = copy.deepcopy(arm["spec"])
            spec["seed"] = seed
            destination = args.output_dir / args.cohort / name / f"seed-{seed}"
            print(
                json.dumps({"event": "starting", "profile": name, "seed": seed}),
                flush=True,
            )
            result = fit(
                spec,
                arm["p_layers"],
                bundle,
                destination,
                device=args.device,
                provenance={**provenance, "historical_seed": seed in arm["seeds"]},
            )
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "profile": name,
                        "seed": seed,
                        "output": str(destination),
                        "test": result["test"],
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list frozen cohorts and available profiles")
    listing.add_argument("--protocols", type=Path, default=PROTOCOLS)
    training = commands.add_parser("train", help="run selected frozen confirmation arms")
    training.add_argument("--protocols", type=Path, default=PROTOCOLS)
    training.add_argument("--cohort", required=True)
    training.add_argument("--profile", nargs="+", default=["uniform"])
    training.add_argument(
        "--seed", nargs="+", type=int, help="default: all historical confirmation seeds"
    )
    training.add_argument("--data-root", type=Path)
    training.add_argument("--output-dir", type=Path)
    training.add_argument(
        "--cache-sha256", help="optional independently recorded cache file checksum"
    )
    training.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    training.add_argument(
        "--describe",
        action="store_true",
        help="show frozen selection without loading data",
    )
    checking = commands.add_parser("smoke", help="CPU checkpoint smoke test; no dataset required")
    checking.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        for name, cohort in read_protocols(args.protocols).items():
            if cohort.get("runner", "benchmark") == "benchmark":
                print(f"{name}: {', '.join(cohort['arms'])}")
    elif args.command == "smoke":
        print(json.dumps(smoke(args.output_dir), indent=2))
    else:
        try:
            run(args)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
