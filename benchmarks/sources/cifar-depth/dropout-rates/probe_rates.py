"""Probe fixed distinct-image pairs at initialization and saved checkpoints."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmarks import run_original as base
from benchmarks.original.rate_sweep import rate_spec

def moments(a, b):
    a, b = a.double(), b.double()
    return torch.stack(((a*b).mean(1), a.square().mean(1), b.square().mean(1)), -1)


def correlation(values):
    if not np.isfinite(values).all() or (values[..., 1:] <= 0).any():
        raise ValueError("Nonfinite or zero-norm pair; do not silently drop images")
    result = values[..., 0] / np.sqrt(values[..., 1]*values[..., 2])
    if np.abs(result).max() > 1 + 1e-6:
        raise ValueError("Correlation outside [-1,1]")
    return np.clip(result, -1, 1)


@torch.no_grad()
def measure(model, x, permutation, probabilities, seed, repeats):
    count = len(x)
    accumulated, cosines = [], []
    for repeat in range(repeats):
        generator = torch.Generator().manual_seed(1000000 + 1000*seed + repeat)
        hidden = torch.cat((x, x[permutation])).flatten(1)
        values = [moments(hidden[:count], hidden[count:])]
        for layer, probability in zip(model.layers[:-1], probabilities):
            hidden = torch.relu(layer(hidden))
            mask = torch.rand(hidden.shape, generator=generator) >= probability
            hidden = hidden * mask / (1-probability)
            values.append(moments(hidden[:count], hidden[count:]))
        values = torch.stack(values).numpy()  # layer, pair, moment
        accumulated.append(values)
        cosines.append(correlation(values).mean(1))
    averaged = np.mean(accumulated, axis=0)
    pairs = correlation(averaged)
    return averaged, {
        "mean_pair_correlation": pairs.mean(1).tolist(),
        "pair_std": pairs.std(1, ddof=1).tolist(),
        "pooled_correlation": correlation(averaged.mean(1)).tolist(),
        "mean_cosine": np.mean(cosines, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("sweep.json"))
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sweep = json.loads(args.config.read_text())
    reference = json.loads(args.reference.read_text())
    seeds = [sweep["seeds"][0]] if args.smoke else sweep["seeds"]
    stages = ["initial", "best", "final"]
    repeats = 2 if args.smoke else sweep["probe_repeats"]
    torch.set_num_threads(4)
    spec = rate_spec(sweep["rates"][0], seeds[0], smoke=args.smoke, profile=sweep["profiles"][0], config=sweep)
    (images, _), _, data = base.load_data(spec, args.data_root, torch.device("cpu"))
    if not args.smoke:
        assert data["sha256"] == reference["original_train_test_sha256"]
    count = min(sweep["probe_pairs"], len(images))
    indices = np.random.RandomState(20260927).choice(len(images), count, replace=False)
    x = images[torch.tensor(indices)]
    permutation = torch.roll(torch.arange(count), 1)
    assert torch.all(permutation != torch.arange(count))
    initial_moments = moments(x.flatten(1), x[permutation].flatten(1)).numpy()
    source = base.source_provenance()
    config = {**sweep, "seeds": seeds, "pairs": count, "repeats": repeats,
              "stages": stages, "smoke": args.smoke,
              "input_correlation_by_pair": correlation(initial_moments).tolist(),
              "training_subset_positions": indices.tolist(), "data": data, "source": source,
              "probe_sha256": base.sha256_file(Path(__file__)),
              "device": "cpu", "precision": "float32 forward, float64 moments",
              "definition": "Mean over pairs of mask-averaged normalized activation moments; post-dropout; uncentered",
              "masks": "Independent between inputs; common random draws across rates, profiles and checkpoints"}
    args.run_root.mkdir(parents=True, exist_ok=True)
    output = args.run_root / ("smoke-activation-probes.json" if args.smoke else "activation-probes.json")
    moment_dir = args.run_root / ("smoke-pair-moments" if args.smoke else "pair-moments")
    moment_dir.mkdir(exist_ok=True)
    expected = len(seeds)*len(stages)*len(sweep["rates"])*len(sweep["profiles"])
    results, started = [], time.monotonic()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = base.build_model(rate_spec(sweep["rates"][0], seed, smoke=args.smoke, profile=sweep["profiles"][0], config=sweep))
        initial_hash = base.initial_state_hash(model)
        if not args.smoke:
            assert initial_hash == reference["initial_state_sha256"][str(seed)]
        initial_state = {name: value.clone() for name, value in model.state_dict().items()}
        for profile in sweep["profiles"]:
            for rate in sweep["rates"]:
                spec = rate_spec(rate, seed, smoke=args.smoke, profile=profile, config=sweep)
                arm = spec["arm"]
                directory = args.run_root / ("original-smoke" if args.smoke else "original") / "relu" / arm / f"seed-{seed}"
                record = json.loads((directory / "result.json").read_text())
                assert record["status"] == "complete" and record["spec"] == spec
                assert record["source"]["sha256"] == source["sha256"]
                assert record["data"]["original_train_test_sha256"] == data["sha256"]
                assert record["runtime"]["initial_state_sha256"] == initial_hash
                rates = spec["recipe"]["dropout_probabilities"]
                for stage in stages:
                    checkpoint, epoch = None, 0
                    if stage == "initial":
                        model.load_state_dict(initial_state, strict=True)
                    else:
                        path = directory / f"{stage}.pt"
                        digest = base.sha256_file(path)
                        assert digest == record[f"{stage}_checkpoint"]["sha256"]
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        assert saved["spec"] == spec and saved["source_sha256"] == source["sha256"]
                        assert saved["data_sha256"] == record["data"]["sha256"]
                        epoch_key = "best_validation_epoch_zero_based" if stage == "best" else "final_epoch_zero_based"
                        assert saved["epoch_zero_based"] == record["metrics"][epoch_key]
                        assert saved["endpoint"] == ("minimum_validation_loss" if stage == "best" else "final_epoch")
                        model.load_state_dict(saved["model_state_dict"], strict=True)
                        epoch = saved["epoch_zero_based"] + 1
                        checkpoint = {"file": str(path), "sha256": digest,
                                      "source_sha256": source["sha256"], "data_sha256": saved["data_sha256"]}
                    before = base.initial_state_hash(model)
                    values, curves = measure(model, x, permutation, rates, seed, repeats)
                    assert base.initial_state_hash(model) == before
                    np.testing.assert_allclose(correlation(values[0]), config["input_correlation_by_pair"], atol=1e-14)
                    path = moment_dir / f"{stage}-{arm}-{seed}.npz"
                    np.savez_compressed(path, moments=values)
                    results.append({"seed": seed, "stage": stage, "profile": profile, "rate": rate,
                                    "condition": arm, "dropout_probabilities": rates,
                                    "epoch_one_based": epoch, "initial_state_sha256": initial_hash,
                                    "measured_state_sha256": before, "checkpoint": checkpoint,
                                    "pair_moments_sha256": base.sha256_file(path), **curves})
                    base.atomic_json(output, {"config": config, "results": results,
                        "complete": len(results) == expected, "elapsed_seconds": time.monotonic()-started})
                    print(json.dumps({"stage": stage, "arm": arm, "seed": seed,
                                      "completed": len(results), "total": expected}), flush=True)


if __name__ == "__main__":
    main()
