"""Train the CIFAR-10 dropout-rate sweep with validation checkpoint selection."""

import argparse
import json
import math
from pathlib import Path

from benchmarks import run_original as base
from benchmarks.original.slide import block_spec
from benchmarks.original.validation import run_trial

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "sweep.json"


def rate_spec(p, seed, smoke=False, profile="uniform", config=None):
    config = json.loads(DEFAULT_CONFIG.read_text()) if config is None else dict(config)
    if profile not in ("uniform", "early") or not math.isfinite(p) or not 0 <= p < 1:
        raise ValueError("Expected uniform or early dropout with 0 <= p < 1")
    if profile == "early" and (2 * p >= 1 or config["depth"] % 2):
        raise ValueError("Early step needs even depth and a mean rate below 0.5")
    spec = block_spec(None, seed, smoke=smoke, control="uniform")
    recipe = spec["recipe"]
    for key in ("depth", "width", "learning_rate", "weight_decay"):
        recipe[key] = config[key]
    if not smoke:
        for key in ("epochs", "batch_size", "train_size", "validation_size", "test_size"):
            recipe[key] = config[key]
    else:
        recipe.update(epochs=2, batch_size=3, train_size=6, validation_size=10, test_size=4)
    depth = recipe["depth"]
    rates = [p] * depth if profile == "uniform" else [2 * p] * (depth // 2) + [0.0] * (depth // 2)
    for key in ("uniform_dropout", "frontloaded_dropout", "block_start", "block_width"):
        recipe.pop(key, None)
    recipe.update(hidden_affine_layers=depth, dropout_probabilities=rates,
                  mean_dropout=p, maximum_dropout=max(rates), dropout_profile=profile)
    recipe["scheduler"] = f"CosineAnnealingLR(T_max={recipe['epochs']}, eta_min=1e-7); step after training"
    recipe["test_endpoint"] = "Per-epoch curves are diagnostics; report validation-selected and final-epoch test metrics"
    spec.update(arm=f"{profile}-p-{p:g}", schedule=profile,
                protocol="cifar_dropout_rate_sweep_validation_v1",
                selection=f"First minimum validation CE checkpoint; fixed {recipe['epochs']}-epoch training; no test-based selection")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--rate", type=float)
    choice.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", choices=("uniform", "early"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.all:
        if args.seed is not None or args.profile is not None:
            parser.error("--all uses seeds and profiles from the configuration")
        trials = ((p, seed, profile) for seed in config["seeds"]
                  for profile in config["profiles"] for p in config["rates"])
    else:
        if args.seed is None:
            parser.error("--rate requires --seed")
        if args.rate not in config["rates"] and not (args.smoke and args.rate == 0):
            parser.error("--rate must appear in the configuration (zero is also allowed for smoke tests)")
        trials = [(args.rate, args.seed, args.profile or "uniform")]
    if args.smoke:
        base.torch.set_num_threads(2)
    for p, seed, profile in trials:
        spec = rate_spec(p, seed, smoke=args.smoke, profile=profile, config=config)
        run_trial("relu", spec["arm"], seed, args.data_root, args.run_root, args.device,
                  smoke=args.smoke, spec=spec)


if __name__ == "__main__":
    main()
