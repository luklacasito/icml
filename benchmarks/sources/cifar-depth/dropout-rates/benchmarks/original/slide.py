"""Slide a six-layer dropout block through a critically initialized 12-layer MLP."""

import argparse
from pathlib import Path

from benchmarks import run_original as original
from benchmarks.original.validation import run_trial


def block_spec(start, seed, smoke=False, control=None):
    if (start is None) == (control is None):
        raise ValueError("Specify a block start or one control")
    if control not in (None, "uniform", "none"):
        raise ValueError("Unknown control")
    spec = original.specification("relu", "frontloaded", seed, smoke=smoke)
    recipe = spec["recipe"]
    recipe.update(depth=12, hidden_affine_layers=12, width=256, sigma_w_sq=2.0, sigma_b_sq=0.0)
    recipe.update(epochs=2 if smoke else 75, batch_size=3 if smoke else 100)
    recipe["scheduler"] = "CosineAnnealingLR over epochs, eta_min=1e-7; step before validation"
    recipe["test_endpoint"] = "After training only: first minimum-validation-CE checkpoint and final epoch"
    recipe["uniform_dropout"] = [0.1] * 12
    recipe["frontloaded_dropout"] = [0.2] * 6 + [0.0] * 6
    if control:
        rates = [0.1 if control == "uniform" else 0.0] * 12
    else:
        if not 0 <= start <= 6:
            raise ValueError("Block start must be between 0 and 6")
        rates = [0.2 if start <= i < start + 6 else 0.0 for i in range(12)]
    recipe.update(dropout_probabilities=rates, block_start=start,
                  block_width=6 if control is None else None,
                  mean_dropout=0.0 if control == "none" else 0.1, maximum_dropout=max(rates))
    spec["arm"] = spec["schedule"] = control or f"block-{start}"
    recipe.update(validation_size=10 if smoke else 1000, validation_split_seed=20260927)
    spec["protocol"] = "cifar_sliding_block_12l_validation_v1"
    spec["selection"] = "First minimum validation CE checkpoint; fixed 75-epoch training; no test-based selection"
    spec["initialization"] = "Clean ReLU critical: Gaussian variance 2/fan_in, zero biases; no dropout-dependent rescaling"
    spec["randomness"] = "Same initial weights by seed; separate seeded minibatch generator across all arms"
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument("--start", type=int)
    arm.add_argument("--control", choices=("uniform", "none"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args())
    if args["smoke"]:
        original.torch.set_num_threads(2)
    spec = block_spec(args.pop("start"), args["seed"], args["smoke"], args.pop("control"))
    run_trial(spec["study"], spec["arm"], spec=spec, **args)


if __name__ == "__main__":
    main()
