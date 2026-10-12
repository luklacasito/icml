"""Move a fixed half-depth dropout block; reuse the original CIFAR training loop."""

import argparse
from pathlib import Path

from benchmarks import run_original as original


def block_spec(model, start, seed, smoke=False):
    spec = original.specification(
        "relu" if model == "mlp" else "vit", "frontloaded", seed, smoke=smoke
    )
    recipe = spec["recipe"]
    if model == "mlp":
        recipe.update(sigma_w_sq=1.98, sigma_b_sq=0.02)
        recipe.update(epochs=2 if smoke else 75, batch_size=3 if smoke else 100)
        recipe["scheduler"] = "CosineAnnealingLR over epochs, eta_min=1e-7; step before test"
    depth = recipe["depth"]
    width = depth // 2
    if not 0 <= start <= depth - width:
        raise ValueError(f"Block start must be between 0 and {depth - width}")
    rate = 0.1 * depth / width
    recipe["frontloaded_dropout"] = [rate] * width + [0.0] * (depth - width)
    recipe["dropout_probabilities"] = [
        rate if start <= i < start + width else 0.0 for i in range(depth)
    ]
    recipe.update(block_start=start, block_width=width, mean_dropout=0.1, maximum_dropout=0.2)
    spec["arm"] = spec["schedule"] = f"block-{start}"
    spec["protocol"] = "cifar_sliding_block_v1"
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "vit"), default="mlp")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args())
    if args["smoke"]:
        original.torch.set_num_threads(2)
    spec = block_spec(args.pop("model"), args.pop("start"), args["seed"], args["smoke"])
    original.run_trial(spec["study"], spec["arm"], spec=spec, **args)


if __name__ == "__main__":
    main()
