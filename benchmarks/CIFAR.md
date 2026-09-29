# Sliding dropout and measured correlations

The [source archive](sources/cifar-depth.tar.gz) contains the exact training code
for these two paper experiments. Every recorded training-source hash matches
the saved results; the activation-probe code also matches its recorded hash.
`manifest.json` inside the archive lists every included file and checksum.

The two studies used different versions of the CIFAR runner. They are kept in
separate directories so reproducing one does not silently substitute the other's
initialization, split or checkpoint rules. Cluster submission scripts and local
paths are unnecessary for running them and are not included.

Use the repository's Python 3.11 environment. From the repository root:

```bash
mkdir -p runs/cifar-source
tar -xzf benchmarks/sources/cifar-depth.tar.gz -C runs/cifar-source
```

## Moving the block

Six hidden layers, width 256, 75 epochs, five seeds. Three consecutive layers
receive dropout 0.2; the others receive none. Average dropout stays at 0.1.
[Saved measurements](../results/sliding_block.json) give the full settings.

```bash
cd runs/cifar-source/sliding-block
python -m benchmarks.original.slide --model mlp --start 0 --seed 47 \
  --data-root /path/to/cifar --run-root /path/to/new-runs --device cuda
```

Repeat for starts `0 1 2 3` and seeds `47 48 49 50 51`. For a quick execution
check, add `--smoke --device cpu` and use a separate output directory. Smoke
runs use synthetic data and do not reproduce the paper's measurements.

## Dropout rates and activation correlations

Twelve hidden layers, width 256, 35 epochs, five seeds, ten uniform rates from
0.001 to 0.25. `sweep.json` contains the exact grid and settings. This study saves
both the model chosen by validation loss and the final model.

From the extracted `dropout-rates/` directory:

```bash
python -m benchmarks.original.rate_sweep --all \
  --data-root /path/to/cifar --run-root /path/to/new-runs --device cuda
python probe_rates.py --data-root /path/to/cifar --run-root /path/to/new-runs
```

The probe measures 1,024 distinct-image pairs with 16 mask draws at initialization,
the minimum-validation-loss epoch and epoch 35. It verifies the saved weights,
source and data hashes before measuring correlations. `reference.json` supplies
the expected initialization and split hashes. The resulting aggregate curves
are in [cifar_correlations.json](../results/cifar_correlations.json).

To smoke-test one fit, use `--rate 0.1 --seed 47 --smoke --device cpu` instead of
`--all`. `test_rate_sweep.py` checks initialization, checkpoint selection and
whether diagnostic evaluations leave training unchanged.

Raw CIFAR images, trained weights and per-pair moments are not included. The
paper plots can be rebuilt from the saved aggregates without retraining; see
the [figure guide](../manuscript/FIGURES.md).
