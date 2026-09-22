# Standalone FI-2010 vanilla MLP

This runner reproduces the recipe for the four standalone vanilla-MLP rows in
the appendix. It uses the recovered original model and training definitions,
including their random-number ordering. It is independent of the depth-12
multi-dataset FI-2010 benchmark and its prepared cache.

**Historical limit:** the retained standalone results contain aggregate curves,
test metrics and best epochs, but no original confirmation plans or selected
learning rates. This code reruns the documented validation screen to select
rates; it does not claim to replay those missing plans. New source and dataset
fingerprints identify the new trials. The saved paper table still describes
the historical runs.

## Recipe

| Component | Setting |
|---|---|
| Input / model | Flatten 128 × 40 into a plain ReLU MLP; 6 or 12 hidden layers, width 256 or 512; separate three-class readout |
| Initialization | Every affine, including readout: Gaussian weights with variance `1.98 / fan_in`, biases with variance `0.02` |
| Dropout | After every hidden ReLU; none, uniform, early/late half-depth step, early one-third big step, decreasing/increasing linear; each nonzero profile has mean probability 0.10 |
| Split | All five stock ordinals; day 6 training, day 7 validation, days 8–10 test; windows never cross stock/day boundaries |
| Labels / purge | First supplied label column (release row 144); endpoint of each window; original horizon name 10 order events = one representation; conservatively remove the final 10 representations per segment |
| Normalization | Training-day rows, excluding the purged tail; NumPy float64 mean/std, cast to float32 for windows; standard deviations below `1e-8` replaced by 1 |
| Optimization | Adam, default betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay; batch 128; gradient clipping at norm 1; float32, deterministic algorithms, TF32 disabled |
| Learning rate | Cosine multiplier `0.01 + 0.99 * (1 + cos(pi * epoch / (epochs - 1))) / 2`, applied before training each epoch, with zero-based `epoch` |
| Rate selection | Uniform dropout only; `{3e-5, 1e-4, 3e-4, 1e-3}`, seeds 0–2, 50 epochs; minimize mean best-validation log loss per architecture, ties choose lower rate |
| Confirmation | Frozen rate shared by all seven profiles; seeds 100–104; 100 epochs |
| Evaluation | Test once at each run's minimum-validation-loss checkpoint; log loss uses float64 softmax; accuracies in output JSON are fractions |

Initialization and minibatch streams are shared across profiles within each
architecture/seed pair. Dropout has one independent generator per hidden
layer. Early/late versions of a profile share a named dropout stream. The
`latest.pt` checkpoint preserves the optimizer, all RNG streams and the current
best-validation model for exact epoch-boundary resume on the same runtime.

## Data preparation

Obtain the FI-2010 **NoAuction decimal-precision** release from the
[dataset source](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649).
Arrange the original one-text-file ZIP archives as follows. If your download
contains unpacked text files, zip each listed text file at the archive root.

| Local archive | Text member |
|---|---|
| `day01.zip` | `Train_Dst_NoAuction_DecPre_CF_1.txt` |
| `day02.zip` … `day10.zip` | `Test_Dst_NoAuction_DecPre_CF_1.txt` … `Test_Dst_NoAuction_DecPre_CF_9.txt` |
| `cumulative06.zip` | `Train_Dst_NoAuction_DecPre_CF_6.txt` |

The preparer detects five stock blocks from isolated price discontinuities,
checks days 1–6 against the cumulative stock-major file, checks subsequent
stock-ordinal price continuity, and writes checksum-verified segment arrays.
It retains the original source attribution in `manifest.json`. No data or
checkpoints are included in this repository.

```bash
python -m benchmarks.run_vanilla prepare \
  --raw /path/to/decimal-precision-zips --output /path/to/fi2010-segments
python -m benchmarks.run_vanilla plan \
  --data-root /path/to/fi2010-segments --output runs/vanilla/plans
```

Run from the repository root with the dependencies in `requirements.txt`.
The standalone path needs Python, NumPy and PyTorch; it does not contact W&B or
any remote service. On CUDA, use the deterministic cuBLAS configuration before
starting Python: `export CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Run and select

A plan only writes configurations. Each `run` command executes one trial;
replace `--index` with each required index. The four optional canaries are in
`canary.json`. The complete learning-rate screen has indices 0–47.

```bash
python -m benchmarks.run_vanilla run \
  --plan runs/vanilla/plans/lr_search.json --index 0 \
  --data-root /path/to/fi2010-segments --output runs/vanilla/lr_search \
  --device cuda
# After all 48 screening trials finish:
python -m benchmarks.run_vanilla select \
  --plan runs/vanilla/plans/lr_search.json --results runs/vanilla/lr_search \
  --output runs/vanilla/plans/confirmation.json
# Run confirmation indices 0–139:
python -m benchmarks.run_vanilla run \
  --plan runs/vanilla/plans/confirmation.json --index 0 \
  --data-root /path/to/fi2010-segments --output runs/vanilla/confirmation \
  --device cuda
```

Output is organized as `<index>_<trial-fingerprint>/`. Each trial records its
specification, runtime, history, best checkpoint and final metrics. Confirmation
also saves test probabilities with stock/day/window coordinates. Resuming uses
the identical command and output directory. Source or data changes reject the
old plan. Test is excluded from screening; final-epoch and per-epoch test losses
are not measured by this historical protocol.

## CPU verification

```bash
python -m benchmarks.run_vanilla smoke --output /tmp/vanilla-smoke
python -m unittest discover -s tests -p test_benchmark_vanilla.py -v
```

Smoke uses two epochs, a small model and synthetic data, is marked
`synthetic_smoke_only`, and never evaluates test. It checks execution only.
Scientific results require the complete data and protocol above. Exact numerical
agreement across different PyTorch versions, CPUs or GPUs is not guaranteed.
