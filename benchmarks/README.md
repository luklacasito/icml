# Retained cross-dataset benchmarks

This package runs the frozen confirmation configurations retained in the paper:
FI-2010, Speech Commands, Jannis, and Tiny ImageNet, with MLPs and transformers.
It includes the Tiny ImageNet 80,000-example and Jannis 100-epoch followups and
the two separately tuned linear-profile transformer followups.
[protocols.json](protocols.json) contains the exact per-arm specifications,
dropout vectors, historical seeds, configuration hashes, and data-split hashes.
The separate [standalone vanilla-MLP instructions](VANILLA.md) cover the four
financial architecture comparisons, which use a different data protocol.

## Install and check

From the repository root, use Python 3.11:

```sh
python -m venv .venv-benchmarks
source .venv-benchmarks/bin/activate
python -m pip install -r benchmarks/requirements.txt
python -m benchmarks.run list
python -m benchmarks.run smoke --output-dir /tmp/dropout-benchmark-smoke
```

The CPU smoke trains a small synthetic MLP for four epochs. Opposed training
and validation labels force the best validation checkpoint to occur before the
final epoch. It reloads both saved checkpoints, reproduces their reported test
metrics, and verifies that their weights differ. This checks execution and
checkpoint selection; it is not a paper result. Output directories must be new.

## Run frozen paper settings

Prepared caches belong at `DATA_ROOT/benchmarks/DATASET.npz`. Inspect a frozen
setting without loading any data:

```sh
python -m benchmarks.run train \
  --cohort openml_jannis-mlp-zero-decay \
  --profile uniform step_early --describe
```

Run the selected profiles and seeds on user-supplied data:

```sh
python -m benchmarks.run train \
  --cohort openml_jannis-mlp-zero-decay \
  --profile uniform step_early --seed 100 101 102 103 104 \
  --data-root /path/to/data --output-dir /path/to/new-runs --device cuda
```

`--profile` accepts the profile IDs printed by `list`: `uniform`, `step_early`,
`big_step`, `linear_early`, `linear_late`, and `none_tuned`, where available.
`frontloaded` resolves to the cohort's explicitly recorded main-table
frontloaded comparison; it fails if that comparison is absent. The broader
appendix comparison can select a different profile: for 100-epoch Jannis it
selects `linear_late`, while the main frontloaded comparison uses `step_early`.
Omitting `--seed` runs every historical seed for each selected arm. Other
nonnegative seeds are allowed and recorded as new seeds in the manifest.
`--device cpu` is supported; `auto` selects CUDA if available, otherwise CPU.

The [September 22 extension](../results/seed_extension.md) adds seeds 105–109
for the nine main-table benchmark comparisons that previously had five pairs.
Pass those seed IDs with `--profile uniform frontloaded` to repeat the frozen
extension settings. Extended-search Jannis already had ten pairs and received
no new runs. The [original CIFAR runner](original/README.md) covers the separate
ReLU and both-block ViT extensions, which use the notebook training protocol.

Every run writes, under `COHORT/PROFILE/seed-N/`:

- `manifest.json`: exact configuration, dropout vector, independent RNG
  streams, source hashes, package versions, cache identity, and split identity.
- `best.pt`: weights at the **first minimum validation cross-entropy**.
- `final.pt`: weights after the configured final training epoch.
- `result.json`: training/validation curves, selected epoch, both test
  endpoints, and checkpoint checksums.

The runner refuses to overwrite a run directory. A directory without a complete
`result.json` is an incomplete run; retry in a new output root. Checkpoint files
contain weights and configuration, not optimizer state for resuming training.
Test data are evaluated only after training and checkpoint selection. There is
no test-loss history, and no claim about minimum test loss over training.

## Prepare data

Install the optional preparation dependencies, then prepare one dataset:

```sh
python -m pip install -r benchmarks/requirements-prepare.txt
python -m benchmarks.prepare openml_jannis --root /path/to/data --raw /path/to/raw
python -m benchmarks.prepare speech_commands --root /path/to/data --raw /path/to/raw
python -m benchmarks.prepare tiny_imagenet --root /path/to/data --raw /path/to/raw
python -m benchmarks.prepare fi2010 --root /path/to/data --raw /path/to/raw
```

Jannis is fetched as OpenML `jannis`, version 1. Speech Commands uses
TorchAudio's V2 training subset and can download the archive. Tiny ImageNet
expects the official `tiny-imagenet-200.zip` or extracted `tiny-imagenet-200/`
under the raw directory; the preparer prints the upstream download URL if
missing. FI-2010 requires the manually downloaded z-scored no-auction data at:

```text
RAW/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/
    NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_9.txt
```

Preparation refuses to replace an existing cache. Dataset archives, cached
features, checkpoints, and generated training outputs are not distributed here.

### Frozen preprocessing and splits

| Dataset | Examples and model inputs | Split |
| --- | --- | --- |
| FI-2010 | First 40 book rows, 100-snapshot windows; target at horizon index 4, labels shifted to 0–2. One cumulative fold, stock 5 only. | Forward 40,000/10,000/20,000 windows, with 100 omitted windows at each boundary. |
| Speech Commands | Sorted 35 keyword IDs; first audio channel padded/truncated to 16,000 samples at 16 kHz. Log-mel arrays are 1×64×64, flattened for MLPs. | Balanced 20,020/5,005/10,010 examples from the upstream training subset. |
| Jannis | 54 float32 numerical features, sorted four-class label mapping. Transformer uses 54 tokens with one scalar each. | Balanced 3,840/960/1,920 examples. |
| Tiny ImageNet | Official training images, sorted class directories and filenames, RGB 3×64×64 float32 divided by 255. | Balanced 20,000/5,000/10,000, or 80,000/10,000/10,000 examples, depending on cohort. |

Balanced splits draw per-class rows without replacement with
`numpy.random.default_rng(20260812)`, then sort each subset's indices. The Tiny
80k held-out subsets are redrawn by that larger split; they are not asserted to
match the 20k held-out subsets. The FI-2010 stock block starts at column 228,633
of fold 9 and is checked against the original 149×362,400 file shape.

Speech preparation uses `MelSpectrogram(sample_rate=16000, n_fft=1024,
hop_length=250, n_mels=64)`, TorchAudio 2.1 defaults for remaining arguments,
and `AmplitudeToDB(top_db=80)`. Processing batches contain 256 clips; the first
64 time frames are retained. Images use no augmentation. Every continuous
coordinate is z-scored using **only the selected training examples**, replacing
standard deviations below `1e-8` with one.

The historical Tiny20 normalization computes mean and population standard
deviation in float32 (`float32_v15`). Tiny80 uses merged float64 moments with
64 MiB input chunks, casts the moments to float32, and normalizes in place
(`chunked_float64_v17`). These numerically distinct recipes are preserved.
Other datasets retain the historical NumPy float32 train-only calculation.

Caches contain numeric arrays `features_mlp`, `features_sequence`, `labels`,
and a `payload_sha256`. The runner recomputes the digest from the actual array
bytes, dtype strings and shape strings, then verifies the frozen split hash.
It also records the whole-file SHA-256; `--cache-sha256` can check an independently
recorded file checksum. A mismatch stops execution before training.

## Model and optimization details

MLPs have the frozen number of affine/ReLU/dropout hidden blocks and a separate
linear readout. Every weight, including the readout, is initialized with
Gaussian variance `sigma_w_sq / fan_in`; biases use variance `sigma_b_sq`.
The retained values are 1.98 and 0.02. Initialization is fixed across profiles.

Transformers use the frozen width, depth and head count, a learned class token
and positional embeddings, pre-LayerNorm residual blocks, bias-free QKV
projections, and ReLU feed-forward layers with expansion ratio four. The
block's configured dropout is applied separately to attention and feed-forward
residual outputs; attention probabilities have no dropout. Vision/audio models
use an 8×8 stride-8 convolutional patch embedding. FI-2010/Jannis use linear
continuous-token embeddings. Linear weights and class/position tokens use
truncated normal initialization with standard deviation 0.02; linear biases are
zero, LayerNorm weights one and biases zero. Patch convolutions preserve the
original PyTorch initialization. These transformer profiles are empirical
extensions of the MLP allocation rule.

Training uses AdamW with its PyTorch 2.1 defaults (`betas=(0.9,0.999)`,
`eps=1e-8`) and the frozen per-arm learning rate and weight decay. A multiplicative
cosine schedule starts at the selected learning rate and reaches 0.001 times
that rate during the final epoch. Batches shuffle using a dedicated seeded
generator, keep the final partial batch, and use zero data-loader workers.
Transformers clip gradients at the frozen norm (normally one); MLPs do not.
CUDA training uses the historical AMP scaler and bfloat16 autocast where
supported, otherwise float16; validation and test use full precision. CPU
training uses float32.

SHA-256 named streams separate initialization, minibatch order and dropout.
The dropout stream hashes the **entire original trial specification**, including
the original cohort ID, numeric JSON types, and reversal-pair profile family.
Changing `cohort_id`, `0.0` to `0`, or another apparently cosmetic specification
field can change the dropout stream. Configuration hashes are checked before
execution. Display IDs and output paths do not alter that specification.

The frozen learning rates/dropout budgets are those selected in the historical
validation searches, separately per arm. This runner executes those frozen
confirmation fits; it does not rerun hyperparameter search. Arms can have
different selected budgets. Big-step and linear extensions use their recorded
caps, which can exceed the primary cap-matched profile's cap. Linear-only
followups have no matched uniform arm in their cohort.

## Provenance and limits

[source_provenance.json](source_provenance.json) identifies the extracted source
files by SHA-256 and retained definitions. The implementation preserves the
historical model construction and training loop, including the recovered
Tiny20 versus Tiny80 preprocessing difference. Unused orchestration,
experiment families and optional parameterizations were removed. Per-cohort
historical evidence is recorded in [protocols.json](protocols.json).

The paper's saved summaries are evidence of the original runs. Executing this
package produces new runs. Matching configuration, seeds and split hashes is
necessary, but does not guarantee bitwise equality across GPUs, kernels or
package versions. No complete original data/cache archive is distributed;
rebuilding raw datasets with changed decoding or dataset-library behavior may
produce a different payload and is rejected against the frozen cohort. The
optional preparation requirements pin the available reconstruction environment,
not a claim that every original raw-data dependency was archived. The preserved
historical split hashes are the acceptance check for rebuilt caches.

Run the benchmark-specific checks with:

```sh
python -m pip install pytest
python -m pytest tests/test_benchmark_runner.py
```
