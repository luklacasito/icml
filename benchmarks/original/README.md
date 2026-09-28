# Original CIFAR seed extensions

This runner reproduces the two fixed CIFAR-10 comparisons extended in September
2026. It retains the completed fits' ReLU/ViT definitions and execution order,
without validation selection or another profile search.

| Study | Frozen comparison | Original seeds | Added seeds | Training |
|---|---|---|---|---|
| `relu` | Uniform versus Big step, mean dropout 0.10 | 42–44 | **45–51**, both arms; 14 fits | 5,000 train / 5,000 test; 6 hidden layers, width 256; 125 epochs |
| `vit` | Uniform versus Step early, dropout in both residual branches | 42–46 | **47–51**, both arms; 10 fits | 50,000 train / 10,000 test; 12 blocks, width 128, 16 heads; 75 epochs |

[recipes.json](recipes.json) freezes the full settings, including initializers,
batch sizes, optimizer, learning rates and schedules. The recipe keeps the
original zero-based seed convention. No additional CIFAR experiments are run.

## Run

Use Python 3.11 with the repository's `requirements.txt`, including
`numpy==1.24.3`, `torch==2.1.0`, and `torchvision==0.16.0`. From the repository root:

```bash
python -m benchmarks.run_original smoke \
  --run-root /tmp/original-extension-smoke --device cpu

python -m benchmarks.run_original run \
  --study relu --arm uniform --seed 45 \
  --data-root /path/to/cifar-cache --run-root /path/to/new-runs --device cuda
python -m benchmarks.run_original run \
  --study relu --arm frontloaded --seed 45 \
  --data-root /path/to/cifar-cache --run-root /path/to/new-runs --device cuda

python -m benchmarks.run_original run \
  --study vit --arm uniform --seed 47 \
  --data-root /path/to/cifar-cache --run-root /path/to/new-runs --device cuda
python -m benchmarks.run_original run \
  --study vit --arm frontloaded --seed 47 \
  --data-root /path/to/cifar-cache --run-root /path/to/new-runs --device cuda
```

Repeat the ReLU commands for seeds 45–51 and the ViT commands for seeds 47–51
to reproduce the 24 added fits. Smoke uses synthetic images, two epochs and
small sample counts; it executes both arms of each model and verifies that
completed fits can be reopened without retraining. It is not a scientific run.

`--data-root` contains the standard TorchVision `cifar-10-batches-py/` directory.
TorchVision downloads the official data if needed and checks its release MD5s.
The runner additionally hashes every CIFAR batch and metadata file and records
the selected row/label hashes. An expected aggregate SHA-256 can be supplied
with `--cache-sha256`; this is the hash of the canonical per-file hash mapping,
not the download archive's checksum.

ReLU subsets use one `numpy.random.RandomState(0)`: choose training rows, then
test rows, without replacement. The full-data ViT retains original row order.
Both normalize channels with the recorded CIFAR means and standard deviations.
ReLU inputs are made contiguous in NCHW order because its frozen model uses
`view(batch, -1)`; the adapter changes memory layout, not feature order.

## Outputs and endpoints

Each fit writes to `original/STUDY/ARM/seed-N/`. `result.json` stores the complete
training/test curves, final metrics, retrospective minimum test loss, input
fingerprints, source hashes and runtime. `final.pt` contains the final model's
weights and its endpoint/epoch identifiers. Accuracies are percentages.

Test is evaluated every epoch. Minimum test loss is a retrospective test-selected
summary, not a validation-selected checkpoint. Only final weights are retained;
the minimum-test epoch has no separately saved checkpoint. Interrupted fits
restart from the same seed; completed fits are reused only after their source,
data, specification and final-checkpoint checksum match.

## Provenance

[provenance.json](provenance.json) maps the completed execution's frozen files
and their hashes to these public definitions. `relu.py` and `vit.py` are retained
verbatim. ViT training and schedules reuse `utils/training.py` and
`utils/schedules.py`; their definitions match the completed execution's extracted
support code exactly. The public runner removes cluster identifiers and adapts
imports and source-hash collection; its training operations are unchanged.

New runs hash the **public** runner, both frozen model files, recipes, provenance,
and imported utilities. Their source fingerprint is deliberately distinct from
the archived execution fingerprint. Recorded original hashes describe the
completed fits, not a claim that a new run used the old wrapper. Numerical
identity across library versions or devices is not guaranteed. The extension
preserves the fixed comparison and data split; it does not independently select
or validate the profile choice.
