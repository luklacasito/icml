# Speech Commands with held-out speakers

The paper's saved Speech Commands runs use a balanced random split of clips
from the official training subset. Reconstructing those filenames shows that
4,979 of 5,005 validation clips and 9,967 of 10,010 test clips have a speaker
also present in training. The clips themselves do not overlap. The
[speaker audit](../results/speech_speaker_audit.json) gives counts and split
hashes; those results measure generalization to other clips, not unseen speakers.

`speech_commands_speakers` is a separate experiment. It keeps the exact 20,020
training filenames and uses **all 9,981 official validation clips and 11,005
official test clips**. The three speaker sets are disjoint. Evaluation sets keep
the official class frequencies: the rarest classes have only 128 validation
and 155 test clips, so the previous balanced sizes cannot be reused.

## Prepare the separate cache

```sh
python -m benchmarks.prepare speech_commands_speakers \
  --root /path/to/data --raw /path/to/raw
```

The raw directory may reuse the existing Speech Commands v0.02 archive or
extracted files. This writes `benchmarks/speech_commands_speakers.npz`, leaving
`speech_commands.npz` unchanged. It stores filenames, speaker IDs, official
partition codes, selected indices and a JSON split manifest alongside the
features. Loading checks these identities, speaker separation, class labels,
counts and manifest hashes before training.

The spectrogram settings remain 64 mel bands, FFT size 1,024, hop 250 and the
first 64 frames. One preprocessing detail also changes: the 80 dB cutoff is
computed **per clip**, using an explicit channel dimension. The old code passed
a three-dimensional batch to TorchAudio, which shares one cutoff across that
batch. The new features therefore do not depend on which other clips happen to
be processed alongside them. Normalization uses only the selected training clips.

## Run ten paired seeds

These commands transfer the saved architecture, learning rate and dropout
settings to the new split. They **do not retune** those settings. Validation on
unseen speakers chooses the epoch separately for every new run; test loss and
accuracy are measured at that epoch and at the end of training.

```sh
python -m benchmarks.run train \
  --cohort speech_commands-mlp-zero-decay \
  --dataset speech_commands_speakers --profile uniform frontloaded \
  --seed 200 201 202 203 204 205 206 207 208 209 \
  --data-root /path/to/data --output-dir /path/to/new-runs --device cuda

python -m benchmarks.run train \
  --cohort speech_commands-transformer-zero-decay \
  --dataset speech_commands_speakers --profile uniform frontloaded \
  --seed 200 201 202 203 204 205 206 207 208 209 \
  --data-root /path/to/data --output-dir /path/to/new-runs --device cuda
```

`frontloaded` means early step for the MLP and big step for the Transformer.
Each model has 20 runs, for 40 in total. Add `--describe` to inspect the settings
without preparing data or training. Output directories and run specifications
use the new dataset name and `transferred-settings` suffix. Each run saves both
sets of weights, training/validation curves, test measurements, configuration
and data hashes. The old results and their protocol hashes remain unchanged.

The new results must be reported separately until complete; they cannot be
pooled with the random-clip runs. Evaluating old saved weights on the official
test set is a useful additional check, but those weights were selected using
validation speakers also present in training, and require the original
preprocessing and normalization.
