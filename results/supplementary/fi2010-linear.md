# FI-2010 Transformer: linear-dropout follow-up

This cohort compares a decreasing linear dropout profile with tuned no dropout.
It has no matching uniform-dropout baseline, so it cannot answer whether moving
the same dropout budget toward the input improves on uniform allocation.

## Saved results

Both profiles have five seeds (100–104). Test cross-entropy (CE) and accuracy
use each run's first minimum-validation-loss checkpoint. Values are means ± SEM.

| Profile | Mean dropout | Learning rate | Test CE | Test accuracy (%) |
|---|---:|---:|---:|---:|
| No dropout (tuned) | 0.00 | 0.001 | 0.955 ± 0.010 | 55.38 ± 0.39 |
| Linear decreasing | 0.10 | 0.001 | 0.930 ± 0.006 | 57.72 ± 0.83 |

Final-epoch test evaluations were not recorded. The [learning curves](fi2010-transformer-linear.pdf)
show training and validation metrics, with SEM bands; their final values are
not final test measurements. Training loss uses a logarithmic axis. Training
metrics average batches during optimization with dropout enabled, whereas
validation uses each epoch's final weights with dropout disabled.

## Protocol

- **Data:** FI-2010, with 40,000/10,000/20,000 training/validation/test windows
  from a single stock, in chronological order with 100-window gaps. Each input
  contains 100 snapshots of 40 features; normalization uses training data only.
- **Model:** 12 pre-LayerNorm Transformer blocks, width 256, eight heads, and
  ReLU feedforward width 1024. Each 40-feature snapshot projects to one token.
  Dropout acts on the attention and feedforward outputs before residual addition.
- **Training:** 50 epochs, batch size 128, AdamW, weight decay 1e-7, gradient
  norm clipping at one, and cosine learning-rate decay to 0.001 times the
  initial rate. Each arm uses initial learning rate 0.001.
- **Dropout:** the linear profile decreases from 0.20 to zero across 12 blocks,
  with mean 0.10; the control has no dropout.
- **Selection:** the documented validation search considers learning rates
  {5e-5, 1e-4, 3e-4, 1e-3, 3e-3} separately for the two linear directions and
  no dropout, then confirms the selected direction and tuned control. The full
  historical tuning ledger and original source revision were not recovered.

[Frozen configurations](../../benchmarks/protocols.json) record the
`fi2010-transformer-linear-followup` cohort, its layerwise probabilities,
configuration hashes, seed IDs, and split hash. The
[benchmark guide](../../benchmarks/README.md) gives data-preparation and training
instructions. [benchmark_curves.npz](../benchmark_curves.npz) stores the original
seed histories under `figures["fi2010-transformer-linear"]`.

To regenerate the individual learning-curve figures:

```sh
python scripts/plot_benchmarks.py --individual
```
