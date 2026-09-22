# Additional seeds, September 22, 2026

We ran 114 additional fits: 57 matched uniform/frontloaded seed pairs across
eleven comparisons. Schedules and hyperparameters were frozen before these
runs. The main tables combine the historical measurements with these new
seeds; no profile was reselected from the new results.

## What changed

| Comparison | Historical pairs | New seed IDs | Pooled pairs | Final-epoch pairs |
|---|---:|---|---:|---:|
| CIFAR-10 ReLU, p=0.1 | 3 | 45–51 | 10 | 10 |
| CIFAR-10 ViT, both-block ablation | 5 | 47–51 | 10 | 10 |
| Speech Commands 20k, MLP and Transformer | 5 each | 105–109 | 10 each | 5 each |
| Jannis, standard MLP and Transformer | 5 each | 105–109 | 10 each | 5 each |
| Tiny ImageNet 20k, MLP and Transformer | 5 each | 105–109 | 10 each | 5 each |
| Tiny ImageNet 80k, MLP and Transformer | 5 each | 105–109 | 10 each | 10 each |
| FI-2010, MLP | 5 | 105–109 | 10 | 5 |

The CIFAR-10 schedule and budget comparisons already had 25 pairs, and the
GELU, CIFAR-100 ViT, and extended-search Jannis Transformer comparisons already
had ten. Those five rows did not receive new runs. The two 25-seed comparisons
share their uniform runs. Extended-search Jannis still lacks final endpoints.

For benchmarks, each run's checkpoint is its first minimum-validation-loss
checkpoint. The final endpoint uses its last training epoch. Where historical
final measurements are absent, the final comparison uses only the five fresh
pairs. The original CIFAR runs evaluated test loss each epoch and have no
validation split; their minimum test loss is retrospective.

## Inspect the measurements

[seed_extension.npz](seed_extension.npz) contains the 114 fresh fits, including
their specifications, learning curves, test endpoints, seed IDs, data/split
identifiers, and original result-file hashes. It uses the same pickle-free
format as the other result archives:

```python
from utils.results import load_npz_result

extension = load_npz_result("results/seed_extension.npz")
fit = extension["fits"][0]
print(fit["task"], fit["spec"])
print(fit["curves"].keys())
```

Benchmark training and validation accuracies, and their saved test accuracies,
are fractions. Original CIFAR accuracy curves explicitly use percent units.
The combined [paired measurements](confidence_seed_metrics.json) convert all
accuracy vectors to percentages. Each metric lists its own ordered seed IDs;
the row also records historical and extension seed IDs, allowing either cohort
to be analyzed separately. Test loss was not measured each epoch for benchmarks.

All fits completed successfully. The downloaded metric files were checked
against their completion-record SHA-256 hashes; endpoint selection and paired
seed IDs were checked before combining them. An independent reconstruction of
all 56 populated table entries reproduced the means and confidence intervals.
The archive contains measurements, not the model checkpoint files.

To regenerate the public tables and intervals:

```sh
python scripts/confidence_intervals.py
```

Outputs go to `runs/confidence/`. Loss reductions use a ratio of paired seed
means and a 95% Fieller interval; accuracy gains use paired Student-t intervals
in percentage points. They describe seed variation conditional on the recorded
split and recipes, without adjustment for the earlier profile search or for
multiple comparisons.

## Repeat the frozen fits

The nine benchmark comparisons use the existing [benchmark runner](../benchmarks/README.md).
For example, the new Speech MLP pairs can be repeated with:

```sh
python -m benchmarks.run train \
  --cohort speech_commands-mlp-zero-decay \
  --profile uniform frontloaded --seed 105 106 107 108 109 \
  --data-root /path/to/data --output-dir /path/to/new-runs --device cuda
```

Use `python -m benchmarks.run list` for the other cohort IDs. The extended-search
Jannis cohort is separate from the standard zero-weight-decay comparison.
The [original CIFAR extension runner](../benchmarks/original/README.md) provides
the frozen ReLU and both-block ViT recipes and their new seed ranges.

The completed extensions used an NVIDIA H100, Python 3.11.15, PyTorch
2.1.0+cu121, NumPy 1.24.3, and CUDA 12.1. Source checks and GPU validation
confirmed the recovered model/training arithmetic and all full cached splits.
Tiny ImageNet 20k retains float32 normalization; 80k retains chunked float64
moment accumulation followed by float32 normalization. The ReLU extension
makes NCHW tensors contiguous before flattening, preserving feature order.
These are fixed-recipe extensions, not claims of bitwise historical replay:
GPUs and kernels can differ from the earlier runs.

The historical multi-profile archives and appendix figures keep their original
seed counts. The updated main tables explicitly include the additional pairs.
