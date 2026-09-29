# Figures, data and code

Figure numbers refer to the September 29 paper. Run commands from the repository
root with the environment described in the [README](../README.md).

```bash
python scripts/build_figures.py
```

This generates the plots below in `runs/figures/`, with the same directory layout
as `manuscript/figures/`. It recomputes the mean-field calculations and reads
saved training measurements; it does not train models or download datasets.
The river and regularization-reach PDFs are supplied vector illustrations and
are copied unchanged. Their drawing code is not part of the figure builder.

Every plotting script accepts `--output-dir`. The older `--output` spelling
remains an alias in scripts that previously used it. For example:

```bash
python scripts/plot_sliding_block.py --output-dir runs/my-figures --png
```

## Figure index

Run a listed script with `python scripts/SCRIPT.py`. Mean-field calculations
write their recomputed arrays beside their figures; the copies in `results/`
remain unchanged.

| Fig. | Subject | Plot script | Measurements | Training or calculation source |
|---|---|---|---|---|
| 1 | Critical exponents | [plot_mean_field.py](../scripts/plot_mean_field.py) | [mean_field.npz](../results/mean_field.npz), [fit definitions](../results/mean_field.json) | [mean_field.ipynb](../notebooks/mean_field.ipynb) |
| 2 | Smooth scaling collapse | [plot_mean_field.py](../scripts/plot_mean_field.py) | [scaling_collapse.npz](../results/scaling_collapse.npz) | [mean_field.ipynb](../notebooks/mean_field.ipynb), [scaling.py](../utils/scaling.py) |
| 3 | River thought experiment | Supplied [river_ink.pdf](figures/theory/river_ink.pdf) | Illustration | No training |
| 4 | Moving the dropout block | [plot_sliding_block.py](../scripts/plot_sliding_block.py) | [sliding_block.json](../results/sliding_block.json) | [Exact runner and commands](../benchmarks/CIFAR.md#moving-the-block) |
| 5 | CIFAR test curves | [plot_paper_results.py](../scripts/plot_paper_results.py) | [mlp_budget_controls.npz](../results/mlp_budget_controls.npz) | [mlp_schedules.ipynb](../notebooks/mlp_schedules.ipynb), [CIFAR extensions](../benchmarks/original/README.md) |
| 6 | Correlations before and after training | [plot_cifar_correlations.py](../scripts/plot_cifar_correlations.py) | [cifar_correlations.json](../results/cifar_correlations.json) | [Exact training and probe code](../benchmarks/CIFAR.md#dropout-rates-and-activation-correlations) |
| 7 | Regularization reach | Supplied [regularization_reach.pdf](figures/theory/regularization_reach.pdf) | Analytic curves defined in the caption | No training |
| 8 | Hermite coefficients | [plot_mean_field.py](../scripts/plot_mean_field.py) | [hermite_coefficients.json](../results/hermite_coefficients.json) | `plot_hermite` in the same script |
| 9 | Kinked scaling collapse | [plot_mean_field.py](../scripts/plot_mean_field.py) | [scaling_collapse.npz](../results/scaling_collapse.npz) | [mean_field.ipynb](../notebooks/mean_field.ipynb), [scaling.py](../utils/scaling.py) |
| 10 | MLP schedules | [plot_paper_results.py](../scripts/plot_paper_results.py) | [mlp_schedules.npz](../results/mlp_schedules.npz) | [mlp_schedules.ipynb](../notebooks/mlp_schedules.ipynb) |
| 11 | Dropout-budget controls | [plot_paper_results.py](../scripts/plot_paper_results.py) | [mlp_budget_controls.npz](../results/mlp_budget_controls.npz) | [mlp_schedules.ipynb](../notebooks/mlp_schedules.ipynb), [CIFAR runner](../benchmarks/run_original.py) |
| 12 | ReLU dropout and width sweeps | [plot_paper_results.py](../scripts/plot_paper_results.py) | [mlp_dropout_sweep.npz](../results/mlp_dropout_sweep.npz), [mlp_width_sweep.npz](../results/mlp_width_sweep.npz) | [Dropout notebook](../notebooks/mlp_dropout_sweep.ipynb), [width notebook](../notebooks/mlp_width_sweep.ipynb) |
| 13 | GELU dropout sweep | [plot_paper_results.py](../scripts/plot_paper_results.py) | [mlp_gelu.npz](../results/mlp_gelu.npz) | [mlp_gelu.ipynb](../notebooks/mlp_gelu.ipynb) |
| 14 | CIFAR-100 ViT schedules | [plot_paper_results.py](../scripts/plot_paper_results.py) | [vit_schedules.json](../results/vit_schedules.json) | [vit_schedules.ipynb](../notebooks/vit_schedules.ipynb); see the [configuration gap](../results/cifar100_configuration.json) |
| 15 | Transformer component ablations | [plot_paper_results.py](../scripts/plot_paper_results.py) | [vit_ablation.json](../results/vit_ablation.json) | [vit_ablation.ipynb](../notebooks/vit_ablation.ipynb) |
| 16 | Speech Commands | [plot_benchmarks.py](../scripts/plot_benchmarks.py) | [benchmark_curves.npz](../results/benchmark_curves.npz) | [Benchmark runner and data preparation](../benchmarks/README.md) |
| 17 | Tiny ImageNet, 20k | [plot_benchmarks.py](../scripts/plot_benchmarks.py) | [benchmark_curves.npz](../results/benchmark_curves.npz) | [Benchmark runner](../benchmarks/README.md) |
| 18 | Tiny ImageNet, 80k | [plot_benchmarks.py](../scripts/plot_benchmarks.py) | [benchmark_curves.npz](../results/benchmark_curves.npz) | [Benchmark runner](../benchmarks/README.md) |
| 19 | FI-2010 MLP | [plot_benchmarks.py](../scripts/plot_benchmarks.py) | [benchmark_curves.npz](../results/benchmark_curves.npz) | [Benchmark runner](../benchmarks/README.md) |
| 20 | Dropout-profile pilot | [plot_benchmarks.py](../scripts/plot_benchmarks.py) | [benchmark_curves.npz](../results/benchmark_curves.npz), including run settings | Saved curves; no dedicated pilot training runner included |

## Tables and uncertainty

`python scripts/confidence_intervals.py` writes the paper's comparisons to
`runs/confidence/`, using [paired seed measurements](../results/confidence_seed_metrics.json).
It computes paired Fieller intervals for relative loss reductions and paired
Student-t intervals for accuracy gains. The [results guide](../results/README.md)
explains selection, sample counts and limitations.

The figures and tables need not use the same seed counts: some tables include
additional runs of the selected profiles, while multi-profile figures retain
the complete earlier comparisons. Each caption gives its own count and whether
bands show SEM or a confidence interval. Supplementary comparisons have a
[separate index](../results/supplementary/README.md).
