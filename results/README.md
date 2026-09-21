# Saved results

The original archives below contain the numerical results used in the
camera-ready paper. The expanded paper adds paired endpoint data and
confidence intervals for its two main comparison tables.

## Main comparison tables

| File | Contents |
|---|---|
| [Confidence intervals](confidence_intervals.md) | Both main tables, endpoint definitions, and statistical assumptions |
| [Paired seed metrics](confidence_seed_metrics.json) | Uniform and selected-schedule endpoints, seed pairing, and source hashes |
| [Calculated intervals, JSON](confidence_intervals.json) | Unrounded point estimates and confidence bounds |
| [Calculated intervals, CSV](confidence_intervals.csv) | The same comparisons as a flat table |

Run the [analysis script](../scripts/confidence_intervals.py) to reproduce these
results. The selected schedules remain fixed during the calculation. The
additional benchmark data contain test metrics at validation-selected
checkpoints, plus final-epoch metrics where recorded; they are not full training
histories or a new independent confirmation study.

## Original experiment archives

| File | Comparison |
|---|---|
| [mlp_schedules.npz](mlp_schedules.npz) | CIFAR-10 MLP dropout schedules |
| [mlp_budget_controls.npz](mlp_budget_controls.npz) | MLP budget controls and concentrated dropout |
| [vit_schedules.json](vit_schedules.json) | CIFAR-100 ViT dropout schedules |
| [vit_ablation.json](vit_ablation.json) | CIFAR-10 ViT component ablations |
| [mlp_dropout_sweep.npz](mlp_dropout_sweep.npz) | CIFAR-10 ReLU MLP dropout strengths; 3 seeds, 125 epochs |
| [mlp_width_sweep.npz](mlp_width_sweep.npz) | CIFAR-10 ReLU MLP widths; 10 seeds, 75 epochs |
| [mlp_gelu.npz](mlp_gelu.npz) | CIFAR-10 GELU MLP dropout strengths; 10 seeds, 75 epochs |

The MLP archives retain the original arrays, theory values, and configuration;
they were converted from pickle to NPZ without changing the numbers.
Load them with `utils.results.load_npz_result`. The ViT files use JSON.
Training and test accuracy are recorded as percentages.

The supplemental sweeps use `all_results`, `theory`, and `config` dictionaries.
Keys in `all_results` and `theory` encode `(dropout_strength, schedule)` or
`(width, schedule)` pairs as strings; the notebooks restore them with
`ast.literal_eval`. The ReLU and GELU dropout sweeps share one no-dropout result
across dropout strengths. The width sweep trains a separate baseline at each width.
The ReLU dropout notebook defaults to the archived 3-seed, 125-epoch run.

The original experiments record test loss at each epoch. Final-epoch comparisons
and comparisons of the lowest recorded test loss answer different questions;
the latter use the test set to choose the epoch. These files have no separate
validation histories.

Run `python scripts/plot_results.py` from the repository root to plot the saved
curves. Notebook reruns write into `runs/`.
