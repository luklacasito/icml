# Saved results

These are the numerical results used in the camera-ready paper.

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
