# Saved results

These are the numerical results supplied with the original ICML submission.

| File | Comparison |
|---|---|
| [mlp_schedules.npz](mlp_schedules.npz) | CIFAR-10 MLP dropout schedules |
| [mlp_budget_controls.npz](mlp_budget_controls.npz) | MLP budget controls and concentrated dropout |
| [vit_schedules.json](vit_schedules.json) | CIFAR-100 ViT dropout schedules |
| [vit_ablation.json](vit_ablation.json) | CIFAR-10 ViT component ablations |

The MLP archives retain the original arrays, theory values, and configuration;
they were converted from pickle to NPZ without changing the numbers.
Load them with `utils.results.load_npz_result`. The ViT files use JSON.
Training and test accuracy are recorded as percentages.

The original experiments record test loss at each epoch. Final-epoch comparisons
and comparisons of the lowest recorded test loss answer different questions;
the latter use the test set to choose the epoch. These files have no separate
validation histories.

Run `python scripts/plot_results.py` from the repository root to plot the saved
curves. Notebook reruns write into `runs/`.
