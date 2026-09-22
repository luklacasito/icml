# Saved results

These are the measurements behind the paper. The original CIFAR experiments
include learning curves; the additional benchmark comparisons contain paired
test measurements at the selected checkpoints and, where recorded, at the end
of training.

## Main comparisons

| File | Contents |
|---|---|
| [Confidence intervals](confidence_intervals.md) | Both main tables, endpoint definitions, and statistical assumptions |
| [Paired seed measurements](confidence_seed_metrics.json) | Uniform and selected-schedule results, seed pairing, and source hashes |
| [Intervals, JSON](confidence_intervals.json) | Unrounded estimates and confidence bounds |
| [Intervals, CSV](confidence_intervals.csv) | The same comparisons in a flat table |

The [analysis script](../scripts/confidence_intervals.py) computes the intervals
with the recorded schedules held fixed. These comparisons reuse the existing
runs, so the intervals describe seed variation on that split and do not account
for the earlier choice of schedule or hyperparameters. The additional benchmark
file contains checkpoint and final test measurements, without full learning
curves or a new independent confirmation study.

The saved Speech Commands records contain test measurements at the checkpoint
selected by validation loss, but no final-epoch test measurements for either
the MLP or the Transformer. All 50 epochs finished. Their final learning-curve
values are validation losses and cannot fill the missing test-loss columns.

## CIFAR experiments

| File | Comparison |
|---|---|
| [mlp_schedules.npz](mlp_schedules.npz) | CIFAR-10 MLP dropout schedules |
| [mlp_budget_controls.npz](mlp_budget_controls.npz) | Early concentration versus increasing uniform dropout |
| [vit_schedules.json](vit_schedules.json) | CIFAR-100 ViT dropout schedules |
| [vit_ablation.json](vit_ablation.json) | CIFAR-10 ViT: attention, MLP branch, or both |
| [mlp_dropout_sweep.npz](mlp_dropout_sweep.npz) | ReLU MLP dropout strengths; 3 seeds, 125 epochs |
| [mlp_width_sweep.npz](mlp_width_sweep.npz) | ReLU MLP widths; 10 seeds, 75 epochs |
| [mlp_gelu.npz](mlp_gelu.npz) | GELU MLP dropout strengths; 10 seeds, 75 epochs |

The MLP files preserve the original arrays, reference theory values, and
configuration, converted from pickle to NPZ without changing the numbers.
Use `utils.results.load_npz_result` to read them. The ViT files are JSON, and
all training and test accuracies are percentages.

The CIFAR-100 ViT archive has ten 75-epoch runs per profile but no configuration.
The original notebook was set to a one-seed, small-data pilot, so its settings
cannot establish how the saved curves were produced. The current notebook
follows the paper's reported recipe; the historical dataset size and training
settings remain unverified.

The sweeps contain `all_results`, `theory`, and `config` dictionaries. Keys in
`all_results` and `theory` represent `(dropout_strength, schedule)` or
`(width, schedule)` pairs, stored as strings and read with `ast.literal_eval`.
ReLU and GELU dropout sweeps reuse one no-dropout run set across strengths;
the width sweep trains a baseline at each width. The ReLU dropout notebook
defaults to the saved three-seed, 125-epoch experiment.

These experiments record test loss every epoch, which lets us ask both how
well the model finishes and how low its test loss ever gets. The second question
uses the test set to choose an epoch; there are no separate validation histories
in these files. That is why the paper distinguishes these results from the
validation-selected comparisons above.

Run `python scripts/plot_results.py` from the repository root for the learning
curves. Notebook reruns write into `runs/`.

## Mean-field calculations

| File | Contents |
|---|---|
| [mean_field.npz](mean_field.npz) | Numerical curves and fit masks |
| [mean_field.json](mean_field.json) | Exponents, fit errors, fitting windows, and source hash |
| [scaling_collapse.npz](scaling_collapse.npz) | Smooth and kinked fixed points used in the collapse plots |
| [scaling_collapse.json](scaling_collapse.json) | Array columns, source hash, and parameter conventions |
| [hermite_coefficients.json](hermite_coefficients.json) | Exact ReLU coefficients and tanh quadrature values |

Run `python scripts/plot_mean_field.py` to recompute these files and figures.
Here the curves are deterministic, and a small regression error only means
that the chosen points sit close to a power law. Quadrature error, finite
iteration counts, and distance from the asymptotic limit can still shift the
fitted exponent.

The tunable ReLU channel probes the local normal form with independently varied
parameters; those parameters need not describe a realizable finite-variance
network initialization. The paper states this restriction alongside the fits.
