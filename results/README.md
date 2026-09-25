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
| [Seed extension](seed_extension.md) | New run counts, frozen settings, and reproduction instructions |
| [Extension measurements](seed_extension.npz) | All 114 new fits: learning curves, endpoints, specifications, and source/data hashes |

The [analysis script](../scripts/confidence_intervals.py) computes the intervals
with the recorded schedules held fixed. The September 22 update adds 114 fits
across eleven comparisons. Profiles and hyperparameters were frozen before
these new seeds; the main tables pool historical and new measurements. The
intervals describe seed variation on that split and do not account for the
earlier choice of schedule or hyperparameters. Each endpoint in
`confidence_seed_metrics.json` lists its own seed IDs, so a five-pair final
measurement cannot be mistaken for a ten-pair checkpoint measurement.

The historical Speech Commands records contain checkpoint test measurements
but no complete paired final test measurements. Their final learning-curve
values are validation losses and cannot fill that gap. The five new pairs
retain both test endpoints: the pooled checkpoint comparison has ten pairs,
while the final comparison has five. The same distinction applies to standard
Jannis, Tiny ImageNet 20k, and FI-2010. Tiny ImageNet 80k has ten pairs at both
endpoints; extended-search Jannis still has no final measurements.

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

These multi-profile archives remain the historical measurements. The new
ReLU and both-block ViT pairs live in [seed_extension.npz](seed_extension.npz);
the main confidence tables combine the two cohorts, while historical plots
retain their original counts.

Run `python scripts/plot_results.py` from the repository root for the learning
curves. Notebook reruns write into `runs/`.

## Historical benchmark curves

[benchmark_curves.npz](benchmark_curves.npz) contains the exact training and
validation arrays behind the thirteen multi-panel appendix figures: 350 runs
from twelve retained benchmark cohorts, plus the 30-run profile-geometry pilot.
It preserves run IDs, per-run configurations, seed order, and the source archive
hashes. The confirmation records came from the September 15 W&B export;
excluded Amazon Reviews and zero-weight-decay FI-2010 Transformer cohorts are
not included. The separate FI-2010 Transformer linear follow-up is retained.

Each profile stores one row per seed and one column per zero-based epoch.
Accuracies are fractions in the archive and percentages in the figures.
Shaded bands are the sample standard deviation divided by the square root of
the seed count. Only training loss uses a logarithmic axis. These curves remain
the historical five- or ten-seed cohorts (three seeds for the pilot); they do
not pool in the later extension or substitute validation loss for test loss.

Run `python scripts/plot_benchmarks.py` to reproduce them in
`runs/figures/benchmarks/`. The exporter uses the same schedule colors as the
presentation and the other paper figures.

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

The ReLU collapse holds the actual field `h` fixed, setting `rho=chi/(chi+h)`.
Its rescaling uses the local coefficient `kappa_loc=chi*2*sqrt(2)/(3*pi)` from
the full map. The archive saves that coefficient for each point. The smooth
scan instead fixes dropout probability and recomputes its field and curvature
at each point. Both plots retain finite-field departures from the leading
equation of state; the horizontal axis runs opposite to the paper's `u`.
