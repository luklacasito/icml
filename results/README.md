# Saved results

These are the measurements behind the paper. The original CIFAR experiments
include learning curves; the additional benchmark comparisons contain paired
test measurements at the epochs chosen by validation loss and, where recorded, at the end
of training.

## Main comparisons

The table below covers the paper. [Supplementary results](supplementary/README.md)
index the separate CIFAR tuning studies, Jannis and the FI-2010 linear follow-up.
Shared measurement archives retain those runs without changing their contents.

| File | Contents |
|---|---|
| [Confidence intervals](confidence_intervals.md) | CIFAR and benchmark tables, endpoint definitions, and statistical assumptions |
| [Paired seed measurements](confidence_seed_metrics.json) | Uniform and selected-schedule results, seed pairing, and source hashes |
| [Intervals, JSON](confidence_intervals.json) | Unrounded estimates and confidence bounds |
| [Intervals, CSV](confidence_intervals.csv) | The same comparisons in a flat table |
| [Seed extension](seed_extension.md) | New run counts, settings kept unchanged, and reproduction instructions |
| [Extension measurements](seed_extension.npz) | All 114 new fits: learning curves, endpoints, specifications, and source/data hashes |

The [analysis script](../scripts/confidence_intervals.py) computes the intervals
with the recorded schedules held fixed. The September 22 update adds 114 fits
across eleven comparisons. Profiles and hyperparameters were fixed before
these new seeds; the main tables pool earlier and new measurements. The
intervals describe seed variation on that split and do not account for the
earlier choice of schedule or hyperparameters. Each endpoint in
`confidence_seed_metrics.json` lists its own seed IDs, so a five-pair final
measurement cannot be mistaken for a ten-pair measurement at the best validation epoch.

The earlier Speech Commands records contain test measurements at the epochs chosen by validation loss
but no complete paired final test measurements. Their final learning-curve
values are validation losses and cannot fill that gap. The five new pairs
retain both test endpoints: the comparison at the best validation epoch has ten pairs,
while the final comparison has five. The same distinction applies to standard
Jannis, Tiny ImageNet 20k, and FI-2010. Tiny ImageNet 80k has ten pairs at both
endpoints; extended-search Jannis still has no final measurements.

The Speech results use a random clip split, with speakers shared across
training and evaluation. The [speaker audit](speech_speaker_audit.json) gives
the overlap counts; the [separate experiment guide](../benchmarks/SPEECH.md)
describes training with held-out speakers.

## CIFAR experiments

| File | Comparison |
|---|---|
| [mlp_schedules.npz](mlp_schedules.npz) | CIFAR-10 MLP dropout schedules |
| [mlp_budget_controls.npz](mlp_budget_controls.npz) | Early concentration versus increasing uniform dropout |
| [sliding_block.json](sliding_block.json) | Six-layer CIFAR-10 MLP: four positions of a three-layer dropout block, five paired seeds |
| [cifar_correlations.json](cifar_correlations.json) | Twelve-layer MLP: ten uniform dropout rates, five seeds, correlations at initialization and two trained checkpoints |
| [vit_schedules.json](vit_schedules.json) | CIFAR-100 ViT dropout schedules |
| [cifar100_configuration.json](cifar100_configuration.json) | Dataset-size evidence, reported settings and remaining CIFAR-100 configuration gap |
| [vit_ablation.json](vit_ablation.json) | CIFAR-10 ViT: attention, MLP branch, or both |
| [mlp_dropout_sweep.npz](mlp_dropout_sweep.npz) | ReLU MLP dropout strengths; 3 seeds, 125 epochs |
| [mlp_width_sweep.npz](mlp_width_sweep.npz) | ReLU MLP widths; 10 seeds, 75 epochs |
| [mlp_gelu.npz](mlp_gelu.npz) | GELU MLP dropout strengths; 10 seeds, 75 epochs |

The MLP files preserve the original arrays, reference theory values, and
configuration, converted from pickle to NPZ without changing the numbers.
Use `utils.results.load_npz_result` to read them. The ViT files are JSON, and
all training and test accuracies are percentages.

The CIFAR-100 ViT archive has ten 75-epoch runs per profile but no saved
configuration or seed identifiers. The [configuration evidence](cifar100_configuration.json)
traces the unchanged result file to the original Git commit. Under the original
notebook's evaluator, the accuracy increments identify the full 50,000-image
training set and 10,000-image test set: each accuracy is 100 times the integer
count of correct predictions divided by the number of examples, with each image
counted once. This rules out the notebook's stale 2,000/5,000-example pilot settings.
The architecture and optimizer settings remain those reported in the notebook;
no configuration links them to the archived runs. Run
`python scripts/audit_cifar100.py` from the repository root to verify the file
fingerprint, curve dimensions and accuracy denominators.

The sweeps contain `all_results`, `theory`, and `config` dictionaries. Keys in
`all_results` and `theory` represent `(dropout_strength, schedule)` or
`(width, schedule)` pairs, stored as strings and read with `ast.literal_eval`.
ReLU and GELU dropout sweeps reuse one no-dropout run set across strengths;
the width sweep trains a baseline at each width. The ReLU dropout notebook
defaults to the saved three-seed, 125-epoch experiment.

The original MLP and ViT runs record test loss every epoch, which lets us ask both how
well the model finishes and how low its test loss ever gets. The second question
uses the test set to choose an epoch; there are no separate validation histories
in these files. That is why the paper distinguishes these results from the
comparisons using validation loss above.

These multi-profile archives remain the earlier measurements. The new
ReLU and both-block ViT pairs live in [seed_extension.npz](seed_extension.npz);
the main confidence tables combine the two cohorts, while earlier plots
retain their original counts.

Run `python scripts/plot_results.py` from the repository root for additional
learning curves in `runs/figures/exploratory/`. Notebook reruns write into `runs/`.

Run `python scripts/plot_sliding_block.py` to regenerate the block-position
figure in `runs/figures/experiments/mlp/`. Its JSON contains the recorded
recipe, split hashes and 20 per-seed final measurements. All four positions use
dropout 0.2 in three of six layers, mean dropout 0.1, and the same learning rate.
The plotted endpoint is epoch 75, with no checkpoint or position selection.
Faint lines connect each seed across positions; the bars are pointwise 95%
Student-t intervals over five seeds on one fixed split. They are not simultaneous
intervals or paired intervals for differences between positions.

Run `python scripts/plot_cifar_correlations.py` to regenerate the correlation
figure. The archive preserves all 150 seed-level curves, measured input-pair
correlations, image-subset indices, checkpoint hashes and source hashes. Each
curve averages 1,024 distinct-image pairs after estimating each pair's normalized
activation moments over 16 independent mask draws. Dropout is active for every
probe, including trained checkpoints. The exporter recomputes the five-seed
Student-t intervals and the ReLU initialization recursion; theory starts from
each measured input correlation before averaging. Fixed-point predictions appear
only in the initialization panel. The second panel uses fixed epoch-35 weights;
minimum-validation-loss checkpoint measurements remain in the archive.

## Benchmark curves

[benchmark_curves.npz](benchmark_curves.npz) contains the exact training and
validation arrays originally shown in thirteen multi-panel figures: 350 runs
from twelve earlier benchmark cohorts, plus the 30-run profile-geometry pilot.
It preserves run IDs, per-run configurations, seed order, and the source archive
hashes. The confirmation records came from the September 15 W&B export;
excluded Amazon Reviews and zero-weight-decay FI-2010 Transformer cohorts are
not included. The separate FI-2010 Transformer linear follow-up is retained.

Each profile stores one row per seed and one column per zero-based epoch.
Accuracies are fractions in the archive and percentages in the figures.
Shaded bands are the sample standard deviation divided by the square root of
the seed count. Only training loss uses a logarithmic axis. These curves remain
the earlier five- or ten-seed cohorts (three seeds for the pilot); they do
not pool in the later extension or substitute validation loss for test loss.

Run `python scripts/plot_benchmarks.py` to reproduce the paper's figures in
`runs/figures/experiments/benchmarks/`, grouping the MLP and Transformer curves by dataset.
Use `--individual` for separate cohort figures. The exporter uses the same
schedule colors as the presentation and the other paper figures.

The [FI-2010 linear follow-up](supplementary/fi2010-linear.md) retains its
protocol, endpoint table, and learning curves here. It lacks a matching uniform
baseline, so the paper gives only a short summary of that comparison.

## Mean-field calculations

| File | Contents |
|---|---|
| [mean_field.npz](mean_field.npz) | Numerical curves and fit masks |
| [mean_field.json](mean_field.json) | Exponents, fit errors, fitting windows, and source hash |
| [scaling_collapse.npz](scaling_collapse.npz) | Smooth and kinked fixed points used in the collapse plots |
| [scaling_collapse.json](scaling_collapse.json) | Array columns, source hash, and parameter conventions |
| [hermite_coefficients.json](hermite_coefficients.json) | Exact ReLU coefficients and tanh quadrature values |

Run `python scripts/plot_mean_field.py` to recompute these files and figures
in `runs/figures/theory/`. The saved copies in `results/` are kept unchanged.
Here the curves are deterministic, and a small regression error only means
that the chosen points sit close to a power law. Quadrature error, finite
iteration counts, and distance from the asymptotic limit can still shift the
fitted exponent.

The tunable ReLU channel probes the local normal form with independently varied
parameters; those parameters need not describe a realizable finite-variance
network initialization. The paper states this restriction alongside the fits.

The ReLU collapse fixes `rho=1/(1+h0)` along each curve. Legends give `h0`,
the field at `chi=1`; rescaling uses the actual field `h=chi*h0` and the
critical coefficient `kappa=2*sqrt(2)/(3*pi)`, as in the paper. Variation of
the full map's coefficient `chi*kappa` contributes to the visible departures
from the leading equation of state. The 70-point sweep and display window
`-1.25 <= -u <= 2`, `0 <= m/(h/kappa)**(2/3) <= 2` reproduce the original
ReLU figure; the archive retains all points outside that window too.
The smooth scan fixes dropout probability and recomputes its field and
curvature at each point. Both plots retain finite-field departures; the
ReLU horizontal axis runs opposite to the paper's `u`.
