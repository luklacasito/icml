# Dropout Universality

**[Read the paper](paper.pdf)** · [LaTeX source](manuscript/) · [Results and confidence intervals](results/confidence_intervals.md)

*Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos*

Lucas Fernandez Sarmiento · ICML 2026

Where should we apply dropout across a network's depth? This paper develops a
mean-field theory near the edge of chaos, with different scaling laws for smooth
and kinked activations. The propagation approximation and a heuristic for how
far each mask affects the network motivate concentrating dropout near the input.
Experiments on vision, speech, and financial time series find gains most
consistently in MLPs, with mixed results in transformers. The gains also depend
on when we measure them: at the final epoch, at the lowest recorded test loss,
or at the epoch chosen by validation loss.

The PDF is the revised **September 29, 2026** paper. It includes 94 additional
fits with the selected schedules held fixed. The original headline comparisons
now have at least ten seed pairs; some final-epoch comparisons use only five
new pairs because earlier runs did not save that measurement. The
[tables](results/confidence_intervals.md) give the counts and confidence intervals.
Separate [CIFAR validation followups](results/cifar_validation.md) report ten MLP
seed pairs and a three-pair ViT pilot, with profiles fixed before tuning.
The original [camera-ready version is on arXiv](https://arxiv.org/pdf/2605.21648v2).

## Repository guide

| Path | Purpose |
|---|---|
| [paper.pdf](paper.pdf), [manuscript/](manuscript/) | Paper and LaTeX source |
| [notebooks/](notebooks/) | Original calculations, experiments, and plots |
| [benchmarks/](benchmarks/README.md) | Command-line runners for additional datasets and seed extensions |
| [utils/](utils/) | Shared notebook schedules, training, and result loading |
| [scripts/](scripts/) | Rebuild figures and tables; run numerical calculations |
| [results/](results/README.md) | Archived measurements and their limitations |
| [tests/](tests/) | Numerical, checkpoint, and provenance checks |

New runs go into `runs/`; archived measurements stay in `results/`.

## Calculations and experiments

| Notebook | Question |
|---|---|
| [Mean field](notebooks/mean_field.ipynb) | How does dropout change critical scaling for smooth and kinked activations? |
| [MLP schedules](notebooks/mlp_schedules.ipynb) | Where should we put dropout across depth on CIFAR-10? |
| [Dropout-budget sweep](notebooks/mlp_dropout_sweep.ipynb) | How does the comparison change as we increase dropout? |
| [Width sweep](notebooks/mlp_width_sweep.ipynb) | What survives as the MLP width goes from 64 to 1024? |
| [GELU sweep](notebooks/mlp_gelu.ipynb) | Does the scheduling idea carry over to a smooth activation? |
| [ViT schedules](notebooks/vit_schedules.ipynb) | How do depth profiles compare in a CIFAR-100 transformer? |
| [ViT ablations](notebooks/vit_ablation.ipynb) | Should dropout act on attention, the MLP branch, or both? |

The MLP budget controls compare early concentration with simply increasing
uniform dropout; their [saved runs](results/mlp_budget_controls.npz) and
[figure](manuscript/figures/experiments/mlp/dropout_budget_comparison.pdf) are
included too. The additional datasets are in the paper's experimental appendix,
with paired test measurements in [results](results/).

## Reproduce the results

Use Python 3.11 and run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The mean-field calculations run on CPU; a CUDA GPU is preferable for training.
Each notebook explains which cells to run for the saved results and which train
new models. CIFAR downloads go into `data/`, reruns into `runs/`, and W&B logging
is off by default.

### Figures and tables

Run these from the repository root; no model training is needed:

```bash
python scripts/plot_paper_results.py    # CIFAR curves and dropout sweeps
python scripts/plot_sliding_block.py    # Dropout-block position experiment
python scripts/plot_cifar_correlations.py  # Measured correlations and initialization theory
python scripts/plot_benchmarks.py       # Additional-dataset learning curves
python scripts/plot_mean_field.py       # Critical exponents, collapse, and Hermite figures
python scripts/confidence_intervals.py  # Main CIFAR and benchmark tables
python scripts/validation_cifar_results.py  # Separate CIFAR validation followups
```

The first three commands update `manuscript/figures/experiments/`. The remaining
commands write to `runs/figures/benchmarks/`, `runs/mean_field/`, and
`runs/confidence/`. For additional saved learning curves, use
`python scripts/plot_results.py --sweeps`; its output goes to `runs/figures/`.
Figure styling is shared through `utils/plot_style.py`. The paper's figures are
included in [manuscript/figures/](manuscript/figures/), and the
[source guide](manuscript/README.md) explains how to build the PDF.

### Training and data

The [benchmark guide](benchmarks/README.md) covers Speech Commands,
Tiny ImageNet, and FI-2010. It includes data preparation, the retained models,
and [configurations and split identifiers](benchmarks/protocols.json)
for the reported profiles. Training saves the model chosen by validation loss
and the final model separately. The [CIFAR extension guide](benchmarks/original/README.md)
covers the additional ReLU and ViT seeds.

The [standalone financial MLP guide](benchmarks/VANILLA.md) reproduces its
separate data pipeline and learning-rate search. Its originally selected
learning rates were not preserved; the guide distinguishes a new search from
reproducing the archived results.

## Reading the comparisons

I report loss reductions relative to uniform dropout and accuracy gains in
percentage points, keeping the same schedule across the endpoints in each row.
The [results table](results/confidence_intervals.md) gives paired 95% intervals:
Fieller intervals for relative loss reductions and Student-t intervals for
accuracy gains. They describe variation across seeds on the recorded split,
conditional on the selected schedules and hyperparameters. They do not account
for earlier selection or adjust for multiple comparisons.

The [seed-extension guide](results/seed_extension.md) records which comparisons
received new seeds and links their saved learning curves and exact settings.
Multi-profile figures retain their original run counts; the main
tables combine those measurements with the fixed-profile extensions.

The endpoint matters. Taking the lowest recorded test loss uses the test set to
choose an epoch. In the validation-based comparisons, the model is chosen at
the first epoch with the lowest validation loss, before its test evaluation.
The paper keeps these comparisons separate. The
[CIFAR-100 audit](results/cifar100_configuration.json) identifies the full dataset
sizes from the accuracy increments under the saved evaluator, while distinguishing
the notebook's reported settings from the configuration missing from the runs.

Speech Commands uses randomly held-out clips, with speakers shared across
training and evaluation; the [speaker audit](results/speech_speaker_audit.json)
and [held-out-speaker guide](benchmarks/SPEECH.md) explain that limitation.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
ruff format --check .
```

When editing code, run `ruff format .` before these checks. Source hashes identify
the exact version of a run, so resume existing runs with their original checkout.

[CITATION.cff](CITATION.cff) contains the citation details. Code is under the
[MIT license](LICENSE).
