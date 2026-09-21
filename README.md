# Dropout Universality

**[Read the camera-ready paper](paper.pdf)** · [LaTeX source](manuscript/)

*Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos*

Lucas Fernandez Sarmiento · ICML 2026

I study dropout as a perturbation of critical signal propagation, then use the
mean-field picture to ask a concrete question: with a fixed dropout budget,
where should we spend it across depth?

This repo contains the camera-ready paper and its experiments, including the
dropout-budget, width, and smooth-activation sweeps. The notebooks contain the
calculations and training code; saved results are included so you can inspect
the comparisons without rerunning them.

## Experiments

| Notebook | What it does |
|---|---|
| [Mean field](notebooks/mean_field.ipynb) | Correlation recursions, critical exponents, and scaling collapse |
| [MLP schedules](notebooks/mlp_schedules.ipynb) | Dropout placement across depth on CIFAR-10 |
| [Dropout-budget sweep](notebooks/mlp_dropout_sweep.ipynb) | ReLU MLP schedules across mean dropout strengths |
| [Width sweep](notebooks/mlp_width_sweep.ipynb) | ReLU MLP schedules from width 64 to 1024 |
| [GELU sweep](notebooks/mlp_gelu.ipynb) | The same dropout-budget comparison with a smooth activation |
| [ViT schedules](notebooks/vit_schedules.ipynb) | Residual-dropout schedules on CIFAR-100 |
| [ViT ablations](notebooks/vit_ablation.ipynb) | Attention, MLP, and both-block dropout on CIFAR-10 |

The additional MLP budget-control comparison is preserved in
[saved results](results/) and [its figure](manuscript/figures/experiments/mlp/dropout_budget_comparison.pdf).
Shared schedules, training loops, and result loading live in [utils](utils/).

## Run

Use Python 3.11, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The mean-field calculation runs on CPU. Training the networks is better suited
to a CUDA GPU. CIFAR downloads go into `data/`; new results and plots go into
`runs/`, leaving the saved paper results intact. W&B logging is disabled by default.

To plot the saved learning curves without training or downloading data:

```bash
python scripts/plot_results.py
```

Plots are written to `runs/figures/`. Add `--sweeps` to include the learning curves
at every saved dropout strength and width. The paper's figure exports are in
[manuscript/figures](manuscript/figures/).

The PDF is the unchanged May 29 camera-ready release. Its historical code link
now points to a private archive; the experiments are maintained here.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See [CITATION.cff](CITATION.cff) for citation details. Code is under the [MIT license](LICENSE).
