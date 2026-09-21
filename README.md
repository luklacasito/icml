# Dropout Universality

**[Read the submitted paper](paper.pdf)**

*Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos*

Lucas Fernandez Sarmiento · ICML 2026 submission

I study dropout as a perturbation of critical signal propagation, then use the
mean-field picture to ask a concrete question: with a fixed dropout budget,
where should we spend it across depth?

This repo contains the original submission and its experiments. The PDF is the
unchanged, anonymized submission. The notebooks contain the calculations and
training code; saved results and figures are included so you can inspect the
comparisons without rerunning them.

## Experiments

| Notebook | What it does |
|---|---|
| [Mean field](notebooks/mean_field.ipynb) | Correlation recursions, critical exponents, and scaling collapse |
| [MLP schedules](notebooks/mlp_schedules.ipynb) | Dropout placement across depth on CIFAR-10 |
| [ViT schedules](notebooks/vit_schedules.ipynb) | Residual-dropout schedules on CIFAR-100 |
| [ViT ablations](notebooks/vit_ablation.ipynb) | Attention, MLP, and both-block dropout on CIFAR-10 |

The additional MLP budget-control comparison is preserved in
[saved results](results/) and [its figure](figures/mlp_budget_controls.png).
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

Plots are written to `runs/figures/`. The original exports are in [figures](figures/).

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See [CITATION.cff](CITATION.cff) for citation details. Code is under the [MIT license](LICENSE).
