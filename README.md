# Paper Data

Research notebooks and supporting materials for overfitting, dropout scheduling, and mean-field analyses.

## Repository layout
- `MLP overfitting/` — Main MLP overfitting notebook, figures, and serialized experiment results.
- `MLP overfitting big step/` — Additional MLP experiment artifacts.
- `Mean field theory recursions/` — Mean-field recursion notebook and exported scaling figures.
- `Transformer CIFAR100 overfitting/` — ViT dropout scheduling experiments on CIFAR-100.
- `Transformer ablations/` — ViT ablation experiments comparing dropout location and schedule.
- `icml2026_submission.pdf` — Submission draft.

## Primary notebooks
- `Mean field theory recursions/full_meanfield_dropout_criticality_clean_v5.ipynb`
- `MLP overfitting/critical_dropout_scheduling_overfit.ipynb`
- `Transformer ablations/vit_dropout_ablation.ipynb`
- `Transformer CIFAR100 overfitting/vit_dropout_scheduling_cifar100_final.ipynb`

## Reproducibility

### Option 1: `venv` + `requirements.txt`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name paperdata --display-name "paperdata"
jupyter lab
```

### Option 2: Conda / Mamba
```bash
conda env create -f environment.yml
conda activate paperdata
jupyter lab
```

### Suggested execution order
1. `Mean field theory recursions/full_meanfield_dropout_criticality_clean_v5.ipynb`
2. `MLP overfitting/critical_dropout_scheduling_overfit.ipynb`
3. `Transformer ablations/vit_dropout_ablation.ipynb`
4. `Transformer CIFAR100 overfitting/vit_dropout_scheduling_cifar100_final.ipynb`

### Expected artifacts
- `MLP overfitting/critical_dropout_scheduling_overfit.ipynb` writes `dropout_experiment_results.pkl`, `dropout_experiment_results.json`, and the MLP figure exports in `MLP overfitting/`.
- `Transformer ablations/vit_dropout_ablation.ipynb` writes `ablation_results.json` and the exported ablation figures in `Transformer ablations/`.
- `Transformer CIFAR100 overfitting/vit_dropout_scheduling_cifar100_final.ipynb` writes `vit_dropout_results.json` and the exported transformer figures in `Transformer CIFAR100 overfitting/`.

## Runtime notes
- The mean-field notebook is CPU-friendly; the transformer notebooks benefit substantially from a CUDA-capable GPU.
- The transformer notebooks are the slowest runs in the repo and are the ones most likely to vary by hardware and backend.
- The transformer notebooks download CIFAR-10 or CIFAR-100 into `./data/` on first run.
- Notebook outputs are stripped to keep diffs clean.
- `Transformer ablations/vit_dropout_ablation.ipynb` defaults to `WANDB_MODE=disabled` so it runs locally without authentication.
- To enable online Weights & Biases logging for that notebook, run Jupyter with `WANDB_MODE=online`.
- `Transformer CIFAR100 overfitting/vit_dropout_scheduling_cifar100_final.ipynb` saves `vit_dropout_results.json` locally and only triggers a browser download when run inside Google Colab.

## Portability note
- The tracked `.pkl` result files are Python-version and NumPy-version sensitive. Prefer the JSON outputs when possible for sharing or downstream analysis.
- The MLP notebook now writes both pickle and JSON so plotting can stay convenient locally while downstream analysis can use the portable JSON artifact.
