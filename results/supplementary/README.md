# Additional comparisons

These results are available alongside the paper's main comparisons.

| Experiment | Results and source |
|---|---|
| CIFAR learning-rate and weight-decay tuning | [Results](cifar_validation.md), [seed measurements and settings](cifar_validation_evidence.json), [intervals](cifar_validation_intervals.json), [exact source](cifar_validation_source.tar.gz) |
| Jannis | [Results and intervals](jannis.md). Records remain in `../benchmark_curves.npz`, `../confidence_seed_metrics.json` and `../seed_extension.npz`; configurations remain in `../../benchmarks/protocols.json` |
| FI-2010 Transformer, linear dropout | [Protocol and results](fi2010-linear.md), [learning curves](fi2010-transformer-linear.pdf) |

The CIFAR studies use profiles fixed before tuning and models chosen by
validation loss. They use different cohorts from the paper's CIFAR comparisons.
Their source archive includes a README and a reconstruction helper that checks
the recovered files against the recorded hashes. Recompute the tables with:

```bash
python scripts/validation_cifar_results.py
```

Jannis uses OpenML `jannis`, version 1: 54 features, four classes and a balanced
3,840/960/1,920 training/validation/test split. Prepare it with
`python -m benchmarks.prepare openml_jannis --root /path/to/data --raw /path/to/raw`.

Jannis is omitted from the paper and the default figure export. Its measurements
remain in the shared archives so their hashes and records stay intact. Export
each saved benchmark cohort, including Jannis, with:

```bash
python scripts/plot_benchmarks.py --individual --output-dir runs/figures/supplementary
```

The FI-2010 linear comparison has no matching uniform baseline. It is reported
separately and contributes no improvement-over-uniform estimate.
