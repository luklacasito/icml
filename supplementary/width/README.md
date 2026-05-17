# Width-Sweep Supplement

This folder contains the materials for the dropout-scheduling width-sweep experiment.

## Files

- `MLP_width_sweep.ipynb`: main notebook for the sweep over hidden-layer width at fixed mean dropout field `\bar{h} = 0.1`.
- `width_sweep_results.pkl`: saved sweep results used by the notebook.
- `accuracy_vs_width.png`: test accuracy as a function of network width.
- `accuracy_vs_aspect_ratio.png`: test accuracy as a function of aspect ratio.
- `baseline_advantage_comparison.png`: comparison of schedule advantages relative to the baseline.
- `all_test_curves.png`: test curves across the full width sweep.

## Experiment Summary

The notebook studies how the benefit of front-loaded dropout schedules changes with network capacity. It compares:

- no dropout
- constant dropout
- early step dropout
- larger early-step dropout

The setup uses a fixed-depth ReLU MLP near critical initialization, keeps the mean dropout field fixed at `\bar{h} = 0.1`, and varies the hidden width.

