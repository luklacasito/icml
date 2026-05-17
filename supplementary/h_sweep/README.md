# h-Sweep Supplement

This folder contains the materials for the dropout-scheduling `\bar{h}` sweep experiment.

## Files

- `MLP_h_sweep.ipynb`: main notebook for the robustness sweep over the mean dropout field `\bar{h}`.
- `h_bar_sweep_results.pkl`: saved sweep results used by the notebook.
- `schedule_comparison.png`: summary comparison figure across schedules.
- `delta_schedules.png`: pairwise performance-delta figure across schedules.
- `curves_sweep.png`: training-curve figure for the sweep.

## Experiment Summary

The notebook studies whether front-loaded dropout schedules outperform constant dropout across a range of dropout budgets. It compares:

- no dropout
- constant dropout
- early step dropout
- larger early-step dropout

The setup uses a fixed-depth ReLU MLP near critical initialization and evaluates performance across multiple `\bar{h}` values.


