# Paired-seed confidence intervals

Nominal 95% paired Fieller confidence sets for percentage loss reduction (ratio of means), and paired Student-t intervals for mean accuracy gain in percentage points; degrees of freedom n-1. Independent seed pairs and approximately normal across-seed outcomes are assumed. Intervals condition on the fixed data split and selected profiles; they exclude schedule/hyperparameter/epoch-selection uncertainty and are not adjusted for multiple comparisons.

- **original final:** Last recorded epoch of training.
- **original minimum:** Minimum recorded test loss separately for each seed and profile; this retrospective endpoint uses the test set to choose an epoch.
- **benchmark checkpoint:** Test evaluation at each run's minimum-validation-loss checkpoint.
- **benchmark final:** Last training epoch. Endpoint-specific seeds identify the available pairs: five fresh pairs where historical finals were not saved, ten for Tiny ImageNet 80k, and no final values for extended Jannis.

Positive values favor frontloading. Loss changes are percentages; accuracy changes are percentage points.
Seed counts refer to the displayed endpoints. If counts differ within a table, they follow the endpoint column order.

## Original paper

| Experiment                         |   n | Profile           |   Final CE reduction % |   Minimum test CE reduction % |   Final accuracy gain pp |
|:-----------------------------------|----:|:------------------|-----------------------:|------------------------------:|-------------------------:|
| CIFAR-10 / MLP schedules           |  25 | Step early        |  +17.86 [17.31, 18.41] |            +1.42 [1.19, 1.65] |       +0.83 [0.55, 1.11] |
| CIFAR-10 / MLP budget controls     |  25 | Big step          |  +22.61 [22.06, 23.15] |            +2.07 [1.80, 2.33] |       +1.08 [0.85, 1.31] |
| CIFAR-10 / ReLU p=0.1              |  10 | Big step          |  +34.90 [34.36, 35.44] |            +2.54 [2.03, 3.06] |       +1.56 [1.22, 1.89] |
| CIFAR-10 / GELU p=0.1              |  10 | Big step          |  +29.79 [28.96, 30.61] |            +2.45 [1.90, 3.00] |       +0.62 [0.16, 1.08] |
| CIFAR-100 / ViT                    |  10 | Linear decreasing |     +4.15 [3.40, 4.90] |            +1.39 [0.56, 2.21] |       +0.66 [0.36, 0.96] |
| CIFAR-10 / ViT both-block ablation |  10 | Step early        |     +7.40 [5.35, 9.41] |            +1.51 [0.33, 2.68] |       +0.68 [0.24, 1.12] |

## Validation-selected test checkpoints

| Experiment                               |   n | Profile           |   Checkpoint CE reduction % |   Checkpoint accuracy gain pp |
|:-----------------------------------------|----:|:------------------|----------------------------:|------------------------------:|
| Speech Commands / MLP (N=20,020)         |  10 | Step early        |       +12.56 [11.36, 13.73] |            +1.72 [1.28, 2.16] |
| Speech Commands / Transformer (N=20,020) |  10 | Big step          |         +9.04 [2.52, 15.24] |            +2.06 [0.72, 3.41] |
| Jannis / MLP (N=3,840)                   |  10 | Step early        |          +2.60 [1.44, 3.74] |            +1.56 [0.94, 2.19] |
| Jannis / Transformer (N=3,840)           |  10 | Big step          |         +0.15 [-0.62, 0.92] |           +0.32 [-0.51, 1.15] |
| Jannis extended / Transformer (N=3,840)  |  10 | Step early        |         -0.27 [-1.16, 0.61] |           +0.39 [-0.31, 1.10] |
| Tiny ImageNet / MLP (N=20,000)           |  10 | Step early        |          +2.04 [1.72, 2.37] |            +1.28 [0.99, 1.57] |
| Tiny ImageNet / Transformer (N=20,000)   |  10 | Step early        |          +2.21 [1.69, 2.73] |            +1.49 [1.06, 1.92] |
| Tiny ImageNet / MLP (N=80,000)           |  10 | Step early        |          +1.17 [0.92, 1.42] |            +0.77 [0.50, 1.05] |
| Tiny ImageNet / Transformer (N=80,000)   |  10 | Linear decreasing |         +0.76 [-0.24, 1.76] |           +0.13 [-0.41, 0.67] |
| FI-2010 / MLP (N=40,000)                 |  10 | Big step          |          +4.25 [2.57, 5.88] |            +0.81 [0.38, 1.25] |

## Recorded final-epoch test results

| Experiment                               |   n | Profile           |   Final CE reduction % |   Final accuracy gain pp |
|:-----------------------------------------|----:|:------------------|-----------------------:|-------------------------:|
| Speech Commands / MLP (N=20,020)         |   5 | Step early        |   +12.59 [9.95, 15.13] |       +1.71 [0.78, 2.64] |
| Speech Commands / Transformer (N=20,020) |   5 | Big step          |   +15.51 [6.44, 23.04] |      +1.10 [-0.10, 2.30] |
| Jannis / MLP (N=3,840)                   |   5 | Step early        |  +16.23 [14.83, 17.60] |       +3.03 [1.96, 4.11] |
| Jannis / Transformer (N=3,840)           |   5 | Big step          |   +13.02 [8.96, 16.81] |      -0.19 [-1.36, 0.98] |
| Tiny ImageNet / MLP (N=20,000)           |   5 | Step early        |  +21.48 [19.59, 23.35] |       +0.80 [0.47, 1.14] |
| Tiny ImageNet / Transformer (N=20,000)   |   5 | Step early        |  +21.06 [20.27, 21.83] |      +0.73 [-0.19, 1.66] |
| Tiny ImageNet / MLP (N=80,000)           |  10 | Step early        |     +8.65 [8.26, 9.04] |       +0.68 [0.48, 0.88] |
| Tiny ImageNet / Transformer (N=80,000)   |  10 | Linear decreasing |     +1.66 [0.51, 2.81] |      -0.00 [-0.41, 0.40] |
| FI-2010 / MLP (N=40,000)                 |   5 | Big step          |  +68.23 [67.30, 69.12] |     -2.48 [-2.86, -2.09] |

Extended Jannis uses weight decay 1e-7; the other benchmark rows use zero weight decay.

[Full-precision CSV](confidence_intervals.csv) · [Structured results](confidence_intervals.json)
