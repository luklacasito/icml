# Paired-seed confidence intervals

Nominal 95% paired Fieller confidence sets for percentage loss reduction (ratio of means), and paired Student-t intervals for mean accuracy gain in percentage points; degrees of freedom n-1. Independent seed pairs and approximately normal across-seed outcomes are assumed. Intervals condition on the fixed data split and selected profiles; they exclude schedule/hyperparameter/epoch-selection uncertainty and are not adjusted for multiple comparisons.

- **original final:** Last recorded epoch of training.
- **original minimum:** Minimum recorded test loss separately for each seed and profile; this retrospective endpoint uses the test set to choose an epoch.
- **benchmark checkpoint:** Test evaluation at each run's minimum-validation-loss checkpoint.
- **benchmark final:** Last training epoch; absent where not recorded for all pairs.

Positive values favor frontloading. Loss changes are percentages; accuracy changes are percentage points.

## Original paper

| Experiment | n | Profile | Final CE reduction % | Minimum test CE reduction % | Final accuracy gain pp |
|---|---:|---|---:|---:|---:|
| CIFAR-10 / MLP schedules | 25 | Step early | +17.86 [17.31, 18.41] | +1.42 [1.19, 1.65] | +0.83 [0.55, 1.11] |
| CIFAR-10 / MLP budget controls | 25 | Big step | +22.61 [22.06, 23.15] | +2.07 [1.80, 2.33] | +1.08 [0.85, 1.31] |
| CIFAR-10 / ReLU p=0.1 | 3 | Big step | +35.43 [33.64, 37.27] | +2.41 [-0.40, 5.21] | +2.04 [0.79, 3.29] |
| CIFAR-10 / GELU p=0.1 | 10 | Big step | +29.79 [28.96, 30.61] | +2.45 [1.90, 3.00] | +0.62 [0.16, 1.08] |
| CIFAR-100 / ViT | 10 | Linear decreasing | +4.15 [3.40, 4.90] | +1.39 [0.56, 2.21] | +0.66 [0.36, 0.96] |
| CIFAR-10 / ViT both-block ablation | 5 | Step early | +6.32 [3.23, 9.30] | +0.92 [-0.47, 2.30] | +0.52 [-0.11, 1.16] |

## Validation-selected test checkpoints

| Experiment | n | Profile | Checkpoint CE reduction % | Checkpoint accuracy gain pp |
|---|---:|---|---:|---:|
| Speech Commands / MLP (N=20,020) | 5 | Step early | +12.71 [11.01, 14.36] | +1.83 [1.20, 2.46] |
| Speech Commands / Transformer (N=20,020) | 5 | Big step | +4.91 [-9.25, 18.60] | +1.53 [-1.30, 4.36] |
| Jannis / MLP (N=3,840) | 5 | Step early | +2.92 [0.49, 5.27] | +1.29 [0.42, 2.16] |
| Jannis / Transformer (N=3,840) | 5 | Big step | +0.15 [-1.24, 1.52] | +0.26 [-1.58, 2.11] |
| Jannis extended / Transformer (N=3,840) | 10 | Step early | -0.27 [-1.16, 0.61] | +0.39 [-0.31, 1.10] |
| Tiny ImageNet / MLP (N=20,000) | 5 | Step early | +1.94 [1.33, 2.55] | +1.40 [1.28, 1.51] |
| Tiny ImageNet / Transformer (N=20,000) | 5 | Step early | +2.17 [1.28, 3.05] | +1.63 [0.92, 2.34] |
| Tiny ImageNet / MLP (N=80,000) | 5 | Step early | +1.26 [0.93, 1.59] | +0.86 [0.29, 1.44] |
| Tiny ImageNet / Transformer (N=80,000) | 5 | Linear decreasing | +1.01 [-0.34, 2.35] | +0.29 [-0.50, 1.08] |
| FI-2010 / MLP (N=40,000) | 5 | Big step | +3.57 [2.41, 4.73] | +0.67 [-0.29, 1.63] |

## Recorded final-epoch test results

| Experiment | n | Profile | Final CE reduction % | Final accuracy gain pp |
|---|---:|---|---:|---:|
| Tiny ImageNet / MLP (N=80,000) | 5 | Step early | +8.84 [8.38, 9.29] | +0.79 [0.42, 1.16] |
| Tiny ImageNet / Transformer (N=80,000) | 5 | Linear decreasing | +1.68 [-1.26, 4.54] | +0.11 [-0.88, 1.09] |

Extended Jannis uses weight decay 1e-7; the other benchmark rows use zero weight decay.

[Full-precision CSV](confidence_intervals.csv) · [Structured results](confidence_intervals.json)
