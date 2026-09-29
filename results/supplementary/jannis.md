# Jannis

Nominal 95% paired Fieller confidence sets for percentage loss reduction (ratio of means), and paired Student-t intervals for mean accuracy gain in percentage points; degrees of freedom n-1. Independent seed pairs and approximately normal across-seed outcomes are assumed. Intervals condition on the fixed data split and selected profiles; they exclude schedule/hyperparameter/epoch-selection uncertainty and are not adjusted for multiple comparisons.

- **Benchmark best:** Test evaluation at the epoch with the lowest validation loss.
- **Benchmark final:** Last training epoch. Each column gives the number of available seed pairs.

Positive values favor frontloading. Loss changes are percentages; accuracy changes are percentage points.
Seed counts refer to the displayed endpoints. If counts differ within a table, they follow the endpoint column order.

## Test results at the best validation epoch

| Experiment                              |   n | Profile    |    Loss reduction % |    Accuracy gain pp |
|:----------------------------------------|----:|:-----------|--------------------:|--------------------:|
| Jannis / MLP (N=3,840)                  |  10 | Step early |  +2.60 [1.44, 3.74] |  +1.56 [0.94, 2.19] |
| Jannis / Transformer (N=3,840)          |  10 | Big step   | +0.15 [-0.62, 0.92] | +0.32 [-0.51, 1.15] |
| Jannis extended / Transformer (N=3,840) |  10 | Step early | -0.27 [-1.16, 0.61] | +0.39 [-0.31, 1.10] |

## Test results at the final epoch

| Experiment                     |   n | Profile    |   Final CE reduction % |   Final accuracy gain pp |
|:-------------------------------|----:|:-----------|-----------------------:|-------------------------:|
| Jannis / MLP (N=3,840)         |   5 | Step early |  +16.23 [14.83, 17.60] |       +3.03 [1.96, 4.11] |
| Jannis / Transformer (N=3,840) |   5 | Big step   |   +13.02 [8.96, 16.81] |      -0.19 [-1.36, 0.98] |

Extended Jannis uses weight decay 1e-7; the other Jannis rows use zero.

[Full-precision CSV](../confidence_intervals.csv) · [Structured results](../confidence_intervals.json)
