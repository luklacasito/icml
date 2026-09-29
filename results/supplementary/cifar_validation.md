# CIFAR followups with validation-based selection

Profiles were fixed before tuning: big step for the MLP and linear decreasing for the ViT, both at mean dropout 0.1. Tuned recipes select learning rate and weight decay separately for each profile using validation loss. Confirmation uses fresh training seeds on the recorded split; the test datasets had appeared in earlier experiments.

Positive = improvement over uniform. CE reductions are percentages of the uniform mean; accuracy gains are percentage points. Brackets show nominal paired 95% intervals (Fieller for CE, Student-t for accuracy), conditional on the split and selected settings.

| Experiment | Recipe | n | Best-epoch loss reduction % | Best-epoch accuracy gain pp | Final loss reduction % | Final accuracy gain pp |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 MLP | Common recipe | 10 | +2.12 [1.70, 2.54] | +1.27 [0.86, 1.69] | +20.47 [19.63, 21.29] | +1.53 [0.96, 2.11] |
| CIFAR-10 MLP | Validation-tuned | 10 | +2.15 [1.27, 3.03] | +1.59 [0.73, 2.46] | +32.51 [31.56, 33.43] | +1.19 [0.76, 1.63] |
| CIFAR-100 ViT | Common recipe | 3 | -0.02 [-0.54, 0.50] | +0.01 [-2.45, 2.48] | +1.43 [-0.30, 3.09] | +0.19 [-1.23, 1.62] |
| CIFAR-100 ViT | Validation-tuned | 3 | -0.19 [-0.56, 0.18] | -0.39 [-2.01, 1.23] | +1.42 [0.20, 2.59] | +0.25 [-1.50, 2.01] |

Best epoch means the first epoch with the lowest validation loss; final means epoch 75. Both columns evaluate test data. These cohorts have no per-epoch test histories. The ViT is a 1,600-training-example pilot. Common and tuned recipes share seed IDs and data, so they are not independent replications.

[Measurements and protocol](cifar_validation_evidence.json) · [Full-precision intervals](cifar_validation_intervals.json)
