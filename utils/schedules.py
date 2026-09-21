"""The paper's layerwise dropout schedules.

Constant, linear, reverse_linear, step, and reverse_step have mean h_bar.
The none, double, and triple controls use zero, twice, and three times that
budget. Big_step applies 3 * h_bar to max(1, depth // 3) early layers; its mean
equals h_bar only when depth is divisible by three.
"""

import numpy as np


def get_dropout_schedule(
    sched: str, depth: int, h_bar: float, h_max: float | None = None
) -> list[float]:
    """Return dropout probabilities in input-to-output layer order.

    Linear and step put more dropout in later layers; their reverse variants
    put it earlier. For step schedules, h_max sets the number of active layers
    while their rate is adjusted to preserve the mean. An absent h_max, or one
    below h_bar, uses 2 * h_bar.
    """
    if sched == "none":
        return [0.0] * depth

    if sched == "constant":
        return [h_bar] * depth

    if sched == "double":
        return [2.0 * h_bar] * depth

    if sched == "triple":
        return [3.0 * h_bar] * depth

    if sched == "linear":
        if depth == 1:
            return [h_bar]
        return [2.0 * h_bar * i / (depth - 1) for i in range(depth)]

    if sched == "reverse_linear":
        if depth == 1:
            return [h_bar]
        return [2.0 * h_bar * (depth - 1 - i) / (depth - 1) for i in range(depth)]

    if sched in ("step", "reverse_step"):
        effective_h_max = (
            h_max if (h_max is not None and h_max >= h_bar) else 2.0 * h_bar
        )
        n_drop = max(1, int(np.ceil(h_bar / effective_h_max * depth)))
        h_adj = h_bar * depth / n_drop
        if sched == "step":
            return [0.0] * (depth - n_drop) + [h_adj] * n_drop
        else:
            return [h_adj] * n_drop + [0.0] * (depth - n_drop)

    if sched == "big_step":
        # Preserve the original integer rounding for depths not divisible by three.
        n_drop = max(1, int(depth / 3))
        return [3.0 * h_bar] * n_drop + [0.0] * (depth - n_drop)

    raise ValueError(f"Unknown dropout schedule: {sched!r}")
