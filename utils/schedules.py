"""The paper's layerwise dropout schedules.

Constant, linear, reverse_linear, step, and reverse_step have mean h_bar.
The none, double, and triple controls use zero, twice, and three times that
budget. Big_step applies 3 * h_bar to max(1, depth // 3) early layers; its mean
equals h_bar only when depth is divisible by three.
"""

import math
from numbers import Integral


def get_dropout_schedule(
    sched: str, depth: int, h_bar: float, h_max: float | None = None
) -> list[float]:
    """Return dropout probabilities in input-to-output layer order.

    Linear and step put more dropout in later layers; their reverse variants
    put it earlier. For step schedules, h_max sets the number of active layers
    while their rate is adjusted to preserve the mean. An absent h_max, or one
    below h_bar, uses 2 * h_bar.
    """
    if not isinstance(depth, Integral) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not math.isfinite(h_bar) or not 0 <= h_bar <= 1:
        raise ValueError("h_bar must be a finite dropout probability")

    if sched == "none":
        rates = [0.0] * depth
    elif sched == "constant":
        rates = [h_bar] * depth
    elif sched == "double":
        rates = [2.0 * h_bar] * depth
    elif sched == "triple":
        rates = [3.0 * h_bar] * depth
    elif sched in ("linear", "reverse_linear"):
        if depth == 1:
            rates = [h_bar]
        else:
            rates = [2.0 * h_bar * i / (depth - 1) for i in range(depth)]
        if sched == "reverse_linear":
            rates.reverse()
    elif sched in ("step", "reverse_step"):
        if h_max is not None and (not math.isfinite(h_max) or h_max < 0):
            raise ValueError("h_max must be finite and nonnegative")
        if h_bar == 0:
            rates = [0.0] * depth
        else:
            effective_h_max = (
                h_max if h_max is not None and h_max >= h_bar else 2.0 * h_bar
            )
            n_drop = max(1, math.ceil(h_bar / effective_h_max * depth))
            h_adj = h_bar * depth / n_drop
            rates = [0.0] * (depth - n_drop) + [h_adj] * n_drop
            if sched == "reverse_step":
                rates.reverse()
    elif sched == "big_step":
        # Preserve the original integer rounding for depths not divisible by three.
        n_drop = max(1, depth // 3)
        rates = [3.0 * h_bar] * n_drop + [0.0] * (depth - n_drop)
    else:
        raise ValueError(f"Unknown dropout schedule: {sched!r}")

    if any(rate > 1 for rate in rates):
        raise ValueError(f"{sched} produces dropout probabilities above 1")
    return rates
