"""Dropout schedule factories.

All schedules preserve the mean dropout rate h_bar across layers, so
comparisons between schedule shapes are fair (equal regularisation budget).

Available schedules
-------------------
none           – no dropout
constant       – uniform h_bar every layer
linear         – ramp 0 → 2·h_bar  (early layers lightly regularised)
reverse_linear – ramp 2·h_bar → 0  (early layers heavily regularised)
step           – h_adj concentrated in the last n_drop layers
reverse_step   – h_adj concentrated in the first n_drop layers
double         – uniform 2·h_bar (twice the budget)
triple         – uniform 3·h_bar (triple the budget)
big_step       – 3·h_bar in the first depth//3 layers, zero elsewhere
"""

import numpy as np


def get_dropout_schedule(sched: str, depth: int, h_bar: float,
                         h_max: float | None = None) -> list[float]:
    """Return per-layer dropout probabilities for *depth* layers.

    Parameters
    ----------
    sched   : schedule name (see module docstring)
    depth   : number of layers
    h_bar   : mean dropout rate (regularisation budget)
    h_max   : maximum allowed per-layer rate used by step/reverse_step to
              determine how many layers receive dropout.  Defaults to 2·h_bar.

    Returns
    -------
    list of floats, length == depth
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
        effective_h_max = h_max if (h_max is not None and h_max >= h_bar) else 2.0 * h_bar
        n_drop = max(1, int(np.ceil(h_bar / effective_h_max * depth)))
        h_adj = h_bar * depth / n_drop
        if sched == "step":
            return [0.0] * (depth - n_drop) + [h_adj] * n_drop
        else:
            return [h_adj] * n_drop + [0.0] * (depth - n_drop)

    if sched == "big_step":
        # 3·h_bar in the first third of layers; equivalent total budget = h_bar
        n_drop = max(1, int(depth / 3))
        return [3.0 * h_bar] * n_drop + [0.0] * (depth - n_drop)

    raise ValueError(f"Unknown dropout schedule: {sched!r}")
