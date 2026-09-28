"""Fixed points and scaling coordinates for the formal ReLU channel."""

import numpy as np
from scipy.optimize import brentq

RELU_KAPPA = 2 * np.sqrt(2) / (3 * np.pi)


def relu_kernel_remainder(m):
    """Return K(1-m) - (1-m), resolving the small-m kink without cancellation."""
    if not 0 <= m <= 2:
        raise ValueError("The correlation gap must lie in [0, 2]")
    if m < 1e-3:
        return RELU_KAPPA * m**1.5 * (1 + m / 20 + 9 * m**2 / 1120 + 5 * m**3 / 2688)
    theta = 2 * np.arcsin(np.sqrt(m / 2))
    return (np.sin(theta) - theta * np.cos(theta)) / np.pi


def relu_order_parameter(chi, h):
    """Stable fixed-point gap of F(c)=chi*K(c)+1-chi-h, at a fixed field h.

    For chi > 0, h > 0 and chi+h < 2, the exact map has one stable crossing
    within -1 < c < 1. The equivalent keep probability is chi/(chi+h).
    """
    if not (np.isfinite(chi) and np.isfinite(h) and chi > 0 and h > 0 and chi + h < 2):
        raise ValueError("Require chi > 0, h > 0 and chi + h < 2")
    return brentq(
        lambda m: chi * relu_kernel_remainder(m) - (chi - 1) * m - h,
        0,
        2,
        xtol=1e-15,
        rtol=1e-13,
    )


def relu_fixed_rho_scan(h0_values, t_values):
    """Return (h0, t, h, m) rows at fixed rho=1/(1+h0) along each curve.

    h0 is the field at criticality. The actual field is h=(1+t)*h0.
    """
    return np.array(
        [
            (h0, t, (1 + t) * h0, relu_order_parameter(1 + t, (1 + t) * h0))
            for h0 in h0_values
            for t in t_values
        ]
    )


def kinked_scaling(t, h, m):
    """Rescale with the critical coefficient kappa_0, as in the paper.

    The horizontal variable x=-u makes the curve decrease, as in the smooth
    plot. The leading equation of state is Y**1.5 + x*Y = 1. The full map's
    coefficient (1+t)*kappa_0 contributes finite-t corrections to scaling.
    """
    t, h, m = np.broadcast_arrays(t, h, m)
    if np.any(h <= 0):
        raise ValueError("Scaling requires a positive field")
    return -t / (RELU_KAPPA ** (2 / 3) * h ** (1 / 3)), m / (h / RELU_KAPPA) ** (2 / 3)


def kinked_universal(x):
    """Positive branch Y=y**2 of y**3+x*y**2-1=0, for horizontal x=-u."""
    x = np.asarray(x, dtype=float)
    values = [brentq(lambda y: y**3 + xi * y**2 - 1, 0, max(1, 1 - xi)) ** 2 for xi in x.ravel()]
    return np.asarray(values).reshape(x.shape)
