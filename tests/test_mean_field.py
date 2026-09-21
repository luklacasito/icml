"""Check plotted Hermite coefficients against independent closed expressions."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermitenorm

from scripts.plot_mean_field import hermite_coefficients


def test_relu_coefficients_match_closed_expressions():
    relu, _ = hermite_coefficients()
    expected = {
        0: 1 / math.sqrt(2 * math.pi),
        1: 0.5,
        2: 1 / (2 * math.sqrt(math.pi)),
        4: -1 / math.sqrt(48 * math.pi),
        36: -math.prod(range(1, 34, 2)) / math.sqrt(2 * math.pi * math.factorial(36)),
    }
    for degree, value in expected.items():
        assert relu[degree] == pytest.approx(value, rel=1e-14)
    np.testing.assert_array_equal(relu[3::2], 0)


def test_tanh_coefficients_match_adaptive_integration():
    _, tanh = hermite_coefficients()
    np.testing.assert_array_equal(tanh[::2], 0)
    for degree in (1, 3, 11):
        coefficient, _ = quad(
            lambda z: (
                np.tanh(z)
                * eval_hermitenorm(degree, z)
                * np.exp(-z * z / 2)
                / math.sqrt(2 * math.pi * math.factorial(degree))
            ),
            -12,
            12,
            epsabs=1e-12,
        )
        assert tanh[degree] == pytest.approx(coefficient, abs=1e-11)
