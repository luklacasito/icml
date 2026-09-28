"""Check mean-field figures against exact maps and the reference scan."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermitenorm

from scripts.plot_mean_field import hermite_coefficients
from utils.scaling import (
    RELU_KAPPA,
    kinked_scaling,
    kinked_universal,
    relu_fixed_rho_scan,
    relu_order_parameter,
)


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


@pytest.mark.parametrize("chi", [0.7, 0.95, 1.0, 1.3])
@pytest.mark.parametrize("h", [1e-4, 1e-2, 1e-1])
def test_fixed_field_relu_points_solve_the_exact_map(chi, h):
    m = relu_order_parameter(chi, h)
    c = 1 - m
    rho = chi / (chi + h)
    kernel = (np.sqrt(1 - c * c) + (np.pi - np.arccos(c)) * c) / np.pi
    assert chi * kernel + 1 - chi / rho == pytest.approx(c, abs=2e-13)
    assert 1 - (chi + 1 - chi / rho) == pytest.approx(h, abs=5e-16)
    assert 0 < chi * (1 - np.arccos(c) / np.pi) < 1


def test_kinked_collapse_converges_to_the_equation_of_state():
    # Hold the critical scaling coordinate fixed as h decreases. Finite-t
    # corrections vanish; an exact collapse is not expected at finite field.
    x = np.linspace(-1.25, 2, 14)
    theory = kinked_universal(x)
    errors = []
    for h in [1e-3, 1e-5, 1e-7]:
        t = -x * RELU_KAPPA ** (2 / 3) * h ** (1 / 3)
        m = np.array([relu_order_parameter(1 + ti, h) for ti in t])
        actual_x, scaled_m = kinked_scaling(t, h, m)
        np.testing.assert_allclose(actual_x, x, atol=1e-9)
        errors.append(np.max(np.abs(scaled_m / theory - 1)))
    assert 0.05 < errors[0] < 0.1  # The original figure shows these departures.
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 3
    assert errors[2] < 0.004


def test_kinked_scan_preserves_the_original_figure():
    # Independent values from the original arc-cosine-map figure generator.
    # Catch changes to the fixed parameter or to the critical normalization.
    rows = relu_fixed_rho_scan([0.1], [-0.3, 0.3])
    np.testing.assert_allclose(rows[:, 2], [0.07, 0.13], atol=1e-15)
    np.testing.assert_allclose(rows[:, 3], [0.1795603267404471, 1.045076230424892], atol=1e-11)
    x, y = kinked_scaling(rows[:, 1], rows[:, 2], rows[:, 3])
    np.testing.assert_allclose(x, [1.623950038999491, -1.321166520908098], atol=1e-11)
    np.testing.assert_allclose(y, [0.4738718796627341, 1.825443151742836], atol=1e-11)


def test_kinked_universal_positive_branch_and_axis_sign():
    x = np.linspace(-1.25, 2, 100)
    y = kinked_universal(x)
    np.testing.assert_allclose(y**1.5 + x * y, 1, atol=1e-11)
    assert np.all(np.diff(y) < 0)
    assert kinked_universal(0) == pytest.approx(1)
    assert kinked_universal(-1.25) > 2  # The displayed window is a zoom.
