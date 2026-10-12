"""Mean-field maps, fixed points, and the paper's exponent scans.

Quadrature grids and recurrence order match the original scientific notebook.
"""

from types import SimpleNamespace

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq

from .scaling import relu_order_parameter, relu_fixed_rho_scan

# Gauss–Hermite quadrature setup
# E_{z~N(0,1)}[f(z)] = sum_i w_i f(sqrt(2) x_i)/sqrt(pi)
GH_N = 60
x, w = hermgauss(GH_N)
z = np.sqrt(2.0) * x
w1 = w / np.sqrt(np.pi)

# For 2D expectations with independent z1,z2 ~ N(0,1):
# E[f(z1,z2)] = sum_{i,j} (w_i w_j / pi) f(sqrt(2)x_i, sqrt(2)x_j)
Z1 = z[:, None]
Z2 = z[None, :]
W2 = (w[:, None] * w[None, :]) / np.pi


def gh_expect_1d(fvals: np.ndarray) -> float:
    """Given f(z_i) on GH nodes z_i, return E[f(z)] for z~N(0,1)."""
    return float(np.sum(w1 * fvals))


def gh_expect_2d(fvals: np.ndarray) -> float:
    """Given f(z_i,z_j) on GH grid, return E[f(z1,z2)] for iid N(0,1)."""
    return float(np.sum(W2 * fvals))


def phi_tanh_p(x):
    """Tanh derivative, sech(x)^2."""
    return 1 / np.cosh(x) ** 2


def solve_qstar_tanh(sigma_w2: float, sigma_b2: float, rho: float) -> float:
    """Solve the stable variance fixed point with Brent's method."""
    if sigma_b2 == 0 and sigma_w2 / rho <= 1:
        return 0.0
    return brentq(
        lambda q: (sigma_w2 / rho) * gh_expect_1d(np.tanh(np.sqrt(q) * z) ** 2) + sigma_b2 - q,
        max(sigma_b2, np.finfo(float).tiny),
        sigma_b2 + sigma_w2 / rho,
        xtol=1e-13,
        rtol=1e-13,
    )


def tanh_chi_g_h(sigma_w2: float, sigma_b2: float, rho: float):
    """Return q*, chi=F'(1), g=F''(1), and h=1-F(1) for tanh."""
    q = solve_qstar_tanh(sigma_w2, sigma_b2, rho)
    u = np.sqrt(q) * z
    f = np.tanh(u)
    fp = phi_tanh_p(u)
    fpp = -2 * f * fp
    chi = sigma_w2 * gh_expect_1d(fp**2)
    g = sigma_w2 * q * gh_expect_1d(fpp**2)
    h = 1 - (sigma_w2 * gh_expect_1d(f**2) + sigma_b2) / q
    return q, chi, g, h


def correlated_inputs(c, q):
    """Gaussian pair with variance q and correlation c, on the quadrature grid."""
    c = np.clip(c, -1 + 1e-15, 1 - 1e-15)
    sq = np.sqrt(q)
    return sq * Z1, sq * (c * Z1 + np.sqrt(1 - c * c) * Z2)


def F_tanh(c: float, sigma_w2: float, sigma_b2: float, rho: float, qstar: float = None):
    """Full tanh map: F(c) = (sigma_w2 E[tanh(u1)tanh(u2)] + sigma_b2) / q*."""
    if qstar is None:
        qstar = solve_qstar_tanh(sigma_w2, sigma_b2, rho)
    u1, u2 = correlated_inputs(c, qstar)
    E = gh_expect_2d(np.tanh(u1) * np.tanh(u2))
    return float((sigma_w2 * E + sigma_b2) / qstar)


def Fp_tanh(c: float, sigma_w2: float, sigma_b2: float, rho: float, qstar: float = None):
    """Price theorem: F'(c) = sigma_w2 E[phi'(u1)phi'(u2)]."""
    if qstar is None:
        qstar = solve_qstar_tanh(sigma_w2, sigma_b2, rho)
    u1, u2 = correlated_inputs(c, qstar)
    return float(sigma_w2 * gh_expect_2d(phi_tanh_p(u1) * phi_tanh_p(u2)))


def cstar_tanh(sigma_w2: float, sigma_b2: float, rho: float) -> float:
    """Stable tanh fixed point, solving the correlation gap with Brent's method."""
    q, chi, _, _ = tanh_chi_g_h(sigma_w2, sigma_b2, rho)
    if rho == 1 and chi <= 1:
        return 1.0
    # Exclude the unstable c=1 root on the no-dropout, chi>1 branch.
    lower = 1e-12 if rho == 1 else 0.0
    m = brentq(
        lambda m: F_tanh(1 - m, sigma_w2, sigma_b2, rho, qstar=q) - (1 - m),
        lower,
        2.0,
        xtol=1e-14,
        rtol=1e-12,
    )
    return 1 - m


def F_relu_base(c: float) -> float:
    """Arc-cosine kernel (order 1) normalized correlation map for ReLU."""
    c = float(np.clip(c, -1.0, 1.0))
    return float((np.sqrt(max(0.0, 1.0 - c * c)) + (np.pi - np.arccos(c)) * c) / np.pi)


def F_relu(c: float, chi: float, rho: float) -> float:
    """F(c)=chi*K(c)+1-chi/rho, with F'(1)=chi and field h=chi*(1/rho-1)."""
    return float(chi * F_relu_base(c) + 1.0 - chi / rho)


def Fp_relu(c: float, chi: float, rho: float) -> float:
    """Derivative of the ReLU correlation map."""
    return float(chi * (1 - np.arccos(np.clip(c, -1, 1)) / np.pi))


def cstar_relu(chi: float, rho: float) -> float:
    """Stable ReLU fixed point, using the shared Brent solver."""
    return 1 - relu_order_parameter(chi, chi * (1 - rho) / rho)


def tune_sigw2_for_chi_tanh(
    target_chi: float,
    sigma_b2: float,
    rho: float,
    *,
    bracket=(0.05, 20.0),
):
    """Tune the weight variance to a target correlation slope."""
    return brentq(
        lambda sw2: tanh_chi_g_h(sw2, sigma_b2, rho)[1] - target_chi,
        *bracket,
        xtol=1e-14,
        rtol=1e-12,
    )


def loglog_fit(x, y, fit_slice=slice(None)):
    """
    Fit log10(y) = a log10(x) + b on x,y>0.
    Returns (a, b, se_a, se_b) where se_a, se_b are standard errors.
    """
    x = np.asarray(x)[fit_slice]
    y = np.asarray(y)[fit_slice]
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    coeffs, cov = np.polyfit(np.log10(x[m]), np.log10(y[m]), 1, cov=True)
    a, b = coeffs
    se_a, se_b = np.sqrt(np.diag(cov))
    return float(a), float(b), float(se_a), float(se_b)


def line_from_logfit(x, a, b):
    """Return y = 10^b * x^a."""
    return (10.0**b) * (np.asarray(x) ** a)


def critical_exponents(sigma_b2_tanh=0.1):
    """No-dropout scans, fitted exponents, errors, and plotting curves."""
    rho_no = 1.0

    # Critical tanh weight variance, where chi=1 without dropout.
    sigw2_c = tune_sigw2_for_chi_tanh(1, sigma_b2_tanh, rho_no, bracket=(0.1, 10.0))
    q_c, chi_c, _, _ = tanh_chi_g_h(sigw2_c, sigma_b2_tanh, rho_no)
    print(f"tanh critical sigma_w^2 = {sigw2_c:.8f}, chi={chi_c:.8f}, q*={q_c:.6f}")

    # Correlation length below criticality.
    offsets = np.logspace(-2, -1, 20)
    tvals_below = -offsets
    xi = -1 / np.log(1 + tvals_below)
    a_xi, b_xi, se_nu, _ = loglog_fit(offsets, xi)
    nu = -a_xi
    xi_fit = line_from_logfit(offsets, a_xi, b_xi)

    # Order parameters above criticality.
    tanh_points = []
    for sw2 in sigw2_c * (1 + offsets):
        _, chi, _, _ = tanh_chi_g_h(sw2, sigma_b2_tanh, rho_no)
        if chi > 1:
            tanh_points.append((chi - 1, 1 - cstar_tanh(sw2, sigma_b2_tanh, rho_no)))
    tvals_t, m_t = np.array(tanh_points).T
    tvals_r = offsets
    m_r = np.array([1 - cstar_relu(1 + t, rho_no) for t in tvals_r])

    # Fit the smaller half of each sweep to capture asymptotic scaling.
    fit_slice_t = slice(0, max(6, len(tvals_t) // 2))
    fit_slice_r = slice(0, max(6, len(tvals_r) // 2))
    beta_t, b_beta_t, se_beta_t, _ = loglog_fit(tvals_t, m_t, fit_slice=fit_slice_t)
    beta_r, b_beta_r, se_beta_r, _ = loglog_fit(tvals_r, m_r, fit_slice=fit_slice_r)
    tline = np.geomspace(tvals_t.min(), tvals_t.max(), 200)
    tline_r = np.geomspace(tvals_r.min(), tvals_r.max(), 200)
    m_fit_t = line_from_logfit(tline, beta_t, b_beta_t)
    m_fit_r = line_from_logfit(tline_r, beta_r, b_beta_r)

    # Iterate both maps at criticality. Each step depends on the previous one.
    Lmax = 20000
    ells = np.arange(1, Lmax + 1)
    m_l_t, m_l_r = np.empty(Lmax), np.empty(Lmax)
    c_t = c_r = 0.0
    for ell in range(Lmax):
        c_t = F_tanh(c_t, sigw2_c, sigma_b2_tanh, rho_no, qstar=q_c)
        c_r = F_relu_base(c_r)
        m_l_t[ell], m_l_r[ell] = 1 - c_t, 1 - c_r

    fit_win = (ells > 500) & (ells < 8000)
    a_p_t, b_p_t, se_p_t, _ = loglog_fit(ells[fit_win], m_l_t[fit_win])
    a_p_r, b_p_r, se_p_r, _ = loglog_fit(ells[fit_win], m_l_r[fit_win])
    p_t, p_r = -a_p_t, -a_p_r
    mfit_t = line_from_logfit(ells, a_p_t, b_p_t)
    mfit_r = line_from_logfit(ells, a_p_r, b_p_r)

    # Print summary of Part I exponents with standard errors
    print("\n" + "=" * 60)
    print("Part I: Critical exponents (no dropout)")
    print("=" * 60)
    print(f"  ν (correlation length):  {nu:.4f} ± {se_nu:.4f}")
    print(f"  β (tanh order param):    {beta_t:.4f} ± {se_beta_t:.4f}")
    print(f"  β (ReLU order param):    {beta_r:.4f} ± {se_beta_r:.4f}")
    print(f"  theta_rel (tanh):     {p_t:.4f} ± {se_p_t:.4f}")
    print(f"  theta_rel (ReLU):     {p_r:.4f} ± {se_p_r:.4f}")
    sigw2_grid = sigw2_c * (1 + offsets)
    _, _, g_c, h_c = tanh_chi_g_h(sigw2_c, sigma_b2_tanh, rho_no)

    return SimpleNamespace(
        sigw2_c=sigw2_c,
        q_c=q_c,
        chi_c=chi_c,
        offsets=offsets,
        tvals_below=tvals_below,
        xi=xi,
        a_xi=a_xi,
        b_xi=b_xi,
        nu=nu,
        se_nu=se_nu,
        xi_fit=xi_fit,
        tvals_t=tvals_t,
        m_t=m_t,
        tvals_r=tvals_r,
        m_r=m_r,
        beta_t=beta_t,
        b_beta_t=b_beta_t,
        se_beta_t=se_beta_t,
        beta_r=beta_r,
        b_beta_r=b_beta_r,
        se_beta_r=se_beta_r,
        tline=tline,
        tline_r=tline_r,
        m_fit_t=m_fit_t,
        m_fit_r=m_fit_r,
        ells=ells,
        m_l_t=m_l_t,
        m_l_r=m_l_r,
        fit_win=fit_win,
        p_t=p_t,
        p_r=p_r,
        se_p_t=se_p_t,
        se_p_r=se_p_r,
        mfit_t=mfit_t,
        mfit_r=mfit_r,
        sigw2_grid=sigw2_grid,
        g_c=g_c,
        h_c=h_c,
    )


def dropout_exponents(sigma_b2_tanh=0.1):
    """Critical dropout scans and fits over the smallest field decade."""
    # Keep the original two dropout ranges, including their shared endpoint.
    p_list = np.concatenate([np.geomspace(1e-3, 1e-2, 8), np.geomspace(1e-2, 0.2, 15)])
    rho_list = 1 - p_list

    # Tanh at chi=1.
    smooth_dropout = []
    for rho in rho_list:
        sw2 = tune_sigw2_for_chi_tanh(1, sigma_b2_tanh, rho)
        q, _, _, h = tanh_chi_g_h(sw2, sigma_b2_tanh, rho)
        c = cstar_tanh(sw2, sigma_b2_tanh, rho)
        lam = Fp_tanh(c, sw2, sigma_b2_tanh, rho, qstar=q)
        smooth_dropout.append((1 - c, -1 / np.log(lam), h))
    m_s, xi_s, h_s = np.array(smooth_dropout).T

    # ReLU at chi=1, with field h=(1-rho)/rho.
    c_k = np.array([cstar_relu(1, rho) for rho in rho_list])
    m_k = 1 - c_k
    h_k = (1 - rho_list) / rho_list
    xi_k = -1 / np.log(1 - np.arccos(c_k) / np.pi)

    # Fit exponents using the smallest h decade
    h_fit_max = 10.0 * min(h_s.min(), h_k.min())
    fit_s, fit_k = h_s <= h_fit_max, h_k <= h_fit_max

    a_ms, b_ms, se_a_ms, _ = loglog_fit(h_s[fit_s], m_s[fit_s])
    a_mk, b_mk, se_a_mk, _ = loglog_fit(h_k[fit_k], m_k[fit_k])

    # Transform each fitted slope and its error to delta=1/slope.
    delta_s, se_delta_s = 1 / a_ms, se_a_ms / a_ms**2
    delta_k, se_delta_k = 1 / a_mk, se_a_mk / a_mk**2

    a_xs, b_xs, se_nu_rho_s, _ = loglog_fit(h_s[fit_s], xi_s[fit_s])
    a_xk, b_xk, se_nu_rho_k, _ = loglog_fit(h_k[fit_k], xi_k[fit_k])

    nu_rho_s, nu_rho_k = -a_xs, -a_xk

    print("\n" + "=" * 60)
    print("Part II: Dropout perturbation exponents (small h fits)")
    print("=" * 60)
    print("  Smooth (tanh):")
    print(f"    1/δ = {a_ms:.4f} ± {se_a_ms:.4f}  (δ = {delta_s:.3f} ± {se_delta_s:.3f})")
    print(f"    ν_ρ = {nu_rho_s:.4f} ± {se_nu_rho_s:.4f}")
    print("  Kinked (ReLU):")
    print(f"    1/δ = {a_mk:.4f} ± {se_a_mk:.4f}  (δ = {delta_k:.3f} ± {se_delta_k:.3f})")
    print(f"    ν_ρ = {nu_rho_k:.4f} ± {se_nu_rho_k:.4f}")

    return SimpleNamespace(
        p_list=p_list,
        rho_list=rho_list,
        h_s=h_s,
        m_s=m_s,
        xi_s=xi_s,
        fit_s=fit_s,
        h_k=h_k,
        m_k=m_k,
        xi_k=xi_k,
        fit_k=fit_k,
        a_ms=a_ms,
        b_ms=b_ms,
        se_a_ms=se_a_ms,
        a_mk=a_mk,
        b_mk=b_mk,
        se_a_mk=se_a_mk,
        a_xs=a_xs,
        b_xs=b_xs,
        se_nu_rho_s=se_nu_rho_s,
        a_xk=a_xk,
        b_xk=b_xk,
        se_nu_rho_k=se_nu_rho_k,
        delta_s=delta_s,
        se_delta_s=se_delta_s,
        delta_k=delta_k,
        se_delta_k=se_delta_k,
        nu_rho_s=nu_rho_s,
        nu_rho_k=nu_rho_k,
    )


def scaling_scan(h_list, t_vals, kink_t_vals, sigma_b2_tanh=0.1):
    """Full-map fixed points at fixed keep probabilities for both activations."""
    smooth = []
    for proxy in h_list:
        rho = 1 / (1 + proxy)
        for t in t_vals:
            sw2 = tune_sigw2_for_chi_tanh(1 + t, sigma_b2_tanh, rho)
            q, chi, g, h = tanh_chi_g_h(sw2, sigma_b2_tanh, rho)
            m = 1 - cstar_tanh(sw2, sigma_b2_tanh, rho)
            smooth.append((proxy, t, h, g, m))
    smooth = np.array(smooth)

    # The original ReLU figure fixes rho, with h0 the field at chi=1.
    kink = relu_fixed_rho_scan(h_list, kink_t_vals)

    return smooth, kink
