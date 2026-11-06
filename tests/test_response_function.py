"""Unit tests for src.response_function (Section 5, Eq. 15, 21-23)."""

from __future__ import annotations

import numpy as np
import pytest

from src.response_function import (
    c_gamma,
    f_gamma,
    f_gamma_high_z_asymptote,
    f_gamma_low_z_asymptote,
    k_gamma_0,
)


@pytest.mark.parametrize("gamma", [0.30, 0.55, 0.80])
def test_c_gamma_positive(gamma: float) -> None:
    """C_gamma = 2*Gamma(1-gamma)*sin(pi*gamma/2) must be positive for 0<gamma<1."""
    assert c_gamma(gamma) > 0.0


@pytest.mark.parametrize("gamma", [0.30, 0.55, 0.80])
def test_k_gamma_0_positive(gamma: float) -> None:
    """K_gamma^(0) (Eq. 23) must be positive for 0<gamma<1."""
    assert k_gamma_0(gamma) > 0.0


def test_gamma_out_of_range_rejected() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            c_gamma(bad)
        with pytest.raises(ValueError):
            k_gamma_0(bad)
        with pytest.raises(ValueError):
            f_gamma_low_z_asymptote(bad)


def test_low_z_asymptotic_constant_known_value() -> None:
    """gamma=0.55: tan(0.275*pi)/(2*0.45) per the paper's worked example (Section 5)."""
    expected = np.tan(0.275 * np.pi) / (2.0 * 0.45)
    assert f_gamma_low_z_asymptote(0.55) == pytest.approx(expected, rel=1e-10)
    assert f_gamma_low_z_asymptote(0.55) == pytest.approx(1.298, abs=0.01)


@pytest.mark.parametrize("gamma", [0.30, 0.55, 0.80])
def test_f_gamma_converges_to_low_z_asymptote(gamma: float) -> None:
    """F_gamma(z) -> tan(pi*gamma/2)/(2*(1-gamma)) for z << 1 (Eq. 22)."""
    z = 1e-3
    numeric = f_gamma(z, gamma)
    asymptote = f_gamma_low_z_asymptote(gamma)
    rel_err = abs(numeric - asymptote) / asymptote
    assert rel_err < 0.01, f"gamma={gamma}: rel_err={rel_err}"


@pytest.mark.parametrize("gamma", [0.30, 0.55, 0.80])
def test_f_gamma_converges_to_high_z_asymptote(gamma: float) -> None:
    """F_gamma(z) ~ K_gamma^(0) z^(gamma-1) for z >> 1 (Eq. 22).

    The subleading correction to this leading-order asymptote grows as
    gamma -> 0, so the relative error at a fixed z is largest for the
    smallest gamma tested; what Eq. (22) guarantees is that the error
    vanishes as z increases, not a uniform rate in gamma. We therefore
    check the convergence trend across an increasing z sequence instead of
    a single fixed threshold.
    """
    z_values = [50.0, 200.0, 1000.0, 5000.0]
    rel_errors = []
    for z in z_values:
        numeric = f_gamma(z, gamma)
        asymptote = float(f_gamma_high_z_asymptote(z, gamma))
        rel_errors.append(abs(numeric - asymptote) / asymptote)
    # Monotone (non-strict) decrease towards the asymptote.
    assert all(a >= b - 1e-9 for a, b in zip(rel_errors, rel_errors[1:]))
    assert rel_errors[-1] < 0.1, f"gamma={gamma}: rel_err at z=5000 is {rel_errors[-1]}"


@pytest.mark.parametrize("gamma", [0.30, 0.55, 0.80])
def test_f_gamma_monotonically_decreasing(gamma: float) -> None:
    """F_gamma(z) must decrease monotonically in z (Section 5 expected outcome)."""
    z_grid = np.logspace(-3, 2, 60)
    values = f_gamma(z_grid, gamma)
    assert np.all(np.diff(values) <= 1e-9)


def test_f_gamma_vectorised_matches_scalar() -> None:
    z_grid = np.array([0.01, 1.0, 10.0])
    vec = f_gamma(z_grid, 0.55)
    scalar = np.array([f_gamma(float(z), 0.55) for z in z_grid])
    np.testing.assert_allclose(vec, scalar, rtol=1e-10)


def test_f_gamma_rejects_nonpositive_z() -> None:
    with pytest.raises(ValueError):
        f_gamma(0.0, 0.55)
    with pytest.raises(ValueError):
        f_gamma(-1.0, 0.55)
