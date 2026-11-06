"""Unit tests for src.surface (Section 6 Eq. 24-25, Section 9 Eq. 38-42)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src.surface import (
    activity_profile,
    forcing_profile,
    liquidity_profile,
    operational_kernel,
    projected_local_volatility,
    resilience_argument,
    response_multiplier,
)


def test_activity_profile_decays_to_alpha0() -> None:
    """alpha(T) -> alpha_0 as T -> infinity, and alpha(0) = alpha_0*(1+a_short)."""
    alpha_0, a_short, tau = 1.0, 0.25, 0.5
    assert activity_profile(np.array([0.0]), alpha_0, a_short, tau)[0] == pytest.approx(
        alpha_0 * (1 + a_short)
    )
    assert activity_profile(np.array([50.0]), alpha_0, a_short, tau)[0] == pytest.approx(
        alpha_0, abs=1e-6
    )


def test_liquidity_profile_floor_binds() -> None:
    """L(m) never drops below L_0 * L_floor (the max(.) branch in Eq. 40)."""
    m = np.linspace(-0.3, 0.3, 50)
    L = liquidity_profile(m, L_0=1.0, L_skew=0.80, L_smile=1.50, L_floor=0.20)
    assert np.all(L >= 0.20 - 1e-12)


def test_liquidity_profile_lower_on_downside() -> None:
    """With L_skew > 0, liquidity is lower for negative m than for positive m."""
    L_minus = liquidity_profile(np.array([-0.2]), 1.0, 0.80, 1.50, 0.20)[0]
    L_plus = liquidity_profile(np.array([0.2]), 1.0, 0.80, 1.50, 0.20)[0]
    assert L_minus < L_plus


def test_resilience_argument_scales_with_regime() -> None:
    z = resilience_argument(np.array([0.0]), z_regime=10.0, nu_skew=1.0)[0]
    assert z == pytest.approx(10.0)


def test_forcing_profile_higher_on_downside_when_stressed() -> None:
    """A_eff(m) = A_eff_0*exp(-A_stress*m) is higher for m<0 when A_stress>0."""
    A_minus = forcing_profile(np.array([-0.2]), 1.0, 0.75)[0]
    A_plus = forcing_profile(np.array([0.2]), 1.0, 0.75)[0]
    assert A_minus > A_plus


def test_response_multiplier_low_regime_is_constant_in_z() -> None:
    z = np.array([0.01, 0.05, 0.1])
    F = response_multiplier(z, gamma=0.55, regime="low")
    assert np.allclose(F, F[0])


def test_response_multiplier_high_regime_decreases_with_z() -> None:
    """K_gamma^(0)*z^(gamma-1) decreases in z since gamma-1<0 for 0<gamma<1."""
    z = np.array([5.0, 10.0, 20.0])
    F = response_multiplier(z, gamma=0.55, regime="high")
    assert np.all(np.diff(F) < 0)


def test_response_multiplier_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError):
        response_multiplier(np.array([1.0]), gamma=0.55, regime="mid")


def test_operational_kernel_formula() -> None:
    xi = operational_kernel(A_eff=np.array([2.0]), L=np.array([1.0]), Delta=1.0, gamma=0.55, F_gamma_z=np.array([3.0]))
    assert xi[0] == pytest.approx(2.0 / 1.0 ** 2 * 1.0 ** (-0.55) * 3.0)


def test_projected_local_volatility_liquidity_only_shape_matches_exp2() -> None:
    """Liquidity-only (nu_skew=0, A_stress=0): sigma_loc driven by alpha(T) and L(m)."""
    m = np.array(cfg.M_GRID)
    T = np.array(cfg.T_GRID)
    M, Tg = np.meshgrid(m, T)
    sigma = projected_local_volatility(
        M, Tg,
        gamma=cfg.GAMMA_BASELINE, Delta=cfg.DELTA_BASELINE,
        alpha_0=cfg.ALPHA_0, a_short=cfg.A_SHORT, tau_alpha=cfg.TAU_ALPHA,
        L_0=cfg.L_0, L_skew=cfg.L_SKEW_ACTIVE, L_smile=cfg.L_SMILE_ACTIVE, L_floor=cfg.L_FLOOR_ACTIVE,
        z_regime=cfg.Z_LOW, nu_skew=cfg.NU_SKEW_FIXED,
        A_eff_0=cfg.A_EFF_0, A_stress=cfg.A_STRESS_FIXED,
        regime="low",
    )
    assert sigma.shape == M.shape
    assert np.all(sigma > 0)
    # Downside (m<0) must be more volatile than upside (m>0) at fixed T since
    # liquidity is lower there (Eq. 40 with L_skew>0).
    mid_T = len(T) // 2
    assert sigma[mid_T, 0] > sigma[mid_T, -1]


def test_projected_local_volatility_resilience_only_low_z_is_flat_in_m() -> None:
    """Resilience-only, low-z: sigma_loc should barely vary across m (Eq. 22 low-z limit)."""
    m = np.array(cfg.M_GRID)
    T = np.array(cfg.T_GRID)
    M, Tg = np.meshgrid(m, T)
    sigma = projected_local_volatility(
        M, Tg,
        gamma=cfg.GAMMA_BASELINE, Delta=cfg.DELTA_BASELINE,
        alpha_0=cfg.ALPHA_0, a_short=cfg.A_SHORT, tau_alpha=cfg.TAU_ALPHA,
        L_0=cfg.L_0, L_skew=cfg.L_SKEW_FIXED, L_smile=cfg.L_SMILE_FIXED, L_floor=cfg.L_FLOOR_FIXED,
        z_regime=cfg.Z_LOW, nu_skew=cfg.NU_SKEW_ACTIVE,
        A_eff_0=cfg.A_EFF_0, A_stress=cfg.A_STRESS_FIXED,
        regime="low",
    )
    spread = sigma.max(axis=1) - sigma.min(axis=1)
    assert np.all(spread < 0.01)


def test_projected_local_volatility_rejects_negative_variance_is_unreachable() -> None:
    """All profile factors are positive, so sigma_loc is always real and finite."""
    m = np.linspace(-0.3, 0.3, 7)
    T = np.linspace(0.0, 2.0, 7)
    M, Tg = np.meshgrid(m, T)
    sigma = projected_local_volatility(
        M, Tg,
        gamma=0.55, Delta=1.0, alpha_0=1.0, a_short=0.25, tau_alpha=0.5,
        L_0=1.0, L_skew=0.8, L_smile=1.5, L_floor=0.2,
        z_regime=10.0, nu_skew=1.0, A_eff_0=1.0, A_stress=0.75, regime="high",
    )
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)
