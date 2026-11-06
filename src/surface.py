"""Projected structural local-volatility surface (Section 6 and Section 9).

Custom implementation of the operational variance kernel Xi (Eq. 24), the
deterministic activity-clock projection (Eq. 25/38), and the phenomenological
state-dependent profiles alpha(T), L(m), z(m), A_eff(m) (Eqs. 39-42). None of
these paper-specific closed-form constructions are covered by a portfolio,
backtesting or volatility-modelling library (skfolio, pyportfolioopt,
vectorbt, arch, ...), so they are implemented directly from the formulas.
"""

from __future__ import annotations

import numpy as np

from src.response_function import f_gamma_high_z_asymptote, f_gamma_low_z_asymptote


def activity_profile(T: np.ndarray, alpha_0: float, a_short: float, tau_alpha: float) -> np.ndarray:
    """alpha(T) = alpha_0 * (1 + a_short * exp(-T / tau_alpha)), Eq. (39)."""
    T = np.asarray(T, dtype=float)
    return alpha_0 * (1.0 + a_short * np.exp(-T / tau_alpha))


def liquidity_profile(
    m: np.ndarray, L_0: float, L_skew: float, L_smile: float, L_floor: float
) -> np.ndarray:
    """L(m) = L_0 * max(L_floor, 1 + L_skew*m + L_smile*m^2), Eq. (40)."""
    m = np.asarray(m, dtype=float)
    return L_0 * np.maximum(L_floor, 1.0 + L_skew * m + L_smile * m ** 2)


def resilience_argument(m: np.ndarray, z_regime: float, nu_skew: float) -> np.ndarray:
    """z(m) = z_regime * exp(nu_skew * m), Eq. (41)."""
    m = np.asarray(m, dtype=float)
    return z_regime * np.exp(nu_skew * m)


def forcing_profile(m: np.ndarray, A_eff_0: float, A_stress: float) -> np.ndarray:
    """A_eff(m) = A_eff_0 * exp(-A_stress * m), Eq. (42)."""
    m = np.asarray(m, dtype=float)
    return A_eff_0 * np.exp(-A_stress * m)


def response_multiplier(z: np.ndarray, gamma: float, regime: str) -> np.ndarray:
    """F_gamma(z(m)) evaluated through the asymptotic branch of Eq. (22).

    Section 9 states that the surface simulations "evaluate the operational
    kernel in Eq. (24), using the asymptotic reaction-boundary closure derived
    in Section 5", so each regime uses the matching asymptotic branch of
    Eq. (22) rather than the full numerical integral of Eq. (21):
    the low-z regime uses the z-independent constant
    tan(pi*gamma/2) / (2*(1-gamma)), and the high-z regime uses
    K_gamma^(0) * z^(gamma-1).
    """
    z = np.asarray(z, dtype=float)
    if regime == "low":
        return np.full_like(z, f_gamma_low_z_asymptote(gamma), dtype=float)
    if regime == "high":
        return f_gamma_high_z_asymptote(z, gamma)
    raise ValueError("regime must be 'low' or 'high'")


def operational_kernel(
    A_eff: np.ndarray, L: np.ndarray, Delta: float, gamma: float, F_gamma_z: np.ndarray
) -> np.ndarray:
    """Xi(S, u; Delta) = [A_eff / L^2] * Delta^(-gamma) * F_gamma(z), Eq. (24)."""
    return (A_eff / L ** 2) * Delta ** (-gamma) * F_gamma_z


def projected_local_volatility(
    m: np.ndarray,
    T: np.ndarray,
    *,
    gamma: float,
    Delta: float,
    alpha_0: float,
    a_short: float,
    tau_alpha: float,
    L_0: float,
    L_skew: float,
    L_smile: float,
    L_floor: float,
    z_regime: float,
    nu_skew: float,
    A_eff_0: float,
    A_stress: float,
    regime: str,
) -> np.ndarray:
    """sigma_loc(m, T) = [alpha(T) * Xi(ln x0 + m, U(T); Delta)]^(1/2), Eq. (38).

    ``m`` and ``T`` must already be broadcastable grids (e.g. from
    ``numpy.meshgrid``). The reference price x0 cancels because the kernel is
    evaluated directly as a function of log-moneyness m (Eq. 37).
    """
    alpha_T = activity_profile(T, alpha_0, a_short, tau_alpha)
    L_m = liquidity_profile(m, L_0, L_skew, L_smile, L_floor)
    z_m = resilience_argument(m, z_regime, nu_skew)
    A_eff_m = forcing_profile(m, A_eff_0, A_stress)
    F_z = response_multiplier(z_m, gamma, regime)
    xi_kernel = operational_kernel(A_eff_m, L_m, Delta, gamma, F_z)
    variance = alpha_T * xi_kernel
    if np.any(variance < 0):
        raise ValueError("projected variance must be non-negative")
    return np.sqrt(variance)
