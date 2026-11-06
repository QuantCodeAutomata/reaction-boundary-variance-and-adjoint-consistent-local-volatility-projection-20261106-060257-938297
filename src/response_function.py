"""Dimensionless spectral response function F_gamma(z) (Section 5, Eq. 21-23).

Custom implementation of Eq. (21):

    F_gamma(z) = (C_gamma / 2*pi) * int_{-inf}^{inf}
                 sin^2(xi/2) |xi|^(gamma-1) / sqrt(z^2 + xi^2) dxi,   0 < gamma < 1

Since the integrand is even in xi (sin^2(xi/2), |xi|^(gamma-1) and
sqrt(z^2+xi^2) are all even functions of xi), Eq. (21) is rewritten as the
one-sided integral used for the numerical quadrature:

    F_gamma(z) = (C_gamma / pi) * int_0^inf sin^2(xi/2) xi^(gamma-1) / sqrt(z^2+xi^2) dxi

The one-sided integral is further split with sin^2(xi/2) = (1-cos(xi))/2
into a non-oscillatory algebraic integral and an oscillatory Fourier-cosine
integral (see ``f_gamma_single``), which is far more robust for large z than
direct quadrature of the raw oscillatory integrand.

No quantitative-finance library implements this paper-specific spectral
integral or its asymptotics, so it is implemented directly with
scipy.special (Gamma function) and scipy.integrate.quad (adaptive
quadrature, including its oscillatory weight='cos' mode), as suggested by
the paper's own numerical notes (Section 5).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import gamma as gamma_fn


def c_gamma(gamma: float) -> float:
    """Normalisation constant C_gamma = 2*Gamma(1-gamma)*sin(pi*gamma/2) (Eq. 15)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between 0 and 1")
    return 2.0 * gamma_fn(1.0 - gamma) * np.sin(np.pi * gamma / 2.0)


def k_gamma_0(gamma: float) -> float:
    """High-z prefactor K_gamma^(0), Eq. (23).

    K_gamma^(0) = Gamma(1-gamma) sin(pi*gamma/2) Gamma(gamma/2) Gamma((1-gamma)/2)
                  / (2 * pi^(3/2))
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between 0 and 1")
    numerator = (
        gamma_fn(1.0 - gamma)
        * np.sin(np.pi * gamma / 2.0)
        * gamma_fn(gamma / 2.0)
        * gamma_fn((1.0 - gamma) / 2.0)
    )
    return numerator / (2.0 * np.pi ** 1.5)


def f_gamma_low_z_asymptote(gamma: float) -> float:
    """Low-z asymptotic constant tan(pi*gamma/2) / (2*(1-gamma)), Eq. (22)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between 0 and 1")
    return np.tan(np.pi * gamma / 2.0) / (2.0 * (1.0 - gamma))


def f_gamma_high_z_asymptote(z: np.ndarray | float, gamma: float) -> np.ndarray:
    """High-z asymptotic form K_gamma^(0) * z^(gamma-1), Eq. (22)."""
    z = np.asarray(z, dtype=float)
    if np.any(z <= 0):
        raise ValueError("z must be strictly positive")
    return k_gamma_0(gamma) * z ** (gamma - 1.0)


def _power_law_kernel(xi: float, z: float, gamma: float) -> float:
    """xi^(gamma-1) / sqrt(z^2+xi^2), guarded against the xi=0 endpoint."""
    if xi <= 0.0:
        return 0.0
    return xi ** (gamma - 1.0) / np.sqrt(z ** 2 + xi ** 2)


def f_gamma_single(z: float, gamma: float, limit: int = 200) -> float:
    """Numerically evaluate F_gamma(z) for one scalar z via adaptive quadrature.

    Using sin^2(xi/2) = (1 - cos(xi)) / 2, Eq. (21)'s one-sided integral splits
    into a non-oscillatory algebraic integral and an oscillatory (Fourier
    cosine) integral of the same kernel xi^(gamma-1)/sqrt(z^2+xi^2):

        F_gamma(z) = (C_gamma / pi) * (1/2) * (I1 - I2)
        I1 = int_0^inf xi^(gamma-1) / sqrt(z^2+xi^2) dxi            (non-oscillatory)
        I2 = int_0^inf cos(xi) xi^(gamma-1) / sqrt(z^2+xi^2) dxi    (oscillatory)

    Both xi^(gamma-1) singularities at xi=0 are integrable for 0 < gamma < 1.
    I1 is evaluated by plain adaptive quadrature; I2 uses scipy's QUADPACK
    oscillatory (weight='cos') quadrature, which is far more accurate and
    much faster for large z than direct quadrature of the raw oscillatory
    integrand of Eq. (21) over an infinite range (not specified in the paper;
    this splitting is a numerical-stability choice for the adaptive
    quadrature used to evaluate Eq. (21)).
    """
    if z <= 0.0:
        raise ValueError("z must be strictly positive")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between 0 and 1")

    # The kernel xi^(gamma-1)/sqrt(z^2+xi^2) transitions from an algebraic
    # singularity (xi << z) to a near power-law decay (xi >> z) around
    # xi = z. Splitting the non-oscillatory integral I1 there (not specified
    # in the paper; a numerical-stability choice) avoids the roundoff
    # ("does not converge") warning QUADPACK otherwise raises for z << 1.
    i1a, _ = integrate.quad(_power_law_kernel, 0.0, z, args=(z, gamma), limit=limit)
    i1b, _ = integrate.quad(_power_law_kernel, z, np.inf, args=(z, gamma), limit=limit)
    i1 = i1a + i1b
    i2, _ = integrate.quad(
        _power_law_kernel, 0.0, np.inf, args=(z, gamma), weight="cos", wvar=1.0, limit=limit
    )
    one_sided_integral = 0.5 * (i1 - i2)
    return c_gamma(gamma) / np.pi * one_sided_integral


def f_gamma(z: np.ndarray | float, gamma: float, limit: int = 200) -> np.ndarray:
    """Vectorised evaluation of F_gamma(z) over an array (or scalar) of z values."""
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    values = np.array([f_gamma_single(zi, gamma, limit) for zi in z_arr])
    return values if np.ndim(z) else values[0]
