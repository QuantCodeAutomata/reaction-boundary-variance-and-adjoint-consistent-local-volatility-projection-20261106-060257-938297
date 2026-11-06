"""exp_4: Combined structural volatility surfaces in the low-z and high-z
regimes (Section 9, Figures 5 and 6).

All three structural channels are state-dependent at once: liquidity L(m)
(Eq. 40), the resilience argument z(m) (Eq. 41), and the effective
signed-forcing A_eff(m) (Eq. 42). In the low-z regime the leading response
F_gamma(z) is independent of z, so the surface combines only the liquidity
and forcing channels. In the high-z regime all three channels act together,
through the inverse-square liquidity factor, the response multiplier
K_gamma^(0) z^(gamma-1), and the downside forcing profile, which Section 9
states "produce the strongest downside asymmetry of the simulated cases".

Run: python -m exp.exp4_combined_structural_surfaces
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src import config as cfg
from src.plotting import plot_surface_and_contour
from src.surface import projected_local_volatility

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _common_kwargs() -> dict:
    return dict(
        gamma=cfg.GAMMA_BASELINE,
        Delta=cfg.DELTA_BASELINE,
        alpha_0=cfg.ALPHA_0,
        a_short=cfg.A_SHORT,
        tau_alpha=cfg.TAU_ALPHA,
        L_0=cfg.L_0,
        L_skew=cfg.L_SKEW_ACTIVE,
        L_smile=cfg.L_SMILE_ACTIVE,
        L_floor=cfg.L_FLOOR_ACTIVE,
        nu_skew=cfg.NU_SKEW_ACTIVE,
        A_eff_0=cfg.A_EFF_0,
        A_stress=cfg.A_STRESS_COMBINED,
    )


def _grid(n_m: int, n_t: int) -> tuple[np.ndarray, np.ndarray]:
    m = np.linspace(cfg.M_GRID[0], cfg.M_GRID[-1], n_m)
    T = np.linspace(cfg.T_GRID[0], cfg.T_GRID[-1], n_t)
    return np.meshgrid(m, T)


def run() -> dict:
    # Section 9's 3D-surface tables for Figures 5/6 use the reduced maturity
    # node set T_GRID_EXP4, while the contour tables use the full T_GRID.
    table_m, table_t_contour = np.meshgrid(np.array(cfg.M_GRID), np.array(cfg.T_GRID))
    table_m4, table_t4 = np.meshgrid(np.array(cfg.M_GRID), np.array(cfg.T_GRID_EXP4))
    plot_m, plot_T = _grid(cfg.PLOT_GRID_POINTS_M, cfg.PLOT_GRID_POINTS_T)

    kwargs = _common_kwargs()
    results: dict = {
        "m_grid": list(cfg.M_GRID),
        "T_grid_contour": list(cfg.T_GRID),
        "T_grid_surface": list(cfg.T_GRID_EXP4),
    }

    for regime, z_regime, fig_name, label in (
        ("low", cfg.Z_LOW, "exp4_fig5_low_z", "Figure 5: combined structural, low-z regime"),
        ("high", cfg.Z_HIGH_0, "exp4_fig6_high_z", "Figure 6: combined structural, high-z regime"),
    ):
        sigma_contour_table = projected_local_volatility(
            table_m, table_t_contour, z_regime=z_regime, regime=regime, **kwargs
        )
        sigma_surface_table = projected_local_volatility(
            table_m4, table_t4, z_regime=z_regime, regime=regime, **kwargs
        )
        sigma_plot = projected_local_volatility(
            plot_m, plot_T, z_regime=z_regime, regime=regime, **kwargs
        )
        plot_surface_and_contour(
            plot_m, plot_T, sigma_plot, label, RESULTS_DIR / f"{fig_name}.png"
        )
        m_spread_by_T = (
            sigma_contour_table.max(axis=1) - sigma_contour_table.min(axis=1)
        ).tolist()
        results[regime] = {
            "z_regime": z_regime,
            "sigma_loc_contour_table": sigma_contour_table.tolist(),
            "sigma_loc_surface_table": sigma_surface_table.tolist(),
            "min": float(sigma_contour_table.min()),
            "max": float(sigma_contour_table.max()),
            "max_m_spread": float(max(m_spread_by_T)),
        }
        logger.info(
            "%s: sigma_loc range [%.4f, %.4f], max downside/upside spread across m = %.4f",
            label,
            sigma_contour_table.min(),
            sigma_contour_table.max(),
            max(m_spread_by_T),
        )

    # Section 9 states that the combined high-z case produces "the strongest
    # downside asymmetry of the simulated cases". The response regime changes
    # the overall volatility *scale* (Section 9), so the absolute spread
    # max(sigma)-min(sigma) across m is not scale-invariant and is not the
    # right diagnostic to compare regimes with different scales. The
    # scale-invariant measure of downside asymmetry is the ratio of the
    # downside (m=-0.3) to the upside (m=+0.3) volatility at matched T; this
    # ratio is unaffected by an overall rescaling of sigma_loc.
    m_extremes = np.array([cfg.M_GRID[0], cfg.M_GRID[-1]])
    t_zero = np.array([0.0, 0.0])
    for regime, z_regime in (("low", cfg.Z_LOW), ("high", cfg.Z_HIGH_0)):
        sig_extremes = projected_local_volatility(
            m_extremes, t_zero, z_regime=z_regime, regime=regime, **kwargs
        )
        results[regime]["downside_upside_ratio"] = float(sig_extremes[0] / sig_extremes[1])

    results["high_z_has_largest_asymmetry_ratio"] = bool(
        results["high"]["downside_upside_ratio"] > results["low"]["downside_upside_ratio"]
    )
    logger.info(
        "Combined low-z downside/upside ratio = %.4f; combined high-z ratio = %.4f; "
        "high-z strongest asymmetry: %s",
        results["low"]["downside_upside_ratio"],
        results["high"]["downside_upside_ratio"],
        results["high_z_has_largest_asymmetry_ratio"],
    )

    with open(RESULTS_DIR / "exp4_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved results to %s", RESULTS_DIR / "exp4_summary.json")
    return results


if __name__ == "__main__":
    run()
