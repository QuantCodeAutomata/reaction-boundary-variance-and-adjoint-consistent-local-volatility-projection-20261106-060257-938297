"""exp_2: Liquidity-only volatility surfaces in the low-z and high-z regimes
(Section 9, Figures 1 and 2).

Only the liquidity profile L(m) is state-dependent (Eq. 40); the resilience
argument z(m) and the effective signed-forcing A_eff(m) are held fixed
(nu_skew = 0, A_stress = 0), so any morphology in sigma_loc(m, T) comes from
the inverse-square liquidity channel 1/L(m)^2 in the operational kernel
Xi (Eq. 24), evaluated through the activity-clock projection (Eq. 38).

Run: python -m exp.exp2_liquidity_only_surfaces
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
        nu_skew=cfg.NU_SKEW_FIXED,
        A_eff_0=cfg.A_EFF_0,
        A_stress=cfg.A_STRESS_FIXED,
    )


def _grid(n_m: int, n_t: int) -> tuple[np.ndarray, np.ndarray]:
    m = np.linspace(cfg.M_GRID[0], cfg.M_GRID[-1], n_m)
    T = np.linspace(cfg.T_GRID[0], cfg.T_GRID[-1], n_t)
    return np.meshgrid(m, T)


def run() -> dict:
    table_m, table_T = np.meshgrid(np.array(cfg.M_GRID), np.array(cfg.T_GRID))
    plot_m, plot_T = _grid(cfg.PLOT_GRID_POINTS_M, cfg.PLOT_GRID_POINTS_T)

    kwargs = _common_kwargs()
    results: dict = {"m_grid": list(cfg.M_GRID), "T_grid": list(cfg.T_GRID)}

    for regime, z_regime, fig_name, label in (
        ("low", cfg.Z_LOW, "exp2_fig1_low_z", "Figure 1: liquidity-only, low-z regime"),
        ("high", cfg.Z_HIGH_0, "exp2_fig2_high_z", "Figure 2: liquidity-only, high-z regime"),
    ):
        sigma_table = projected_local_volatility(
            table_m, table_T, z_regime=z_regime, regime=regime, **kwargs
        )
        sigma_plot = projected_local_volatility(
            plot_m, plot_T, z_regime=z_regime, regime=regime, **kwargs
        )
        plot_surface_and_contour(
            plot_m, plot_T, sigma_plot, label, RESULTS_DIR / f"{fig_name}.png"
        )
        results[regime] = {
            "z_regime": z_regime,
            "sigma_loc_table": sigma_table.tolist(),
            "min": float(sigma_table.min()),
            "max": float(sigma_table.max()),
        }
        logger.info(
            "%s: sigma_loc range [%.4f, %.4f]",
            label,
            sigma_table.min(),
            sigma_table.max(),
        )

    # Diagnostic: the two regimes should share the same shape up to an overall
    # scale, since only the liquidity channel is state-dependent (Section 9:
    # "changing the response regime changes the volatility scale ... without
    # introducing a new source of state-dependent morphology").
    low = np.array(results["low"]["sigma_loc_table"])
    high = np.array(results["high"]["sigma_loc_table"])
    ratio = high / low
    shape_preserved = bool(np.allclose(ratio, ratio[0, 0], rtol=5e-3))
    results["shape_preserved_across_regimes"] = shape_preserved
    results["scale_ratio_high_over_low"] = float(ratio[0, 0])
    logger.info(
        "Shape preserved across regimes: %s (scale ratio %.4f)",
        shape_preserved,
        ratio[0, 0],
    )

    with open(RESULTS_DIR / "exp2_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved results to %s", RESULTS_DIR / "exp2_summary.json")
    return results


if __name__ == "__main__":
    run()
