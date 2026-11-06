"""exp_3: Resilience-response-only volatility surfaces in the low-z and
high-z regimes (Section 9, Figures 3 and 4).

Only the resilience argument z(m) = z_regime * exp(nu_skew * m) is
state-dependent (Eq. 41); liquidity L(m) = L_0 and the effective
signed-forcing A_eff(m) = A_eff_0 are held fixed. In the low-z regime the
leading response F_gamma(z) is, to leading order, independent of z
(Eq. 22), so this case is a diagnostic control: sigma_loc(m, T) should be
flat in m and vary only with maturity T through alpha(T). In the high-z
regime F_gamma(z) ~ K_gamma^(0) z^(gamma-1), so z(m) becomes a direct
source of state-dependent morphology.

Run: python -m exp.exp3_resilience_only_surfaces
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
        L_skew=cfg.L_SKEW_FIXED,
        L_smile=cfg.L_SMILE_FIXED,
        L_floor=cfg.L_FLOOR_FIXED,
        nu_skew=cfg.NU_SKEW_ACTIVE,
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
        ("low", cfg.Z_LOW, "exp3_fig3_low_z", "Figure 3: resilience-only, low-z regime (diagnostic control)"),
        ("high", cfg.Z_HIGH_0, "exp3_fig4_high_z", "Figure 4: resilience-only, high-z regime"),
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
        # Spread across m at fixed T: should be ~0 in the low-z regime
        # (diagnostic control) and material in the high-z regime.
        m_spread_by_T = (sigma_table.max(axis=1) - sigma_table.min(axis=1)).tolist()
        results[regime] = {
            "z_regime": z_regime,
            "sigma_loc_table": sigma_table.tolist(),
            "sigma_loc_vs_T_at_m0": sigma_table[:, len(cfg.M_GRID) // 2].tolist(),
            "m_spread_by_maturity": m_spread_by_T,
            "max_m_spread": float(max(m_spread_by_T)),
        }
        logger.info(
            "%s: sigma_loc range [%.4f, %.4f], max spread across m at fixed T = %.5f",
            label,
            sigma_table.min(),
            sigma_table.max(),
            max(m_spread_by_T),
        )

    low_flat = results["low"]["max_m_spread"] < 0.01
    high_varies = results["high"]["max_m_spread"] > 0.05
    results["low_z_is_flat_in_m"] = low_flat
    results["high_z_shows_state_dependence"] = high_varies
    logger.info(
        "Low-z flat in m (diagnostic control holds): %s; high-z shows state "
        "dependence: %s",
        low_flat,
        high_varies,
    )

    with open(RESULTS_DIR / "exp3_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved results to %s", RESULTS_DIR / "exp3_summary.json")
    return results


if __name__ == "__main__":
    run()
