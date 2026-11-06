"""exp_1: Computation and asymptotic verification of F_gamma(z) (Section 5).

Computes the dimensionless spectral response function F_gamma(z) of Eq. (21)
by adaptive quadrature (src.response_function.f_gamma) for gamma in
{0.30, 0.55, 0.80} over z in [1e-3, 1e2], and checks the two asymptotic
limits of Eq. (22): the low-z constant tan(pi*gamma/2)/(2*(1-gamma)) and the
high-z form K_gamma^(0) * z^(gamma-1).

Run: python -m exp.exp1_response_function
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src.response_function import (
    c_gamma,
    f_gamma,
    f_gamma_high_z_asymptote,
    f_gamma_low_z_asymptote,
    k_gamma_0,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run() -> dict:
    z_grid = np.logspace(
        np.log10(cfg.Z_GRID_MIN), np.log10(cfg.Z_GRID_MAX), cfg.Z_GRID_POINTS
    )
    logger.info(
        "z grid: %d points from %.4g to %.4g", len(z_grid), z_grid[0], z_grid[-1]
    )

    summary: dict = {"z_grid_points": len(z_grid), "gamma_values": {}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for gamma in cfg.GAMMA_VALUES_EXP1:
        F_numeric = f_gamma(z_grid, gamma)
        low_const = f_gamma_low_z_asymptote(gamma)
        high_asym = f_gamma_high_z_asymptote(z_grid, gamma)

        # Verify monotonic decrease: for 0<gamma<1, F_gamma(z) should be
        # monotonically decreasing in z (Section 5 asymptotic expected outcome).
        monotone = bool(np.all(np.diff(F_numeric) <= 1e-9))

        # Relative error against each asymptotic branch across the grid.
        rel_err_low = np.abs(F_numeric - low_const) / low_const
        rel_err_high = np.abs(F_numeric - high_asym) / high_asym

        low_mask = z_grid < cfg.LOW_Z_CHECK_THRESHOLD
        high_mask = z_grid > cfg.HIGH_Z_CHECK_Z
        low_z_match = float(np.max(rel_err_low[low_mask])) if low_mask.any() else float("nan")
        high_z_match = float(np.max(rel_err_high[high_mask])) if high_mask.any() else float("nan")

        logger.info(
            "gamma=%.2f  C_gamma=%.6f  K_gamma0=%.6f  low-z const=%.6f  "
            "max rel.err z<%.3g: %.4f%%  max rel.err z>%.3g: %.4f%%",
            gamma,
            c_gamma(gamma),
            k_gamma_0(gamma),
            low_const,
            cfg.LOW_Z_CHECK_THRESHOLD,
            100 * low_z_match,
            cfg.HIGH_Z_CHECK_Z,
            100 * high_z_match,
        )

        summary["gamma_values"][f"{gamma:.2f}"] = {
            "C_gamma": float(c_gamma(gamma)),
            "K_gamma_0": float(k_gamma_0(gamma)),
            "low_z_asymptotic_constant": float(low_const),
            "monotonically_decreasing": monotone,
            "max_rel_error_vs_low_z_asymptote_below_threshold": low_z_match,
            "max_rel_error_vs_high_z_asymptote_above_threshold": high_z_match,
        }

        axes[0].loglog(z_grid, F_numeric, label=f"gamma={gamma:.2f} (numeric)")
        axes[0].loglog(
            z_grid, high_asym, "--", color=axes[0].lines[-1].get_color(), alpha=0.6,
            label=f"gamma={gamma:.2f} high-z asymptote",
        )
        axes[0].axhline(low_const, color=axes[0].lines[-1].get_color(), linestyle=":", alpha=0.6)

        axes[1].loglog(z_grid, rel_err_low, label=f"gamma={gamma:.2f} vs low-z const")
        axes[1].loglog(z_grid, rel_err_high, "--", label=f"gamma={gamma:.2f} vs high-z asym")

    axes[0].set_xlabel("z")
    axes[0].set_ylabel("F_gamma(z)")
    axes[0].set_title("Response function F_gamma(z), Eq. (21), vs asymptotes Eq. (22)")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("z")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("Relative error of F_gamma(z) vs each asymptotic branch")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_DIR / "exp1_response_function.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Save the full numeric table for one representative gamma (baseline 0.55)
    # plus the summary for all gammas, as the machine-readable deliverable.
    table = {
        "z": z_grid.tolist(),
        **{
            f"F_gamma_{g:.2f}": f_gamma(z_grid, g).tolist()
            for g in cfg.GAMMA_VALUES_EXP1
        },
    }
    with open(RESULTS_DIR / "exp1_F_gamma_table.json", "w") as fh:
        json.dump(table, fh)

    summary_path = RESULTS_DIR / "exp1_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Saved figure to %s", fig_path)
    logger.info("Saved summary to %s", summary_path)
    return summary


if __name__ == "__main__":
    run()
