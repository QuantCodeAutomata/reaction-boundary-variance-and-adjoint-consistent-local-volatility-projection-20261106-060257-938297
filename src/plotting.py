"""Shared matplotlib plotting helpers for the surface experiments (Section 9).

Each experiment produces a matched 3D surface / 2D contour pair, as described
for Figures 1-6 of the paper ("Each case is shown as a matched three-
dimensional surface and contour pair so that the overall morphology and the
corresponding level-set structure can be read together.").
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def plot_surface_and_contour(
    m_grid: np.ndarray,
    T_grid: np.ndarray,
    sigma: np.ndarray,
    title: str,
    out_path: Path,
    contour_levels: int = 15,
) -> None:
    """Save a matched 3D surface and 2D contour plot of sigma_loc(m, T).

    ``m_grid``/``T_grid`` are 2D meshgrid arrays and ``sigma`` has the same
    shape (rows indexed by maturity T, columns by log-moneyness m).
    """
    fig = plt.figure(figsize=(13, 5.5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(m_grid, T_grid, sigma, cmap="viridis", edgecolor="none")
    ax1.set_xlabel("log-moneyness m")
    ax1.set_ylabel("maturity T (years)")
    ax1.set_zlabel("sigma_loc")
    ax1.set_title(f"{title}\n3D surface")

    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(m_grid, T_grid, sigma, levels=contour_levels, cmap="viridis")
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel("log-moneyness m")
    ax2.set_ylabel("maturity T (years)")
    ax2.set_title(f"{title}\ncontour")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
