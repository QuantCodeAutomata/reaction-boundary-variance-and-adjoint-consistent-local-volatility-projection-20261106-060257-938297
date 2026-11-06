"""Paper-derived parameters for every experiment.

All parameters are module-level UPPER_CASE variables; no classes or
dictionaries. Each comment states the meaning, unit (dimensionless unless
noted) and source: the paper section/equation, or "not specified in the
paper" when the paper leaves the value open.

Paper: "Reaction-boundary variance and adjoint-consistent local-volatility
projection". Section and equation numbers below follow that paper.
"""

# ---------------------------------------------------------------------------
# exp_1: response function F_gamma(z) (Section 5, Eq. 21-23)
# ---------------------------------------------------------------------------

# Values of the long-memory exponent gamma tested in exp_1, matching the
# Base/Min/Max column of Table 1 (Section 9): gamma in (0, 1).
GAMMA_VALUES_EXP1 = (0.30, 0.55, 0.80)

# Dense logarithmic grid of the dimensionless response argument z = nu_u*Delta
# spanning the deep low-z and deep high-z regimes (Section 5).
Z_GRID_MIN = 1e-3   # deep low-z endpoint (not specified numerically in the paper)
Z_GRID_MAX = 1e2    # deep high-z endpoint (not specified numerically in the paper)
Z_GRID_POINTS = 500  # grid resolution (not specified in the paper)

# Convergence thresholds used to verify Eq. (22): relative deviation of the
# numerical F_gamma(z) from each asymptotic branch, evaluated at the deep
# low-z and deep high-z ends of the grid (Section 5, "the two limits...").
LOW_Z_CHECK_THRESHOLD = 0.01   # z < 0.01 should match the low-z constant to 1%
HIGH_Z_CHECK_Z = 50.0          # z > 50 should match K_gamma^(0) z^(gamma-1)

# ---------------------------------------------------------------------------
# exp_2, exp_3, exp_4: structural volatility-surface simulations (Section 9)
# ---------------------------------------------------------------------------
# Dimensionless baseline values from Table 1 (Section 9).

GAMMA_BASELINE = 0.55        # gamma, Table 1 baseline ("Base" column)
DELTA_BASELINE = 1.0         # Delta, operational coarse-graining scale, Table 1
A_EFF_0 = 1.0                # A_eff,0, effective signed-forcing intensity scale, Table 1
L_0 = 1.0                    # L_0, baseline liquidity slope, Table 1
ALPHA_0 = 1.0                # alpha_0, baseline activity, Table 1 (Eq. 39)
A_SHORT = 0.25               # a_short, short-maturity activity lift, Eq. (39)
TAU_ALPHA = 0.50             # tau_alpha, activity decay scale in years, Eq. (39)
Z_LOW = 0.05                 # z_low, low-z response regime baseline, Table 1
Z_HIGH_0 = 10.0              # z_high,0, high-z response regime baseline, Table 1

# Liquidity profile L(m) = L_0 * max(L_floor, 1 + L_skew*m + L_smile*m^2), Eq. (40).
L_SKEW_ACTIVE = 0.80    # lower downside liquidity when positive (Section 9)
L_SMILE_ACTIVE = 1.50   # liquidity variation away from centre (Section 9)
L_FLOOR_ACTIVE = 0.20   # positive floor on the liquidity profile (Section 9)

# Liquidity held fixed (resilience-only experiment, exp_3): L(m) = L_0, i.e.
# L_skew = L_smile = 0 so that the max(.) branch with L_floor never binds;
# L_floor is set to 1.0 so the floor is inactive at L(m) = L_0 (not specified
# numerically in the paper beyond "we hold liquidity ... fixed", Section 9).
L_SKEW_FIXED = 0.0
L_SMILE_FIXED = 0.0
L_FLOOR_FIXED = 1.0

# Resilience-response profile z(m) = z_regime * exp(nu_skew*m), Eq. (41).
NU_SKEW_ACTIVE = 1.00   # weaker downside resilience when positive (Section 9)
NU_SKEW_FIXED = 0.0     # resilience held fixed (liquidity-only experiment, exp_2)

# Effective signed-forcing profile A_eff(m) = A_eff_0*exp(-A_stress*m), Eq. (42).
A_STRESS_COMBINED = 0.75  # stronger downside forcing in the combined profile, Section 9
A_STRESS_FIXED = 0.0      # forcing held fixed (exp_2 and exp_3)

# Rectangular evaluation grid in log-moneyness m and maturity T (years), used
# for exp_2 and exp_3; matches the contour-table abscissae in Section 9.
M_GRID = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
T_GRID = (0.0, 0.25, 0.50, 1.00, 1.50, 1.75, 2.00)

# Reduced maturity grid used for the combined-channel 3D surface table in
# exp_4, matching the abscissae printed in Section 9 for Figures 5 and 6.
T_GRID_EXP4 = (0.0, 0.25, 0.50, 1.00, 2.00)

# Finer grid used only for plotting (3D surface / contour resolution); not
# specified in the paper beyond the contour-table abscissae above.
PLOT_GRID_POINTS_M = 61
PLOT_GRID_POINTS_T = 61
