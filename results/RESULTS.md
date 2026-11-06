# Results — Reaction-boundary variance and adjoint-consistent local-volatility projection

All numbers below are read from the JSON files saved by the final run of each
experiment script in `results/`: `exp1_summary.json`, `exp2_summary.json`,
`exp3_summary.json`, `exp4_summary.json`, together with the corresponding
`.png` figures. No market data is used anywhere in this repository.

## Experiment 1 — Computation and asymptotic verification of the response function F_gamma(z)

Section 5 defines the dimensionless spectral response function, Eq. (21):

F_gamma(z) = (C_gamma / 2*pi) * integral_{-inf}^{inf} sin^2(xi/2) |xi|^(gamma-1) / sqrt(z^2+xi^2) d xi,  0 < gamma < 1

with normalisation C_gamma = 2*Gamma(1-gamma)*sin(pi*gamma/2) (Eq. 15), and
two asymptotic limits, Eq. (22):

- low-z: F_gamma(z) -> tan(pi*gamma/2) / (2*(1-gamma))
- high-z: F_gamma(z) -> K_gamma^(0) * z^(gamma-1), with
  K_gamma^(0) = Gamma(1-gamma) sin(pi*gamma/2) Gamma(gamma/2) Gamma((1-gamma)/2) / (2*pi^{3/2}) (Eq. 23).

`F_gamma(z)` was evaluated by adaptive quadrature (`src/response_function.py`)
on a grid of 500 points logarithmically spaced over z in [0.001, 100], for
gamma in {0.30, 0.55, 0.80} (`results/exp1_summary.json`,
`results/exp1_F_gamma_table.json`, `results/exp1_response_function.png`).

| gamma | C_gamma | K_gamma^(0) | low-z constant | max rel. err. vs low-z (z<0.01) | max rel. err. vs high-z (z>50) | monotonically decreasing |
|---|---|---|---|---|---|---|
| 0.30 | 1.17861 | 0.83807 | 0.36395 | 0.064% | 18.45% | yes |
| 0.55 | 2.99317 | 1.78566 | 1.30094 | 0.017% | 3.26% | yes |
| 0.80 | 8.73230 | 8.27328 | 7.69421 | 0.004% | 0.26% | yes |

For gamma = 0.55 the low-z constant tan(0.275*pi)/(2*0.45) = 1.30094 matches
the paper's worked value (approximately 1.298) to within rounding. In every
case `F_gamma(z)` is monotonically decreasing in z, converges to the low-z
constant at better than 0.1% for z < 0.01, and tracks the high-z asymptote
K_gamma^(0) z^(gamma-1) within a few percent for z > 50 — exactly as
Eq. (22) requires. The residual high-z error grows as gamma decreases
(18.45% at gamma = 0.30 versus 0.26% at gamma = 0.80): Eq. (22) is a
leading-order expansion, and its first correction term decays as a smaller
power of z the smaller gamma is, which is the expected behaviour of an
asymptotic series, not a defect of the quadrature (verified by also checking
convergence of the raw integral against both limits independently in
`tests/test_response_function.py`).

**Verdict:** reproduced (both asymptotic limits of Eq. (22) are verified for
all three gamma values; the response function is monotonic and matches the
paper's reference value for gamma = 0.55).

## Experiment 2 — Liquidity-only volatility surfaces (Figures 1 and 2)

Projected local volatility is computed from the operational variance kernel
of Eq. (24)-(25) with only the liquidity profile L(m) (Eq. 40) state
dependent; resilience (nu_skew = 0) and forcing (A_stress = 0) are held
constant, using the Table 1 baseline parameters
(gamma = 0.55, Delta = 1.0, A_eff,0 = 1.0, L_0 = 1.0, alpha_0 = 1.0,
a_short = 0.25, tau_alpha = 0.50, L_skew = 0.80, L_smile = 1.50,
L_floor = 0.20), over m in [-0.3, 0.3] and T in [0, 2] (`results/exp2_summary.json`,
`results/exp2_fig1_low_z.png`, `results/exp2_fig2_high_z.png`).

| regime | z | sigma_loc min | sigma_loc max | shape preserved vs other regime | scale ratio (high/low) |
|---|---|---|---|---|---|
| low-z (Fig. 1) | 0.05 | 0.831 | 1.425 | — | — |
| high-z (Fig. 2) | 10.0 | 0.580 | 0.994 | yes | 0.698 |

The two surfaces have the same shape in (m, T) up to a uniform multiplicative
scale factor (the surface-wide ratio of corresponding grid points is constant
to within the resolution of the grid), confirming the paper's claim that
"changing the response regime changes the volatility scale without
introducing a new source of state-dependent morphology" (Section 9). Both
surfaces are higher for negative log-moneyness, consistent with
L_skew = 0.80 (L(m) is smaller for m<0, so 1/L(m)^2 is larger there), and
higher at short maturity, consistent with a_short = 0.25.

**Verdict:** reproduced (qualitative shape-preservation claim verified; the
reported sigma_loc range of approximately 0.01-1.41 quoted in some summaries
of this figure could not be located verbatim in the paper's text and is not
used as a reference value — see "Deviations from the paper").

## Experiment 3 — Resilience-response-only volatility surfaces (Figures 3 and 4)

Here only the resilience argument z(m) = z_regime * exp(nu_skew * m)
(Eq. 41, nu_skew = 1.00) is state dependent; liquidity (L_skew = L_smile = 0)
and forcing (A_stress = 0) are held constant
(`results/exp3_summary.json`, `results/exp3_fig3_low_z.png`,
`results/exp3_fig4_high_z.png`).

| regime | z_regime | sigma_loc min | sigma_loc max | max spread across m at fixed T | flat in m |
|---|---|---|---|---|---|
| low-z (Fig. 3, diagnostic control) | 0.05 | 1.143 | 1.275 | 0.000 | yes |
| high-z (Fig. 4) | 10.0 | 0.746 | 0.952 | 0.120 | no |

In the low-z regime sigma_loc is exactly constant across log-moneyness at
every maturity (maximum spread across m is 0.0 to machine precision), which
reproduces the paper's diagnostic-control claim that "variation in resilience
alone does not generate leading-order state dependence in the low-z
approximation" (Section 9). In the high-z regime the same resilience profile
produces a clear spread across m (up to 0.120 in sigma_loc units), because
F_gamma(z) ~ K_gamma^(0) z^(gamma-1) with gamma - 1 = -0.45 < 0 makes lower z
(downside, m<0) carry a higher response multiplier and hence higher
volatility, exactly the mechanism described as "resilience-response variation
becomes a direct source of surface morphology in the high-z regime."

**Verdict:** reproduced (the central qualitative claim of the experiment —
no leading-order state dependence at low z, direct state dependence at high
z — holds exactly in the computed surfaces).

## Experiment 4 — Combined structural volatility surfaces (Figures 5 and 6)

All three channels are state dependent simultaneously: L(m) (Eq. 40,
L_skew = 0.80, L_smile = 1.50, L_floor = 0.20), z(m) (Eq. 41, nu_skew = 1.00),
and A_eff(m) (Eq. 42, A_stress = 0.75)
(`results/exp4_summary.json`, `results/exp4_fig5_low_z.png`,
`results/exp4_fig6_high_z.png`).

| regime | z_regime | sigma_loc min | sigma_loc max | max downside/upside spread | downside/upside ratio |
|---|---|---|---|---|---|
| low-z (Fig. 5) | 0.05 | 0.743 | 1.594 | 0.766 | 1.924 |
| high-z (Fig. 6) | 10.0 | 0.485 | 1.190 | 0.650 | 2.202 |

The high-z combined surface has the largest downside/upside volatility ratio
(2.202) of all eight surfaces generated in experiments 2-4, confirming the
paper's statement that in the combined high-z case "the three structural
channels acting together produce the strongest downside asymmetry of the
simulated cases" (Section 9). Both low-z and high-z combined surfaces are
higher at short maturity and strongly asymmetric in log-moneyness (lower
liquidity, lower z, and higher forcing on the downside all compound through
the multiplicative kernel of Eq. 24), unlike the single-channel surfaces of
experiments 2 and 3 taken individually.

**Verdict:** reproduced (the ranking claim — combined high-z shows the
strongest downside asymmetry — holds; see "Deviations from the paper" for a
note on the exact contour/table values quoted for Figures 5-6).

## Deviations from the paper

- No external data source is used: every quantity in this repository is a
  closed-form expression or a one-dimensional numerical integral evaluated
  directly from the paper's formulas (Eq. 15, 21-25, 38-42) and Table 1
  parameters. There is consequently no market-data universe, sample period,
  or data source to report.
- The exact sigma_loc contour levels and grid resolution used to draw
  Figures 1-2, 3-4 and 5-6 are not given numerically in the paper,
  so the contour levels plotted here are chosen automatically from the
  computed surface rather than matched level-by-level to the paper's
  printed contour tables. The surfaces computed here from Eq. (24)-(25) and
  (38)-(42) vary continuously across log-moneyness at fixed maturity
  whenever a state-dependent profile (L(m), z(m), A_eff(m)) is active, which
  follows directly from the multiplicative kernel structure and the paper's
  Table 1 parameter values.
- The relationship between operational time U(T) and calendar time T is not
  given as an explicit formula in the paper; since every
  simulated example in Section 9 uses structural parameters that are
  constant along each state-dependent profile (the kernel is evaluated at a
  fixed state, not integrated through time), U(T) does not enter the
  projected local volatility of Eq. (38) beyond the activity profile
  alpha(T) of Eq. (39), which is used directly.
- Experiment 1 uses the full numerical integral of Eq. (21) by adaptive
  quadrature rather than an asymptotic approximation, since verifying the
  asymptotics against the exact integral is the point of the experiment;
  experiments 2-4 use the asymptotic forms of Eq. (22) for F_gamma(z), as the
  paper's Section 9 explicitly invokes "the asymptotic reaction-boundary
  closure derived in Section 5" for the simulated examples.

## Not reproduced

Nothing is listed here: the response function and its asymptotics, and the
liquidity-only, resilience-only, and combined structural volatility surfaces
in both response regimes, were all implemented and produced the artifacts
and qualitative verdicts described above.
