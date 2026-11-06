# Reaction-boundary variance and adjoint-consistent local-volatility projection

Replication of selected numerical experiments from **"Reaction-boundary
variance and adjoint-consistent local-volatility projection"**.

This repository reproduces the purely mathematical/numerical results of the
paper: the dimensionless spectral response function `F_gamma(z)` (Section 5)
and the projected structural local-volatility surfaces built from it
(Section 9). No market data is used: every quantity is a closed-form or
numerically-integrated function of dimensionless parameters taken from the
paper's Table 1.

## What is reproduced

| Experiment | Paper reference | Script |
|---|---|---|
| `exp_1` Computation and asymptotic verification of `F_gamma(z)` | Section 5, Eq. (15), (21)-(23) | `python -m exp.exp1_response_function` |
| `exp_2` Liquidity-only volatility surfaces, low-z / high-z (Figures 1-2) | Section 6 Eq. (24)-(25); Section 9 Eq. (38)-(42) | `python -m exp.exp2_liquidity_only_surfaces` |
| `exp_3` Resilience-response-only volatility surfaces, low-z / high-z (Figures 3-4) | Section 9 | `python -m exp.exp3_resilience_only_surfaces` |
| `exp_4` Combined structural volatility surfaces, low-z / high-z (Figures 5-6) | Section 9 | `python -m exp.exp4_combined_structural_surfaces` |

## Repository layout

```
src/            Library modules: response_function.py, surface.py, config.py, plotting.py
exp/            One script per experiment, calling only src/ functions
tests/          pytest unit tests for src/ components
results/        Figures, JSON summaries, and RESULTS.md (overwritten on rerun)
data/           Empty package: no market data or synthetic data generation is needed
```

## Setup

```bash
python -m pip install -r requirements.txt
```

No API keys are required: every experiment here is a closed-form or
numerically-integrated computation from the paper's formulas and Table 1
parameters, with no external data dependency.

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1_response_function
python -m exp.exp2_liquidity_only_surfaces
python -m exp.exp3_resilience_only_surfaces
python -m exp.exp4_combined_structural_surfaces
```

or, equivalently:

```bash
./reproduce.sh
```

Each run overwrites the figures and JSON summaries in `results/`. See
`results/RESULTS.md` for the comparison against the paper's reported values.
