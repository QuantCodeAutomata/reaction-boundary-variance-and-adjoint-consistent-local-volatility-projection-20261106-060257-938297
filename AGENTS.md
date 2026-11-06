# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A reproduction of selected numerical experiments from "Reaction-boundary
variance and adjoint-consistent local-volatility projection": the
dimensionless spectral response function `F_gamma(z)` (paper Section 5) and
the projected structural local-volatility surfaces built from the
operational variance kernel (paper Section 6 and Section 9). All quantities
are evaluated from closed-form formulas and a one-dimensional numerical
integral; no market data or synthetic data generation is used.

## Layout

```
src/                       Flat library modules (no subdirectories)
  config.py                All experiment parameters, UPPER_CASE, paper-sourced
  response_function.py     F_gamma(z), Eq. (21)-(23): quadrature + asymptotes
  surface.py                Operational kernel and structural profiles, Eq. (24)-(25), (38)-(42)
  plotting.py               Shared 3D surface / 2D contour plot helpers
exp/                       One script per experiment, using src/ only
  exp1_response_function.py
  exp2_liquidity_only_surfaces.py
  exp3_resilience_only_surfaces.py
  exp4_combined_structural_surfaces.py
tests/                     pytest unit tests for src/ components
results/                   Figures, JSON summaries, RESULTS.md (overwritten by reruns)
data/                      Empty package: no loader or generator is needed here
```

## Commands

Setup:

```bash
python -m pip install -r requirements.txt
```

Tests (no market-data marker is used in this repository, since no experiment
loads market data):

```bash
python -m pytest
```

Experiments (paper's exact baseline configuration from Table 1; there is no
reduced configuration, since every run is a closed-form/quadrature
computation that finishes in well under a minute):

```bash
python -m exp.exp1_response_function
python -m exp.exp2_liquidity_only_surfaces
python -m exp.exp3_resilience_only_surfaces
python -m exp.exp4_combined_structural_surfaces
```

`./reproduce.sh` installs dependencies, runs the tests, and runs all four
experiments in sequence.

## Conventions

- All numeric parameters live in `src/config.py` as module-level UPPER_CASE
  variables with a comment on meaning, unit and paper source; no classes or
  dictionaries.
- `src/` contains only library code; experiment scripts live in `exp/` and
  call `src/` functions with their documented signatures; they do not
  reimplement the kernel or the response function.
- `data/` is kept as an empty package because no experiment in this
  repository loads market data or needs generated synthetic data: every
  input is a deterministic grid of dimensionless parameters from Table 1.
- No API keys are used anywhere in this repository.
- After a rerun, the experiment commands overwrite the figures and JSON
  summaries in `results/`; the prose, numbers and verdicts in
  `results/RESULTS.md` must then be checked against the new files and
  updated by hand.
