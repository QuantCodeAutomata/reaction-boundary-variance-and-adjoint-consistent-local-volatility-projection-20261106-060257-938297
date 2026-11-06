#!/usr/bin/env bash
# Reproduce every experiment of "Reaction-boundary variance and
# adjoint-consistent local-volatility projection" from a clean checkout.
set -uo pipefail

python -m pip install -r requirements.txt

python -m pytest -q

python -m exp.exp1_response_function
python -m exp.exp2_liquidity_only_surfaces
python -m exp.exp3_resilience_only_surfaces
python -m exp.exp4_combined_structural_surfaces

echo "Done. See results/ for figures, JSON summaries, and RESULTS.md."
