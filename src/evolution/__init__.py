# Phase 2: Factor Evolution System
# AlphaGo-style genetic search over factor space
#
# Modules (to be built):
#   gene.py        — factor gene representation (operator tree)
#   mutate.py      — mutation operators (param shift, field swap, window change)
#   crossover.py   — factor crossover (linear/nonlinear combination)
#   fitness.py     — backtest-based fitness: IC_IR, Sharpe, decorrelation reward
#   population.py  — generation management, survival selection
#   evolve.py      — main evolutionary loop
