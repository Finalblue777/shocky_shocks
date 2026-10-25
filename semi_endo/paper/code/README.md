# Code for *Time to Build and the Macroeconomy*

These scripts produce every number and figure in:
- `../time_to_build_policy.tex` (the paper);
- `../union_wages.tex` (union-wage robustness note);
- `../../slides/ttb_policy_slides.tex` and `../../slides/ttb_irf_slides.tex`.

Figures are written to `../../slides/figs/`, which both the paper and the slides read.

Run everything with `./run_all.sh`. It takes about 3 hours on a 16-core laptop, almost all of it the
policy optimisations. Use the Anaconda interpreter; the system Python lacks numpy and pandas.

## The model

All model code is built on `../../fixed_wage_cohorts.ipynb`. `nbsrc.py` loads its calibration and
solver cells and overrides one thing: the steady-state interest rate.

- **Rate:** set by the environment variable `RATE`. The default is 4% a year, i.e. `1.04**(1/12) - 1`
  per month. Changing it re-calibrates κ and every N_j, so all results must be rerun.
- **Period:** one month of 30 days. The inventory-to-COGS data are converted to months as
  `365 * ratio / 30` (see the paper's appendix "Discretisation and calibration").
- **κ:** calibrated to Antràs–Tubdenov (2025, Table 5). Their coefficient of −0.090 means a 30-day
  change in durations corresponds to a change of `(30/365)/0.090 = 0.913` in the log rate.

`exp_horizon.model(ttb, K, lock)` builds the fixed-wage cohort economy:
- with time to build (`ttb=True`) or without it (`ttb=False`, all τ_j = 0);
- with news `K` months ahead;
- optionally with production locked as if the shock were unexpected (`lock=True`).

## Scripts, by section

| Script | Produces | Time |
|---|---|---|
| `calib_stats.py` | Calibration numbers (Section 3, Table 1); duration gaps (Section 4.3) | seconds |
| `nodebt.py` | Competitive economy without debt; checks Corollary 1 against the nonlinear model (Section 2.8) | ~1 min |
| `ce_irf.py` | Section 4 figures (`ce_*.png`), Table 2, re-timing incentives | ~5 min |
| `dev_where.py` | Which batches would re-time (Section 4.3) | ~2 min |
| `run_sec6.py` | Section 6 policy runs (k = 0, 1, 2, 3, 5, 13; policy from news or shock date; commitment case) → `monthly/s6_*.pkl` | ~45 min |
| `asym.py monthly` | Planner-only case → `monthly/asym_*.pkl` | ~10 min |
| `run_sec6_more.py` | Remaining news horizons for the slides (k = 4, 6–12) → `monthly/s6_*.pkl` | ~40 min |
| `info_tab.py` | Table 4 and the "who should know" numbers | ~1 min |
| `pol_figs.py` | Section 6 figures (`pol_*.png`) | seconds |
| `slides_monthly.py` | Every figure in the two policy decks (patches and runs `ttb_plots_v2.py` and `asym_fig.py`) | ~10 min |
| `union.py`, `run_union.py` | Rotemberg union wages, 12- and 3-month contracts → `union_runs/u*_*.pkl` | ~60 min |
| `union_note.py` | Union note figures (`uw_*.png`) and table numbers | seconds |

`monthly/` and `union_runs/` hold the saved results from the current runs. With them in place, the
figure scripts (`pol_figs.py`, `slides_monthly.py`, `union_note.py`, `info_tab.py`) work without
rerunning the optimisations.

## Notes

- **Threads:** set `OMP_NUM_THREADS=1` (as `run_all.sh` does). Without it, each pool worker starts its
  own BLAS threads and the machine thrashes.
- **Optimiser:** policy optima use L-BFGS-B over the free rates, with bounds [0, 5%] a month and finite
  differences. Two starting points are used: the peg and a flat 0.5% a month. The union runs start from
  the fixed-wage optimum.
- **Welfare units:** % of one month's steady-state consumption. Gains are valued at t = −13 by
  discounting at the steady-state rate.
- **Old notebooks:** `new_fixed_planner*.ipynb` in `semi_endo/` apply κ once instead of κ^τ. Do not use
  them for these results.
