# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research repo (not an application) for a New Keynesian model with **time-to-produce price stickiness**: firms hold inventory, so each firm has a production lag `tau` calibrated from industry inventory/COGS data, and price adjustment is staggered by that lag rather than by Calvo draws. Output is ~200 PNG figures at the repo root plus ~50 Jupyter notebooks. There is no package, no build, no test suite, and no CI.

## Running notebooks

Use the Anaconda interpreter — the system `python3` at `/Library/Frameworks/Python.framework` does not have pandas/numpy installed:

```bash
/Users/samuelzhao/anaconda3/bin/jupyter lab
/Users/samuelzhao/anaconda3/bin/python -c "..."   # for scratch checks
```

**Notebooks must be run with the working directory set to the notebook's own folder.** They load data with a bare relative path (`pd.read_csv('APP_data_usa_goods.csv', ...)`), and `ss/` notebooks additionally do `from economic_utils import ...`. `APP_data_usa_goods.csv` is deliberately duplicated into each folder that needs it (root, `demand_stuff/`, `semi_endo/`; the `ss/` copy is currently deleted in the working tree). If you add a notebook to a folder, copy the CSV in too rather than reaching up with `../`.

Figures are saved to the **repo root** regardless of which folder the notebook lives in, under short ad-hoc names (`A_exo.png`, `pct3.png`, `zzzz_corrected_plot.png`). Names are frequently reused across notebooks — check before overwriting.

## Code organization

Only one real module exists: [ss/economic_utils.py](ss/economic_utils.py). It is imported by just two notebooks ([ss/tip_wage_rigidity.ipynb](ss/tip_wage_rigidity.ipynb), [ss/demand_shock.ipynb](ss/demand_shock.ipynb)). Everywhere else, the same handful of functions is **copy-pasted into each notebook and locally edited**. Treat `economic_utils.py` as one variant among many, not the source of truth — do not "refactor" other notebooks to import from it unless asked, since their copies differ deliberately.

The most important consequence: **the shock magnitude is a hardcoded literal inside `generate_pattern_array_with_extra_rows`** (`A[i, ...] = 0.99` for a 1% shock, `0.95` for 5%, etc.), not a parameter. Notebooks that look identical often differ only in that literal. Similarly, `RISK_AVERSION`, `NOMINAL_INTEREST_RATE`, and other "constants" defined in `economic_utils.py` are routinely shadowed by different values in the notebook cells that import it.

## Model pipeline

Nearly every notebook follows the same sequence; recognizing it makes unfamiliar notebooks fast to read:

1. **Calibration** — load `invtCogsRatio`, scale to days (`* 36.5 / (30 * 0.1)`) to get `data_tau`. Grid-search `c` over `linspace(0, 0.99, 1000)` so that the implied `tau` gap between the two discount factors `beta` and `beta1` equals 1 period. `beta = 1/(1+r)`; `beta1` applies the inventory-duration wedge `exp(30/(36.5*0.9))`.
2. **Time-to-produce distribution** — `N = -data_tau - 1/log(beta*c)` (labor input), `tau = -1/log(beta*c) - N`, clipped at 0, then rounded to whichever of `floor`/`ceil` maximizes `(beta*c)**tau * (tau + N)`. `tau` must end up integer-valued.
3. **Staggering matrix** — `generate_pattern_array_with_extra_rows(tau)` builds a lower-triangular-ish matrix whose diagonal band of shock factors encodes which firms have re-optimized by period `t`.
4. **Prices and aggregates** — firm prices are `markup / (A * tprod)` with `markup = sigma/(sigma-1)`; the CES index is `P0 = (sum(price**(1-sigma), axis=1))**(1/(1-sigma))`. Output is `P0**power` with `power = (1+eta)/(-gamma-eta)`; inflation comes from the Euler equation applied to consecutive `P0`.
5. **Policy experiments** — sweep the nominal rate (`linspace(0, 0.05, 50)`) or enumerate rate paths with `itertools.product` via `generate_rate_combinations`, then pick the sequence minimizing an output/inflation loss.

## Directory map

- `exo/`, `endo/`, `semi_endo/` — the same shocks under exogenous, endogenous, and partially-endogenous labor supply `N`.
- `demand_stuff/` — demand shocks and interest-rate rules; `sticky_*` variants add sticky prices, `fixed_*` hold policy fixed, `free_*` let the rate adjust.
- `ss/` — the `economic_utils.py`-based versions plus wage-rigidity work.
- `unanticipated/` — surprise (non-perfect-foresight) shocks.
- Repo root — older exploratory notebooks (`beta_*.ipynb`, `dist_alpha_*.ipynb`, `shocky.ipynb`, `rando.ipynb`) and all figures.

Notebook and figure names are informal and often near-duplicates (`beta_chang.ipynb` vs `beta_change_recent.ipynb`, `note*`/`notrot*`/`nona*` figure families). When asked to modify "the X notebook", confirm which one if several match — the newest mtime is usually the live one.

## Working conventions

Commit messages in this repo are placeholders (`asdf`, `a`, `g`). Don't infer history from them; use file mtimes and notebook outputs instead.
