#!/bin/zsh
# Reproduce every number and figure in the paper, the policy slide decks and the union-wage note.
# Total run time: about 3 hours on a 16-core laptop (most of it the policy optimisations).
# RATE (monthly steady-state nominal rate) defaults to 4% a year; e.g. RATE=0.0025 ./run_all.sh
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")"
PY=${PY:-/Users/samuelzhao/anaconda3/bin/python}
mkdir -p monthly union_runs slides_data
# 1. competitive economy (Sections 2-4): calibration, first-order checks, propagation figures
$PY calib_stats.py      > logs_calib.txt
$PY nodebt.py           > logs_firstorder.txt
$PY ce_irf.py           > logs_propagation.txt
$PY dev_where.py        > logs_retiming.txt
# 2. optimal policy (Section 6): main experiments, commitment case, planner-only case, extra horizons
$PY run_sec6.py         > logs_sec6.txt
$PY asym.py monthly     > logs_asym.txt
$PY run_sec6_more.py    > logs_sec6_more.txt
$PY info_tab.py         > logs_info.txt
$PY pol_figs.py
# 3. slide decks (ttb_policy_slides, ttb_irf_slides)
$PY slides_monthly.py   > logs_slides.txt
# 4. union-wage robustness note
$PY run_union.py        > logs_union.txt
$PY union_note.py       > logs_union_note.txt
echo "done"
