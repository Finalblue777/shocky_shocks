# Notebook source for the fixed-wage cohort model with the interest rate set from the environment.
# RATE = monthly nominal rate in steady state (default: 4% a year, compounded monthly).
import json, os
RATE = float(os.environ.get('RATE', 1.04 ** (1 / 12) - 1))
NB = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'fixed_wage_cohorts.ipynb')
_d = json.load(open(NB))
def cell(k):
    s = ''.join(_d['cells'][k]['source'])
    if k == 3:
        old = 'NOMINAL_INTEREST_RATE     = 0.01'
        assert old in s
        s = s.replace(old, f'NOMINAL_INTEREST_RATE     = {RATE!r}')
    return s
