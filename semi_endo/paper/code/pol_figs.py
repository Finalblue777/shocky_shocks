# Figures for Section 6 of the paper from the monthly runs (monthly/s6_*.pkl, monthly/asym_*.pkl).
# Red = time to build, black = no time to build; solid = optimal rate, dotted = peg.
import glob, os, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SP = os.path.dirname(os.path.abspath(__file__)); M = SP + '/monthly'
FIG = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'slides', 'figs')
R = {}
for f in glob.glob(M + '/s6_*.pkl'):
    r = pickle.load(open(f, 'rb')); R[(r['ttb'], r['case'][:3], r['K'], r['when'], r.get('lock', False))] = r
A = {}
for f in glob.glob(M + '/asym_*.pkl'):
    r = pickle.load(open(f, 'rb')); A[(r['case'][:3], r['K'])] = r
rho = R[(True, 'TFP', 0, 'news', False)]['ipath'][-1]
ann = lambda i: 100 * ((1 + np.asarray(i)) ** 12 - 1)
RED, BLK = '#d00000', '#000000'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def infl(P):                       # log-deviation path of P (in %), P before the first date = s.s.
    return np.diff(np.r_[0.0, P])

def shock_fig(nm, K, fname, t_lo):
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.6))
    for ttb, col in ((True, RED), (False, BLK)):
        r = R[(ttb, nm, K, 'news', False)]; T = len(r['Y']['peg']); t = np.arange(T) - K
        keep = (t >= t_lo) & (t <= 13)
        for pol, ls, lw in (('opt', '-', 2.4), ('peg', ':', 2.0)):
            ax[0, 0].plot(t[keep], r['Y'][pol][keep], ls, color=col, lw=lw)
            ax[0, 1].plot(t[keep], r['Lab'][pol][keep], ls, color=col, lw=lw)
            ax[1, 0].plot(t[keep], infl(r['P'][pol])[keep], ls, color=col, lw=lw)
        ax[1, 1].step(t[keep], ann(r['ipath'])[keep], where='post', color=col, lw=2.4)
    ax[1, 1].axhline(ann(rho), color='0.4', ls=':', lw=1.6)
    for a, ttl in zip(ax.flat, ('consumption $C_t$ (% from s.s.)', 'employment $L_t$ (% from s.s.)',
                                'inflation $\\pi_t$ (% per month)', 'nominal rate $i_t$ (% a year)')):
        a.set_title(ttl, fontsize=11, loc='left'); a.axhline(0, color='0.75', lw=0.8); a.axvline(0, color='0.85', lw=0.8)
        a.set_xlabel('months from shock')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=RED, lw=2.4), Line2D([], [], color=BLK, lw=2.4),
         Line2D([], [], color='0.3', lw=2.4), Line2D([], [], color='0.3', lw=2.0, ls=':')]
    fig.legend(h, ['time to build', 'no time to build', 'optimal rate', 'peg'], loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f'{FIG}/{fname}', dpi=200); plt.close(fig)

shock_fig('TFP', 0, 'pol_unexp_tfp.png', 0)
shock_fig('dem', 0, 'pol_unexp_dem.png', 0)
shock_fig('TFP', 13, 'pol_exp_tfp.png', -13)
shock_fig('dem', 13, 'pol_exp_dem.png', -13)

import sys
if len(A) < 12: print('asym runs missing; skipping horizon figure'); sys.exit(0)
# gains by news horizon: everyone (TTB, no TTB), commitment, planner only
KS = [0, 1, 2, 3, 5, 13]
def g13(r): return r['gain13']
def asym13(nm, K): return A[(nm, K)]['gain'] * (1 + rho) ** (K - 13)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, nm, sc, ttl in ((ax[0], 'TFP', 1, '3% productivity fall'), (ax[1], 'dem', 1e3, '1% demand fall ($\\times10^{-3}$)')):
    x = np.arange(len(KS))
    a.plot(x, [sc * g13(R[(True, nm, K, 'news', False)]) for K in KS], 'o-', color=RED, lw=2.2, label='time to build, everyone informed')
    a.plot(x, [sc * (g13(R[(True, nm, K, 'news', True)]) if K else g13(R[(True, nm, 0, 'news', False)])) for K in KS],
           's--', color=RED, mfc='white', lw=1.8, label='time to build, production committed')
    a.plot(x, [sc * (asym13(nm, K) if K else g13(R[(True, nm, 0, 'news', False)])) for K in KS],
           '^:', color=RED, lw=1.8, label='time to build, planner only')
    a.plot(x, [sc * g13(R[(False, nm, K, 'news', False)]) for K in KS], 'o-', color=BLK, lw=2.2, label='no time to build')
    a.set_xticks(x); a.set_xticklabels([str(k) for k in KS]); a.set_xlabel('news horizon $k$ (months; 0 = unexpected)')
    a.set_title(ttl, fontsize=11, loc='left'); a.set_ylim(bottom=0)
ax[0].set_ylabel('gain, % of a month of s.s. consumption')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f'{FIG}/pol_horizon.png', dpi=200); plt.close(fig)
print('figures saved; rho =', rho)
