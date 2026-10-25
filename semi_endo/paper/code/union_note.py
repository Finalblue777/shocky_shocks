# Figures and table for the union-wage robustness note: fixed wage vs Rotemberg unions (12- and 3-month contracts).
import glob, os, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
SP = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'slides', 'figs')
F = {}
for f in glob.glob(SP + '/monthly/s6_*.pkl'):
    r = pickle.load(open(f, 'rb')); F[('fixed', r['ttb'], r['case'][:3], r['K'], r['when'], r.get('lock', False))] = r
for f in glob.glob(SP + '/union_runs/u*.pkl'):
    r = pickle.load(open(f, 'rb')); F[(f"u{int(r['dur'])}", r['ttb'], r['case'][:3], r['K'], r['when'], r.get('lock', False))] = r
rho = F[('fixed', True, 'TFP', 0, 'news', False)]['ipath'][-1]
ann = lambda i: 100 * ((1 + np.asarray(i)) ** 12 - 1)
WAGES = [('fixed', 'fixed wage'), ('u12', 'union, 12-month contracts'), ('u3', 'union, 3-month contracts')]

# ---------------------------------------------------------------- table
def share(r): U = r['U']; return (U['opt'] - U['peg']) / (U['flex'] - U['peg'])
def loss(r): return (r['U']['flex'] - r['U']['peg']) * (1 + rho) ** (r['K'] - 13)
CASES = [('Unexpected', 0, 'news'), ('News 1 month ahead', 1, 'news'), ('News 13 months ahead', 13, 'news'),
         ('News 13 months ahead, policy from $t=0$', 13, 'shock')]
lines = []
for nm, lab, sc in (('TFP', '3\\% productivity fall', 1), ('dem', '1\\% demand fall ($\\times10^{-3}$)', 1e3)):
    lines.append(f"    \\multicolumn{{7}}{{l}}{{\\emph{{{lab}}}}}\\\\")
    for cl, K, w in CASES:
        cells = []
        for ttb in (True, False):
            for wg, _ in WAGES:
                r = F.get((wg, ttb, nm, K, w, False))
                cells.append('--' if r is None else (f"{sc * r['gain13']:.3f}" if sc == 1 else f"{sc * r['gain13']:.2f}"))
        lines.append(f"    {cl} & " + ' & '.join(cells) + ' \\\\')
    lines.append('    \\midrule')
open(SP + '/union_table.tex', 'w').write('\n'.join(lines[:-1]) + '\n')
# optimal rate on impact and share removed, unexpected and 13 months ahead
rows2 = []
for nm in ('TFP', 'dem'):
    for ttb in (True, False):
        for wg, _ in WAGES:
            r0 = F[(wg, ttb, nm, 0, 'news', False)]; r13 = F[(wg, ttb, nm, 13, 'news', False)]
            rows2.append((nm, ttb, wg, ann(r0['ipath'][0]), 100 * share(r0), loss(r0), ann(r13['ipath'][12]), ann(r13['ipath'][13]), 100 * share(r13)))
pickle.dump(rows2, open(SP + '/union_rows2.pkl', 'wb'))
for r in rows2: print('%s %s %-5s i0 %6.2f share %5.1f%% loss %.4f | k13: i-1 %6.2f i0 %6.2f share %5.1f%%' % (r[0], 'TTB ' if r[1] else 'none', r[2], *r[3:]))
print('commitment check (u12):', {nm: [(K, round(F[('u12', True, nm, K, 'news', True)]['gain13'], 5)) for K in (1, 13) if ('u12', True, nm, K, 'news', True) in F] for nm in ('TFP', 'dem')},
      'unexpected', {nm: round(F[('u12', True, nm, 0, 'news', False)]['gain13'], 5) for nm in ('TFP', 'dem')})

# ---------------------------------------------------------------- figure: unexpected TFP, rate path and wages
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'fixed': '#000000', 'u12': '#d00000', 'u3': '#2a78d6'}; LS = {True: '-', False: ':'}
def fig(nm, K, fname, t_lo):
    f, ax = plt.subplots(2, 2, figsize=(10, 5.6))
    for wg, _ in WAGES:
        for ttb in (True, False):
            r = F[(wg, ttb, nm, K, 'news', False)]; T = len(r['ipath']); t = np.arange(T) - K; keep = (t >= t_lo) & (t <= 10)
            ax[0, 0].step(t[keep], ann(r['ipath'])[keep], where='post', color=COL[wg], ls=LS[ttb], lw=2.2)
            ax[0, 1].plot(t[keep], r['Y']['opt'][keep], color=COL[wg], ls=LS[ttb], lw=2.2)
            ax[1, 0].plot(t[keep], r['Lab']['opt'][keep], color=COL[wg], ls=LS[ttb], lw=2.2)
            W = r.get('W', {}).get('opt') if wg != 'fixed' else np.zeros(T)
            if W is not None: ax[1, 1].plot(t[keep], W[keep], color=COL[wg], ls=LS[ttb], lw=2.2)
    for a, ttl in zip(ax.flat, ('optimal nominal rate (% a year)', 'consumption (% from s.s.), optimal rate',
                                'employment (% from s.s.), optimal rate', 'nominal wage (% from s.s.), optimal rate')):
        a.set_title(ttl, fontsize=10.5, loc='left'); a.axhline(0, color='0.8', lw=0.8); a.axvline(0, color='0.85', lw=0.8)
        a.set_xlabel('months from shock')
    h = [Line2D([], [], color=COL[w], lw=2.2, label=l) for w, l in WAGES] + \
        [Line2D([], [], color='0.3', lw=2.2, label='time to build'), Line2D([], [], color='0.3', lw=2.2, ls=':', label='no time to build')]
    f.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9)
    f.tight_layout(rect=(0, 0.06, 1, 1)); f.savefig(f'{FIG}/{fname}', dpi=200); plt.close(f)
fig('TFP', 0, 'uw_unexp_tfp.png', 0)
fig('dem', 0, 'uw_unexp_dem.png', 0)
fig('TFP', 13, 'uw_exp_tfp.png', -6)
print('figures saved')
