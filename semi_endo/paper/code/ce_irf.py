# Shock propagation in the no-debt competitive economy (labour numeraire, fixed durations).
# Produces figures for Section 4 of the paper and the propagation table.
import json, io, contextlib, os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import root
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SP = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'slides', 'figs')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nbsrc
g = {}
with contextlib.redirect_stdout(io.StringIO()):
    for k in (1, 3): exec(nbsrc.cell(k), g)
TV0, ZETA, NARR, kappa = g['TVALS'], g['ZETA'], np.asarray(g['N_ARRAY']), g['c']
eps, sig, eta, chi, rho = g['sigma'], g['gamma'], g['eta'], g['chi'], g['rho']; delta = 1 / (1 + rho)

def economy(TV):
    NF = len(TV); TM = int(TV.max())
    z = delta ** TV * ZETA; At = np.sum(z ** (eps - 1)) ** (1 / (eps - 1))
    Th = At ** eps / np.sum(z ** (eps - 1) * delta ** TV)
    L = (At / (chi * Th ** sig)) ** (1 / (sig + eta)); Y = Th * L
    p = 1 / z; P = 1 / At; y = (z * P) ** eps * Y
    SS = dict(L=L, Y=Y, P=P, p=p, y=y, l=y / ZETA, Pi=Y - L / P)
    def solve(T, i_path, A_path):
        pad = TM + 2
        ix = np.r_[np.asarray(i_path, float), np.full(pad, rho)]; Ax = np.r_[np.asarray(A_path, float), np.ones(pad)]
        cum = np.r_[0.0, np.cumsum(np.log1p(ix))]
        t = np.arange(T)[:, None]; h = t - TV[None, :]; FREE = h >= 0; hc = np.maximum(h, 0)
        pc = np.exp(cum[t] - cum[hc]) / (ZETA[None, :] * Ax[hc])
        Q = np.sum(np.where(FREE, pc ** (1 - eps), 0), 1)
        S = np.sum(np.where(~FREE, y[None, :] ** ((eps - 1) / eps), 0), 1); Yflr = S ** (eps / (eps - 1))
        def parts(Yv):
            P_ = (Q / (1 - S * Yv ** ((1 - eps) / eps))) ** (1 / (1 - eps))
            yy = np.where(FREE, (pc / P_[:, None]) ** (-eps) * Yv[:, None], y[None, :])
            pp = np.where(FREE, pc, P_[:, None] * (Yv[:, None] / y[None, :]) ** (1 / eps))
            yx = np.vstack([yy, np.tile(y, (pad, 1))])
            l = yx[t + TV[None, :], np.arange(NF)] / (ZETA * Ax[:T, None])
            return P_, yy, pp, l
        def res(u):
            Yv = Yflr + np.exp(u); P_, _, _, l = parts(Yv)
            return np.log(chi) + eta * np.log(l.sum(1)) + sig * np.log(Yv) + np.log(P_)
        s = root(res, np.log(np.maximum(Y - Yflr, 1e-9)), method='hybr', tol=1e-14)
        Yv = Yflr + np.exp(s.x); P_, yy, pp, l = parts(Yv)
        assert np.abs(res(s.x)).max() < 1e-9
        return dict(Y=Yv, P=P_, L=l.sum(1), l=l, y=yy, p=pp, Pi=Yv - l.sum(1) / P_, i=ix, A=Ax)
    return dict(TV=TV, SS=SS, solve=solve, s=y * p / (P * Y), lam=(y / ZETA) / L)

E = {'ttb': economy(TV0), 'none': economy(np.zeros_like(TV0))}

def deviation_incentive(e, sol, s_range):
    """max over Delta != tau_j of R_{j,s}(Delta) - 1, for each producer j and start date s."""
    TV = e['TV']; out = np.full((len(s_range), len(TV)), -np.inf)
    lcum = np.r_[0.0, np.cumsum(np.log1p(sol['i']))]
    for a, s in enumerate(s_range):
        for D in range(0, 20):
            pd_ = sol['p'][s + D]                                          # price of each good at s+D
            R = pd_ * np.exp(-(lcum[s + D] - lcum[s])) * kappa ** D * (D + NARR) * sol['A'][s]
            out[a] = np.where(D != TV, np.maximum(out[a], R - 1), out[a])
    return out

T = 160; t0 = 50; H = np.arange(-8, 15)
shocks = {'tfp': dict(A=0.01, i=0.0), 'rate': dict(A=0.0, i=0.01 / 12)}
res = {}
for sk, sh in shocks.items():
    for ek, e in E.items():
        base = e['solve'](T, np.full(T, rho), np.ones(T))
        for when, ts in (('ant', t0), ('unexp', 0)):
            A = np.ones(T); ip = np.full(T, rho); A[ts] *= 1 + sh['A']; ip[ts] += sh['i']
            sol = e['solve'](T, ip, A); SS = e['SS']
            r = dict(Y=100 * np.log(sol['Y'] / base['Y']), L=100 * np.log(sol['L'] / base['L']),
                     P=100 * np.log(sol['P'] / base['P']), Pi=100 * (sol['Pi'] - base['Pi']) / SS['Y'], ts=ts, sol=sol)
            if ek == 'ttb':
                inc = deviation_incentive(e, sol, range(max(ts - 15, 0), ts + 15))
                r['inc_max'] = inc.max(); r['inc_share'] = e['s'][(inc > 0).any(0)].sum()
            res[sk, ek, when] = r
            print(sk, ek, when, 'impact Y %.3f  cumY %.3f  cumL %.3f  peakY %.3f' % (
                r['Y'][ts], r['Y'].sum(), r['L'].sum(), r['Y'][np.argmax(np.abs(r['Y']))]),
                ('  incentive max %.3f%%, expenditure share wanting to deviate %.1f%%' % (100 * r['inc_max'], 100 * r['inc_share'])) if ek == 'ttb' else '')
# small-shock check of invariance region
e = E['ttb']; base = e['solve'](T, np.full(T, rho), np.ones(T))
for a in (1e-4, 1e-3, 3e-3, 1e-2):
    A = np.ones(T); A[t0] *= 1 + a; sol = e['solve'](T, np.full(T, rho), A)
    inc = deviation_incentive(e, sol, range(t0 - 15, t0 + 15))
    print('TFP %.2f%%: expenditure share with a profitable deviation %.1f%%' % (100 * a, 100 * e['s'][(inc > 0).any(0)].sum()))

# by-lag decomposition (TTB, anticipated and unexpected TFP)
e = E['ttb']; TV = e['TV']; groups = [(1,), (2,), (3,), tuple(range(4, 14))]; gl = ['1', '2', '3', r'$\geq$4']
for when in ('ant', 'unexp'):
    r = res['tfp', 'ttb', when]; sol = r['sol']; SS = e['SS']; ts = r['ts']
    r['hire_by_lag'] = {k: 100 * (sol['l'][:, np.isin(TV, gg)].sum(1) - SS['l'][np.isin(TV, gg)].sum()) / SS['L'] for k, gg in zip(gl, groups)}
    r['deliv_by_lag'] = {k: 100 * (np.sum(SS['p'][np.isin(TV, gg)] * sol['y'][:, np.isin(TV, gg)], 1)
                                   - np.sum(SS['p'][np.isin(TV, gg)] * SS['y'][np.isin(TV, gg)])) / (SS['P'] * SS['Y']) for k, gg in zip(gl, groups)}
pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != 'sol'} for k, v in res.items()}, open(SP + '/monthly/ce_irf.pkl', 'wb'))

# ---------------- figures ----------------
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'ant': '#c0392b', 'unexp': 'black', 'none': '#7f8c8d'}
def irf_fig(sk, fname, ttl):
    fig, ax = plt.subplots(2, 2, figsize=(9, 5.6))
    for a, (k, lab) in zip(ax.flat, [('Y', 'Output = consumption (%)'), ('L', 'Hours (%)'),
                                      ('P', 'Price level in wage units (%)'), ('Pi', r'Profits (% of s.s. output)')]):
        for when, ls in (('ant', '-'), ('unexp', '--')):
            r = res[sk, 'ttb', when]; ts = r['ts']; idx = ts + H; ok = idx >= 0
            a.plot(H[ok], r[k][idx[ok]], ls, color=COL[when], lw=2, marker='o', ms=3,
                   label={'ant': 'time to build, anticipated', 'unexp': 'time to build, unexpected'}[when])
        r = res[sk, 'none', 'unexp']; ok = H >= 0
        a.plot(H[ok], r[k][H[ok]], ':', color=COL['none'], lw=2, marker='s', ms=3, label='no time to build')
        a.axhline(0, color='0.6', lw=0.6); a.axvline(0, color='0.85', lw=0.6); a.set_title(lab, fontsize=10)
        a.set_xlabel('periods from shock')
    ax[0, 0].legend(frameon=False, fontsize=8); fig.suptitle(ttl, fontsize=11); fig.tight_layout()
    fig.savefig(f'{FIG}/{fname}', dpi=200); plt.close(fig)
irf_fig('tfp', 'ce_tfp.png', 'One-month 1% rise in productivity')
irf_fig('rate', 'ce_rate.png', 'One-month rise in the interest rate of 1 pp (annualised)')

fig, ax = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
LC = ['#c0392b', '#2471a3', '#27ae60', '#8e44ad']
for col, when in enumerate(('ant', 'unexp')):
    r = res['tfp', 'ttb', when]; ts = r['ts']; idx = ts + H; ok = idx >= 0
    for row, (k, lab) in enumerate((('hire_by_lag', 'Hours hired, by lag (% of s.s. hours)'),
                                    ('deliv_by_lag', 'Deliveries, by lag (% of s.s. output)'))):
        a = ax[row, col]
        for c_, gk in zip(LC, gl):
            a.plot(H[ok], r[k][gk][idx[ok]], color=c_, lw=2, marker='o', ms=3, label=(r'$\tau_j\geq4$' if gk.endswith('4') else r'$\tau_j=' + gk + '$'))
        a.axhline(0, color='0.6', lw=0.6); a.axvline(0, color='0.85', lw=0.6)
        a.set_title(('Anticipated: ' if when == 'ant' else 'Unexpected: ') + lab, fontsize=9)
        if row == 1: a.set_xlabel('periods from shock')
ax[0, 0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f'{FIG}/ce_tfp_bylag.png', dpi=200); plt.close(fig)

e = E['ttb']; TM = e['TV'].max()
fig, ax = plt.subplots(figsize=(6.5, 3))
k = np.arange(TM + 1); w = 0.4
ax.bar(k - w / 2, 100 * np.bincount(e['TV'], weights=e['s'], minlength=TM + 1), w, color='black', label=r'expenditure, $s_j$')
ax.bar(k + w / 2, 100 * np.bincount(e['TV'], weights=e['lam'], minlength=TM + 1), w, color='#c0392b', label=r'employment, $\lambda_j$')
ax.bar(k - w / 2, 0, 0)
ax2 = ax.twinx(); ax2.plot(k, np.bincount(e['TV'], minlength=TM + 1), 'o', color='#7f8c8d', ms=4, label='producers (right)')
ax2.spines['top'].set_visible(False); ax.set_xticks(k); ax.set_xlabel(r'production lag $\tau_j$ (periods)'); ax.set_ylabel('share (%)')
ax2.set_ylabel('number of producers')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/ce_lags.png', dpi=200); plt.close(fig)
print('figures saved')
