# No-debt competitive equilibrium, labour numeraire (W_t = 1), fixed durations, hiring-dated labour.
# Household: chi L^eta C^sigma = 1/P_t (static budget P C = L + Pi).  Producers discount at exogenous i_t.
# Checks the first-order formulas for the paper against the nonlinear model.
import json, io, contextlib, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import root
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nbsrc
g = {}
with contextlib.redirect_stdout(io.StringIO()):
    for k in (1, 3): exec(nbsrc.cell(k), g)
TV, ZETA, eps, sig, eta, chi, rho = g['TVALS'], g['ZETA'], g['sigma'], g['gamma'], g['eta'], g['chi'], g['rho']
SS = g['SS']; NF = len(TV); TM = TV.max(); delta = 1 / (1 + rho)

def solve(T, i_path, A_path):
    i_path = np.asarray(i_path, float); A_path = np.asarray(A_path, float)
    pad = TM + 2
    ix = np.r_[i_path, np.full(pad, rho)]; Ax = np.r_[A_path, np.ones(pad)]
    cum = np.r_[0.0, np.cumsum(np.log1p(ix))]                    # cum[t] = sum_{s<t} log(1+i_s)
    t = np.arange(T)[:, None]; h = t - TV[None, :]                 # hiring date of delivery at t
    FREE = h >= 0; hc = np.maximum(h, 0)
    p = np.exp(cum[t] - cum[hc]) / (ZETA[None, :] * Ax[hc])         # cost-determined prices
    Q = np.sum(np.where(FREE, p ** (1 - eps), 0), 1)
    S = np.sum(np.where(~FREE, SS['y'][None, :] ** ((eps - 1) / eps), 0), 1)
    Yflr = S ** (eps / (eps - 1))
    def parts(Y):
        P = (Q / (1 - S * Y ** ((1 - eps) / eps))) ** (1 / (1 - eps))
        y = np.where(FREE, (p / P[:, None]) ** (-eps) * Y[:, None], SS['y'][None, :])
        yx = np.vstack([y, np.tile(SS['y'], (pad, 1))])
        l = yx[t + TV[None, :], np.arange(NF)] / (ZETA * Ax[:T, None])
        return P, y, l.sum(1)
    def res(u):
        Y = Yflr + np.exp(u); P, _, L = parts(Y)
        return np.log(chi) + eta * np.log(L) + sig * np.log(Y) + np.log(P)
    s = root(res, np.log(np.maximum(SS['Y'] - Yflr, 1e-9)), method='hybr', tol=1e-14)
    Y = Yflr + np.exp(s.x); P, y, L = parts(Y)
    return dict(Y=Y, P=P, L=L, ok=s.success, err=np.abs(res(s.x)).max())

# steady-state weights
s_j = SS['p'] * SS['y'] / (SS['P'] * SS['Y']); lam = SS['l'] / SS['l'].sum()
print('check SS: chi L^eta Y^sig P =', chi * SS['L'] ** eta * SS['Y'] ** sig * SS['P'], ' sum s', s_j.sum(), ' sum lam', lam.sum())
sbar = np.bincount(TV, weights=s_j, minlength=TM + 1); lbar = np.bincount(TV, weights=lam, minlength=TM + 1)
Sbar = np.array([s_j[TV >= k].sum() for k in range(TM + 2)])
print('lambda_0 =', lbar[0], ' mean lag s:', (s_j * TV).sum(), ' mean lag lambda:', (lam * TV).sum())
print('expenditure shares by lag 0..5:', np.round(100 * sbar[:6], 1))

# psi: power series of 1/(sigma + eta Lambda(z))
K = 400; c = np.zeros(K); c[0] = sig
for k in range(TM + 1): c[k] += eta * lbar[k]
psi = np.zeros(K); psi[0] = 1 / c[0]
for k in range(1, K): psi[k] = -np.dot(c[1:k + 1], psi[k - 1::-1][:k]) / c[0]
print('psi[:6]', np.round(psi[:6], 4), ' sum psi', psi.sum(), ' 1/(sig+eta)', 1 / (sig + eta))
roots = np.roots(c[:TM + 1][::-1]); print('min |root| of sigma+eta*Lambda(z):', np.abs(roots).min())

def lin(T, t0, kind, x):
    RHS = np.zeros(T + K); tt = np.arange(T + K)
    if kind == 'A':
        sb = lambda k: np.where((k >= 0) & (k <= TM), sbar[np.clip(k, 0, TM)], 0.0)
        RHS = x * (sb(tt - t0) + eta * (1 - eps) * (tt == t0)
                   + eta * eps * sum(lam[j] * sb(tt - t0 + TV[j]) for j in range(NF)))
    else:
        Sb = lambda k: np.where((k >= 1) & (k <= TM), Sbar[np.clip(k, 0, TM + 1)], 0.0)
        RHS = x * (-Sb(tt - t0) + eta * eps * (sum(lam[j] * ((tt <= t0) & (t0 < tt + TV[j])) for j in range(NF))
                                               - sum(lam[j] * Sb(tt + TV[j] - t0) for j in range(NF))))
    return np.array([np.dot(psi, RHS[t:t + K]) for t in range(T)])

T, t0, x = 140, 60, 1e-4
for kind in ('A', 'i'):
    A = np.ones(T); ip = np.full(T, rho)
    if kind == 'A': A[t0] = np.exp(x)
    else: ip[t0] = np.exp(np.log1p(rho) + x) - 1
    s = solve(T, ip, A); base = solve(T, np.full(T, rho), np.ones(T))
    yn = np.log(s['Y'] / base['Y']) / x; yl = lin(T, t0, kind, 1.0)
    print(f'\n{kind}: ok={s["ok"]} err={s["err"]:.1e}  max|nonlin-lin|={np.abs(yn - yl).max():.2e}')
    print(' Y response t0-4..t0+6 (nonlin):', np.round(yn[t0 - 4:t0 + 7], 4))
    print(' Y response t0-4..t0+6 (lin)   :', np.round(yl[t0 - 4:t0 + 7], 4))
    print(' cumulative nonlin', yn.sum(), ' lin', yl.sum())
    Pn = np.log(s['P'] / base['P']) / x
    print(' P response t0..t0+6:', np.round(Pn[t0:t0 + 7], 4))
    if kind == 'A': print(' predicted -sbar :', np.round(-sbar[:7], 4), ' predicted cum (1+eta)/(sig+eta)=', (1 + eta) / (sig + eta))
    else:
        ts, tl = (s_j * TV).sum(), (lam * TV).sum()
        print(' predicted Sbar  :', np.round(np.r_[0, Sbar[1:7]], 4), ' predicted cum', -(ts + eta * eps * (ts - tl)) / (sig + eta))

# unexpected shocks at date 0 (economy in SS before 0, batches in production fixed)
for kind in ('A', 'i'):
    A = np.ones(T); ip = np.full(T, rho)
    if kind == 'A': A[0] = np.exp(x)
    else: ip[0] = np.exp(np.log1p(rho) + x) - 1
    s = solve(T, ip, A); base = solve(T, np.full(T, rho), np.ones(T))
    yn = np.log(s['Y'] / base['Y']) / x
    print(f'unexpected {kind}: Y 0..8', np.round(yn[:9], 4), ' cum', yn.sum())
