# Asymmetric information: the planner learns the shock at t = -K and commits to a rate path; firms and
# households observe the path but believe there is no shock.  At t = 0 the shock is revealed; everyone
# re-optimises, but deliveries already in production stay at the quantities planned under the old beliefs.
#
# Stage 1 (index 0 = news date, T1 = 30 + K): no-shock equilibrium under the announced rate path.
# Stage 2 (index 0 = shock date, T2 = 30): continuation with the shock, committed quantities from stage 1.
import sys, os, pickle, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # the notebook reads its CSV by a relative path
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import root, minimize
import exp_horizon as eh

def make(ttb, K):
    g1 = eh.model(ttb, K)          # stage 1 economy
    g2 = eh.model(ttb, 0)          # stage 2 economy (T = 30, cohort structure from the shock date)
    return g1, g2

def stage2(g2, YL, i2, A2, xi2):
    """Continuation from the shock date with committed quantities YL[t, j] for firms with tau_j > t."""
    T = g2['T']; SS = g2['SS']; TV = g2['TVALS']; FREE = g2['FREE']; SMAT = g2['SMAT']; TPT = g2['TPT']
    ZETA = g2['ZETA']; NF = g2['NF']; sigma = g2['sigma']; mu = g2['mu']; rho = g2['rho']; TAUMAX = g2['TAUMAX']
    gamma = g2['gamma']; beta_hh = g2['beta_hh']
    YLk = np.where(FREE, SS['y'][None, :], YL)                       # locked entries only matter where ~FREE
    S_QTY = np.sum(np.where(~FREE, YLk ** ((sigma - 1) / sigma), 0.0), axis=1)
    Y_FLR = S_QTY ** (sigma / (sigma - 1))
    def alloc(W, Y):
        pad = TAUMAX + 2
        Wx = np.r_[W, np.full(pad, SS['W'])]; Ax = np.r_[A2, np.full(pad, A2[-1])]
        ix = np.r_[i2, np.full(pad, rho)]; cd = np.r_[1.0, np.cumprod(1.0 / (1.0 + ix))]
        p_f = mu * Wx[SMAT] / ((cd[np.arange(T)][:, None] / cd[SMAT]) * ZETA[None, :] * Ax[SMAT])
        Q = np.sum(np.where(FREE, p_f ** (1 - sigma), 0.0), axis=1)
        P = (Q / (1.0 - S_QTY * Y ** ((1 - sigma) / sigma))) ** (1 / (1 - sigma))
        y = np.where(FREE, (p_f / P[:, None]) ** (-sigma) * Y[:, None], YLk)
        yx = np.vstack([y, np.tile(SS['y'], (pad, 1))])
        l = yx[TPT, np.arange(NF)] / (ZETA * Ax[:T, None])
        return P, y, l, l.sum(axis=1)
    def resid(u):
        W = np.exp(u[:T]); Y = Y_FLR + np.exp(u[T:])
        P, _, _, L = alloc(W, Y)
        Px = np.r_[P, SS['P']]; Yx = np.r_[Y, SS['Y']]; xix = np.r_[xi2, 0.0]
        wage = np.log(W) - np.log(SS['W'])
        eu = (xix[:T] - gamma * np.log(Yx[:T]) - np.log(Px[:T])) \
           - (np.log(beta_hh) + np.log(1 + i2) + xix[1:] - gamma * np.log(Yx[1:]) - np.log(Px[1:]))
        return np.r_[wage, eu]
    u0 = np.r_[np.full(T, np.log(SS['W'])), np.log(np.maximum(SS['Y'] - Y_FLR, 1e-6))]
    s = root(resid, u0, method='hybr', tol=1e-13)
    W = np.exp(s.x[:T]); Y = Y_FLR + np.exp(s.x[T:]); P, y, l, L = alloc(W, Y)
    return dict(Y=Y, L=L, P=P, y=y, l=l, ok=s.success)

def evaluate(g1, g2, K, nm, ipath):
    """ipath: full nominal-rate path from the news date (length T1).  Returns welfare and both stages."""
    T1 = g1['T']; rho = g1['rho']; SS = g1['SS']; gam = g1['gamma']; eta = g1['eta']; chi = g1['chi']
    s1 = g1['solve'](ipath, phi=1.0)                         # beliefs: no shock
    T2 = g2['T']; i2 = ipath[K:K + T2]
    A2 = np.r_[0.97, np.ones(T2 - 1)] if nm == 'TFP shock' else np.ones(T2)
    xi2 = np.r_[-0.01, np.zeros(T2 - 1)] if nm == 'demand shock' else np.zeros(T2)
    s2 = stage2(g2, s1['y'][K:K + T2], i2, A2, xi2)
    C = np.r_[s1['Y'][:K], s2['Y']]; L = np.r_[s1['L'][:K], s2['L']]
    xi = np.r_[np.zeros(K), xi2]; w = g1['DISC'][:K + T2] * np.exp(xi)
    U = float(np.sum(w * (C ** (1 - gam) / (1 - gam) - chi * L ** (1 + eta) / (1 + eta))))
    P = np.r_[s1['P'][:K], s2['P']]
    return U, dict(C=C, L=L, P=P, ok=(s1['ok'], s2['ok']), l1=s1['l'][:K], l2=s2['l'])

def run(job):
    ttb, nm, K = job
    g1, g2 = make(ttb, K); T1 = g1['T']; rho = g1['rho']
    full = lambda ip: np.r_[np.asarray(ip, float), np.full(T1 - len(ip), rho)]
    def negU(ip):
        try:
            U, out = evaluate(g1, g2, K, nm, full(ip))
        except Exception:
            return 1e3
        return 1e3 if (not all(out['ok']) or not np.isfinite(U)) else -U
    L = K + 8; t0 = time.time()
    seeds = [np.full(L, rho), np.full(L, 0.005)]
    try:   # warm start from the symmetric-news optimum
        h = pickle.load(open(f"{sys.argv[1]}/hz_{'ttb' if ttb else 'none'}_{nm[:3]}_{K}_news.pkl", 'rb'))
        seeds.insert(0, h['ipath'][:L])
    except Exception:
        pass
    best = None
    for sd in seeds:
        r = minimize(negU, sd, method='L-BFGS-B', bounds=[(0, 0.05)] * L,
                     options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
        if best is None or r.fun < best.fun: best = r
    Uo, oo = evaluate(g1, g2, K, nm, full(best.x)); Up, op = evaluate(g1, g2, K, nm, np.full(T1, rho))
    SS = g1['SS']; unit = SS['Y'] ** (1 - g1['gamma']) / 100
    out = dict(ttb=ttb, case=nm, K=K, ipath=full(best.x), gain=(Uo - Up) / unit, opt=oo, peg=op, secs=time.time() - t0)
    pickle.dump(out, open(f"{sys.argv[1]}/asym_{'ttb' if ttb else 'none'}_{nm[:3]}_{K}.pkl", 'wb'))
    print(f"done {'TTB' if ttb else 'none'} {nm} K={K} {out['secs']:.0f}s ok={oo['ok']} gain(news date)={out['gain']:.4f} "
          f"valued at -13: {out['gain'] * 1.01 ** (K - 13):.4f}  i*={np.round(100 * best.x[:K + 4], 2)}", flush=True)
    return out

if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(True, nm, K) for K in (13, 8, 5, 3, 2, 1) for nm in ('TFP shock', 'demand shock')]
    with Pool(12) as p: p.map(run, jobs, chunksize=1)
    print('ALL DONE', flush=True)
