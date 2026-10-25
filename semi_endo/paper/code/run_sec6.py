# Section 6 of the paper: optimal policy vs peg, with and without time to build, for every shock experiment.
# Same fixed-wage model as exp_horizon.py (exact Euler, W fixed at SS, labour dated at hiring).
#   K = 0             : unexpected shock, policy from the shock date
#   K > 0, 'news'     : news K periods ahead, policy free from the news date (K + 8 rates)
#   K > 0, 'shock'    : news K periods ahead, policy free only from the shock date (8 rates)
# Saves welfare of optimal / peg / flexible wage, split into consumption and hours terms.
import sys, os, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import exp_horizon as eh

def terms(g, s):
    C, L, xi = s['Y'], s['L'], s['xi']; w = g['DISC'] * np.exp(xi)
    return float(np.sum(w * C ** (1 - g['gamma']) / (1 - g['gamma']))), float(-np.sum(w * g['chi'] * L ** (1 + g['eta']) / (1 + g['eta'])))

def run(job):
    ttb, nm, K, when = job[:4]; lock = len(job) > 4 and job[4]
    from scipy.optimize import minimize
    np.seterr(all='ignore'); g = eh.model(ttb, K, lock); T = g['T']; rho = g['rho']; SS = g['SS']; solve = g['solve']
    kw = {'demand shock': dict(xi=np.r_[np.zeros(K), -0.01, np.zeros(T - K - 1)]),
          'TFP shock': dict(A_path=np.r_[np.ones(K), 0.97, np.ones(T - K - 1)])}[nm]
    start = 0 if when == 'news' else K; L = (K + 8) if when == 'news' else 8
    full = lambda ip: np.r_[np.full(start, rho), np.asarray(ip, float), np.full(T - start - len(ip), rho)]
    def negU(ip):
        s = solve(full(ip), phi=1.0, key=None, **kw)
        return 1e3 if (not s['ok'] or not np.isfinite(s['welfare'])) else -s['welfare']
    best = None; t0 = time.time()
    for sd in (np.full(L, rho), np.full(L, 0.005)):
        r = minimize(negU, sd, method='L-BFGS-B', bounds=[(0, 0.05)] * L, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
        if best is None or r.fun < best.fun: best = r
    peg = solve(np.full(T, rho), phi=1.0, **kw); opt = solve(full(best.x), phi=1.0, **kw); fx = solve(phi=0.0, **kw)
    nos = solve(np.full(T, rho), phi=1.0)                                  # no shock (steady state)
    unit = SS['Y'] ** (1 - g['gamma']) / 100
    lg = lambda s, k: 100 * np.log(s[k] / SS[k])
    out = dict(ttb=ttb, case=nm, K=K, when=when, lock=lock, ipath=full(best.x), ok=(peg['ok'], opt['ok'], fx['ok']), secs=time.time() - t0,
               U={p: s['welfare'] / unit for p, s in (('opt', opt), ('peg', peg), ('flex', fx), ('ss', nos))},
               CL={p: tuple(v / unit for v in terms(g, s)) for p, s in (('opt', opt), ('peg', peg), ('flex', fx), ('ss', nos))},
               Y={p: lg(s, 'Y') for p, s in (('peg', peg), ('opt', opt), ('flex', fx))},
               Lab={p: lg(s, 'L') for p, s in (('peg', peg), ('opt', opt), ('flex', fx))},
               P={p: lg(s, 'P') for p, s in (('peg', peg), ('opt', opt))})
    out['gain'] = out['U']['opt'] - out['U']['peg']; out['gain13'] = out['gain'] * (1 + rho) ** (K - 13)
    pickle.dump(out, open(f"{SP}/monthly/s6_{'ttb' if ttb else 'none'}_{nm[:3]}_{K}_{when}{'_lock' if lock else ''}.pkl", 'wb'))
    print(f"done {'TTB' if ttb else 'none'} {nm} K={K} {when}{' lock' if lock else ''} {out['secs']:.0f}s ok={out['ok']} gain={out['gain']:.4f} "
          f"gain@-13={out['gain13']:.4f} i*={np.round(100 * best.x[:K + 4], 2)}", flush=True)
    return out

if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(ttb, nm, K, 'news') for K in (0, 1, 2, 3, 5, 13) for nm in ('TFP shock', 'demand shock') for ttb in (True, False)]
    jobs += [(ttb, nm, 13, 'shock') for nm in ('TFP shock', 'demand shock') for ttb in (True, False)]
    jobs += [(True, nm, K, 'news', True) for K in (1, 2, 3, 5, 13) for nm in ('TFP shock', 'demand shock')]
    jobs.sort(key=lambda j: (j[0], j[1] == 'demand shock'))
    with Pool(14) as p: p.map(run, jobs, chunksize=1)
    print('ALL DONE', flush=True)
