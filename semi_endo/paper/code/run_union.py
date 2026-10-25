# Section 6 experiments with Rotemberg union wage setting instead of a fixed nominal wage.
# Same jobs and output format as run_sec6.py; DUR = average Calvo-equivalent wage duration in months.
import sys, os, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import exp_horizon as eh, union

def run(job):
    dur, ttb, nm, K, when = job[:5]; lock = len(job) > 5 and job[5]
    from scipy.optimize import minimize
    np.seterr(all='ignore'); g = union.add_union(eh.model(ttb, K, lock), dur)
    T = g['T']; rho = g['rho']; SS = g['SS']; solve = g['solve']
    kw = {'demand shock': dict(xi=np.r_[np.zeros(K), -0.01, np.zeros(T - K - 1)]),
          'TFP shock': dict(A_path=np.r_[np.ones(K), 0.97, np.ones(T - K - 1)])}[nm]
    start = 0 if when == 'news' else K; L = (K + 8) if when == 'news' else 8
    full = lambda ip: np.r_[np.full(start, rho), np.asarray(ip, float), np.full(T - start - len(ip), rho)]
    def negU(ip):
        s = solve(full(ip), phi=1.0, **kw)
        return 1e3 if (not s['ok'] or not np.isfinite(s['welfare'])) else -s['welfare']
    best = None; t0 = time.time()
    tagf = f"{'ttb' if ttb else 'none'}_{nm[:3]}_{K}_{when}{'_lock' if lock else ''}"
    try:   # warm start from the fixed-wage optimum (the union wage barely moves, so it is close)
        prev = pickle.load(open(f"{SP}/monthly/s6_{tagf}.pkl", 'rb'))['ipath'][start:start + L]
        seeds = (prev,)
    except Exception:
        seeds = (np.full(L, rho), np.full(L, 0.005))
    for sd in seeds:
        r = minimize(negU, sd, method='L-BFGS-B', bounds=[(0, 0.05)] * L, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
        if best is None or r.fun < best.fun: best = r
    peg = solve(np.full(T, rho), phi=1.0, **kw); opt = solve(full(best.x), phi=1.0, **kw); fx = solve(phi=0.0, **kw)
    unit = SS['Y'] ** (1 - g['gamma']) / 100; lg = lambda s, k: 100 * np.log(s[k] / SS[k])
    out = dict(dur=dur, ttb=ttb, case=nm, K=K, when=when, lock=lock, ipath=full(best.x), ok=(peg['ok'], opt['ok'], fx['ok']),
               secs=time.time() - t0, U={p: s['welfare'] / unit for p, s in (('opt', opt), ('peg', peg), ('flex', fx))},
               Y={p: lg(s, 'Y') for p, s in (('peg', peg), ('opt', opt), ('flex', fx))},
               Lab={p: lg(s, 'L') for p, s in (('peg', peg), ('opt', opt), ('flex', fx))},
               P={p: lg(s, 'P') for p, s in (('peg', peg), ('opt', opt))},
               W={p: lg(s, 'W') for p, s in (('peg', peg), ('opt', opt), ('flex', fx))})
    out['gain'] = out['U']['opt'] - out['U']['peg']; out['gain13'] = out['gain'] * (1 + rho) ** (K - 13)
    tag = f"{'ttb' if ttb else 'none'}_{nm[:3]}_{K}_{when}{'_lock' if lock else ''}"
    pickle.dump(out, open(f"{SP}/union_runs/u{int(dur)}_{tag}.pkl", 'wb'))
    print(f"done dur={dur} {tag} {out['secs']:.0f}s ok={out['ok']} gain@-13={out['gain13']:.4f} i*={np.round(100 * best.x[:K + 4], 2)}", flush=True)
    return out

if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(dur, ttb, nm, K, 'news') for dur in (12, 3) for K in (0, 1, 2, 3, 5, 13) for nm in ('TFP shock', 'demand shock') for ttb in (True, False)]
    jobs += [(dur, ttb, nm, 13, 'shock') for dur in (12, 3) for nm in ('TFP shock', 'demand shock') for ttb in (True, False)]
    jobs += [(12, True, nm, K, 'news', True) for K in (1, 13) for nm in ('TFP shock', 'demand shock')]
    jobs.sort(key=lambda j: (j[1], j[2] == 'demand shock', -j[3]))
    with Pool(14) as p: p.map(run, jobs, chunksize=1)
    print('UNION ALL DONE', flush=True)
