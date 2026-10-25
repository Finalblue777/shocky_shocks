# Partially expected shocks: news arrives k periods before the shock (k = 1..5).  Same fixed-wage model as
# exp_cohort.py (exact Euler, W fixed at SS, labour dated at hiring).  Firms with tau_i > k are still locked
# at the shock date.  Policy free either from the news date (k + 8 rates) or from the shock date (8 rates).
import json, warnings, sys, io, contextlib, pickle, time
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np
import nbsrc
def model(ttb,K,lock=False):
    g={}; c3=nbsrc.cell(3)
    if lock:
        f='FREE  = TVALS[None, :] <= np.arange(T)[:, None]'; assert f in c3
        c3=c3.replace(f, f"FREE  = (TVALS[None, :] <= np.arange(T)[:, None]) & ((np.arange(T)[:, None] < {K}) | (TVALS[None, :] <= np.arange(T)[:, None] - {K}))")
    assert 'T     = 30' in c3; c3=c3.replace('T     = 30', f'T     = {30+K}')
    key='ZETA   = kappa ** TVALS * (TVALS + N_ARRAY)\n'; assert key in c3
    if not ttb: c3=c3.replace(key, key+'TVALS = np.zeros_like(TVALS); TAUMAX = 0\n')
    with contextlib.redirect_stdout(io.StringIO()):
        for s in (nbsrc.cell(1), c3): exec(s,g)
    return g
def run(job):
    ttb,nm,K,when=job[:4]; lock=len(job)>4 and job[4]; g=model(ttb,K,lock); T=g['T']; rho=g['rho']; SS=g['SS']; solve=g['solve']
    from scipy.optimize import minimize
    np.seterr(all='ignore')
    kw={'demand shock':dict(xi=np.r_[np.zeros(K),-0.01,np.zeros(T-K-1)]),
        'TFP shock':dict(A_path=np.r_[np.ones(K),0.97,np.ones(T-K-1)])}[nm]
    start=0 if when=='news' else K; L=(K+8) if when=='news' else 8
    full=lambda ip: np.r_[np.full(start,rho), np.asarray(ip,float), np.full(T-start-len(ip),rho)]
    def negU(ip):
        s=solve(full(ip),phi=1.0,key=None,**kw)
        return 1e3 if (not s['ok'] or not np.isfinite(s['welfare'])) else -s['welfare']
    best=None; t0=time.time()
    for sd in (np.full(L,rho), np.full(L,0.005)):
        r=minimize(negU,sd,method='L-BFGS-B',bounds=[(0,0.05)]*L,options={'maxiter':2000,'ftol':1e-16,'gtol':1e-12})
        if best is None or r.fun<best.fun: best=r
    peg=solve(np.full(T,rho),phi=1.0,**kw); opt=solve(full(best.x),phi=1.0,**kw); fx=solve(phi=0.0,**kw)
    lg=lambda s,k: 100*np.log(s[k]/SS[k])
    out=dict(ttb=ttb,case=nm,K=K,when=when,lock=lock,ipath=full(best.x),ok=(peg['ok'],opt['ok'],fx['ok']),secs=time.time()-t0,
             Y={p:lg(s,'Y') for p,s in (('peg',peg),('opt',opt),('flex',fx))},
             Lab={p:lg(s,'L') for p,s in (('peg',peg),('opt',opt),('flex',fx))},
             P={p:lg(s,'P') for p,s in (('peg',peg),('opt',opt))}, l={'peg':peg['l'],'opt':opt['l']}, lss=SS['l'], tv=g['TVALS'])
    pickle.dump(out,open(f"{sys.argv[1]}/hz_{'ttb' if ttb else 'none'}_{nm[:3]}_{K}_{when}{'_lock' if lock else ''}.pkl",'wb'))
    print(f"done {'TTB' if ttb else 'none'} {nm} K={K} {when} {out['secs']:.0f}s ok={out['ok']} i*={np.round(100*best.x[:K+4],2)}",flush=True)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    jobs=[(ttb,nm,K,w) for K in (1,2,3,4,5) for w in ('news','shock') for nm in ('TFP shock','demand shock') for ttb in (True,False)]
    jobs.sort(key=lambda j: (j[0], j[1]=='demand shock'))       # slow no-TTB TFP jobs first
    with Pool(14) as p: res=p.map(run,jobs,chunksize=1)
    pickle.dump(res,open(sys.argv[1]+'/exp_horizon.pkl','wb')); print('ALL DONE',flush=True)
