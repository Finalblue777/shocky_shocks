src=open('ce_irf.py').read().split('T = 160')[0]
exec(src)
T=160; t0=50
for nm,(da,di) in {'tfp':(0.01,0),'rate':(0,0.01/12)}.items():
  for ts in (t0,0):
    e=E['ttb']; A=np.ones(T); ip=np.full(T,rho); A[ts]*=1+da; ip[ts]+=di
    sol=e['solve'](T,ip,A); rng=list(range(max(ts-15,0),ts+15))
    TV=e['TV']; lcum=np.r_[0.0,np.cumsum(np.log1p(sol['i']))]
    best=[]
    for a,s in enumerate(rng):
        for D in range(20):
            R=sol['p'][s+D]*np.exp(-(lcum[s+D]-lcum[s]))*kappa**D*(D+NARR)*sol['A'][s]-1
            for j in np.where((R>0)&(D!=TV))[0]: best.append((R[j],s-ts,D-TV[j],e['s'][j]))
    best=np.array(best)
    import collections
    c=collections.Counter()
    for r,sr,dd,w in best: c[(int(sr),int(dd))]+=w
    print(nm,'shock at',ts,' top (start date rel. shock, duration change) by expenditure:',[(k,round(v,3)) for k,v in c.most_common(6)])
