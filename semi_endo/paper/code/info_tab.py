import pickle, glob, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
SP=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,SP); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import exp_horizon as eh
R={}
for f in glob.glob(SP+'/monthly/s6_*.pkl'):
    r=pickle.load(open(f,'rb')); R[(r['ttb'],r['case'][:3],r['K'],r['when'],r.get('lock',False))]=r
A={}
for f in glob.glob(SP+'/monthly/asym_*.pkl'):
    r=pickle.load(open(f,'rb')); A[(r['case'][:3],r['K'])]=r
rho=R[(True,'TFP',0,'news',False)]['ipath'][-1]
print('horizon table')
for k in (0,1,2,3,5,8,13):
    row=[]
    for nm,sc in (('TFP',1),('dem',1e3)):
        ev=R.get((True,nm,k,'news',False)); lk=(R.get((True,nm,k,'news',True)) if k else ev); no=R.get((False,nm,k,'news',False))
        pl=A[(nm,k)]['gain']*(1+rho)**(k-13) if k else ev['gain13']
        f=lambda r: sc*r['gain13'] if r else float('nan')
        row.append(f"{nm}: ev {f(ev):.4f} lock {f(lk):.4f} pl {sc*pl:.4f} none {f(no):.4f}")
    print(k,' | '.join(row))
print('utility levels (opt) relative to no-shock, at -13')
for nm in ('TFP','dem'):
    for k in (1,2,3,5,8,13):
        ev=R[(True,nm,k,'news',False)]; fct=(1+rho)**(k-13)
        g=eh.model(True,k); SS=g['SS']; unit=SS['Y']**(1-g['gamma'])/100
        a=A[(nm,k)]; o=a['opt']; T=len(o['C']); xi=np.r_[np.zeros(k),-0.01,np.zeros(T-k-1)] if nm=='dem' else np.zeros(T)
        w=g['DISC'][:T]*np.exp(xi)
        Upl=np.sum(w*(o['C']**(1-g['gamma'])/(1-g['gamma'])-g['chi']*o['L']**(1+g['eta'])/(1+g['eta'])))/unit
        Uss=ev['U']['ss']
        print(nm,k,'planner %.5f everyone %.5f diff %+.5f'%((Upl-Uss)*fct,(ev['U']['opt']-Uss)*fct,(Upl-ev['U']['opt'])*fct))
    r0=R[(True,nm,0,'news',False)]; print(nm,0,'unexpected %.5f'%((r0['U']['opt']-r0['U']['ss'])*(1+rho)**(-13)))
