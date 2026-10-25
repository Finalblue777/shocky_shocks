# Rebuild every figure of ttb_policy_slides / ttb_irf_slides from the monthly runs.
# 1. Convert monthly/s6_*.pkl and monthly/asym_*.pkl into the data files the old plotting scripts expect
#    (rates stored annualised, labour by producer re-solved at the stored rate paths).
# 2. Run patched copies of ttb_plots_v2.py and asym_fig.py from semi_endo/slides.
import glob, os, pickle, re, subprocess, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); M = SP + '/monthly'; OUT = SP + '/slides_data'
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SP); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import exp_horizon as eh

R = {}
for f in glob.glob(M + '/s6_*.pkl'):
    r = pickle.load(open(f, 'rb'))
    if not r.get('lock', False): R[(r['case'], r['ttb'], r['K'], r['when'])] = r
rho = R[('TFP shock', True, 0, 'news')]['ipath'][-1]
ann = lambda i: (1 + np.asarray(i)) ** 12 - 1
MODELS = {}
def model(ttb, K):
    if (ttb, K) not in MODELS: MODELS[(ttb, K)] = eh.model(ttb, K)
    return MODELS[(ttb, K)]
def shock_kw(nm, K, T):
    return {'demand shock': dict(xi=np.r_[np.zeros(K), -0.01, np.zeros(T - K - 1)]),
            'TFP shock': dict(A_path=np.r_[np.ones(K), 0.97, np.ones(T - K - 1)])}[nm]
def labour(nm, K, when):
    g = model(True, K); r = R[(nm, True, K, when)]; kw = shock_kw(nm, K, g['T'])
    return {pol: g['solve'](path, phi=1.0, **kw)['l'] for pol, path in (('peg', np.full(g['T'], rho)), ('opt', r['ipath']))}
def utility_terms(nm, ttb, K, Ylog, Llog):
    g = model(ttb, K); SS = g['SS']; T = g['T']; gam, eta, chi = g['gamma'], g['eta'], g['chi']
    xi = shock_kw(nm, K, T).get('xi', np.zeros(T)); w = g['DISC'] * np.exp(xi); unit = SS['Y'] ** (1 - gam) / 100
    C = SS['Y'] * np.exp(Ylog / 100); L = SS['L'] * np.exp(Llog / 100)
    return w * C ** (1 - gam) / (1 - gam) / unit, -w * chi * L ** (1 + eta) / (1 + eta) / unit

def rec(r, with_l=None, i_from=0):
    d = dict(case=r['case'], ttb=r['ttb'], K=r['K'], when='news', Y=r['Y'], Lab=r['Lab'], L=r['Lab'], P=r['P'],
             i=ann(r['ipath'][i_from:]), ipath=ann(r['ipath']))
    if with_l is not None:
        g = model(True, r['K']); d.update(l=with_l, lss=g['SS']['l'], tv=g['TVALS'])
    return d

NMS = ('TFP shock', 'demand shock'); K13 = 13
XE = [rec(R[(nm, tb, K13, 'news')], labour(nm, K13, 'news') if tb else None) for nm in NMS for tb in (True, False)]
for d in XE: d['K'] = K13
pickle.dump(XE, open(OUT + '/exp_cohort_news.pkl', 'wb'))
XU = [rec(R[(nm, tb, 0, 'news')], labour(nm, 0, 'news') if tb else None) for nm in NMS for tb in (True, False)]
pickle.dump(XU, open(OUT + '/ttb_unexp.pkl', 'wb'))
X0 = [rec(R[(nm, tb, K13, 'shock')], labour(nm, K13, 'shock') if tb else None, i_from=K13) for nm in NMS for tb in (True, False)]
pickle.dump(X0, open(OUT + '/exp_cohort.pkl', 'wb'))
for k in range(1, 13):
    for nm in NMS:
        for tb in (True, False):
            r = R[(nm, tb, k, 'news')]
            d = rec(r, labour(nm, k, 'news') if tb else None); d['when'] = 'news'
            pickle.dump(d, open(f"{OUT}/hz_{'ttb' if tb else 'none'}_{nm[:3]}_{k}_news.pkl", 'wb'))

# welfare by date: gains (optimal - peg) and losses vs the flexible wage
G2, GT, D = {}, {}, {}
for nm in NMS:
    for tb in (True, False):
        for case, K, when in (('unexp', 0, 'news'), ('expnews', K13, 'news'), ('exp0', K13, 'shock')):
            r = R[(nm, tb, K, when)]
            T = {q: utility_terms(nm, tb, K, r['Y'][q], r['Lab'][q]) for q in ('opt', 'peg', 'flex')}
            for pol in ('opt', 'peg'):
                D[(case, nm, tb, pol)] = (T[pol][0] - T['flex'][0], T[pol][1] - T['flex'][1])
            dC, dL = T['opt'][0] - T['peg'][0], T['opt'][1] - T['peg'][1]
            if case == 'unexp': G2[('unexpected', nm, tb)] = dict(dC=dC, dL=dL)
            else: GT[(nm, tb, 'from news' if when == 'news' else 'from t=0')] = dict(dC=dC, dL=dL, i=100 * ann(r['ipath']))
pickle.dump(G2, open(OUT + '/gain_decomp2.pkl', 'wb')); pickle.dump(GT, open(OUT + '/gain_timing.pkl', 'wb'))
pickle.dump(D, open(OUT + '/loss_decomp.pkl', 'wb'))

# horizon table (everyone informed, valued at t = -13) and planner-informed runs
rows = []
for k in range(14):
    row = [k]
    for nm in NMS:
        r = R[(nm, True, k, 'news')]; U = r['U']; f = (1 + rho) ** (k - 13)
        row += [(U['peg'] - U['flex']) * f, (U['opt'] - U['flex']) * f, r['gain13']]
    rows.append(row)
pickle.dump(rows, open(OUT + '/horizon_loss_table.pkl', 'wb'))
for f in glob.glob(M + '/asym_ttb_*.pkl'):
    r = pickle.load(open(f, 'rb')); r = dict(r, ipath=ann(r['ipath']))
    pickle.dump(r, open(OUT + '/' + os.path.basename(f), 'wb'))
print('data written; rho =', rho, ' annualised', ann(rho))

# ---------------------------------------------------------------- patched plotting scripts
RA = float(ann(rho)); RM = float(rho)
src = open(SP + '/ttb_plots_v2.py').read()
src = src.replace("RHO=1.0;", f"RHO={100 * RA!r}; RA={RA!r}; RM={RM!r};")
src = re.sub(r",0\.01\)", ",RA)", src)
src = src.replace("RHO/100", "RM")
old_ul = src[src.index("d2=json.load(open(NB+'fixed_wage_cohorts.ipynb')); h={}"):src.index("def lab_unexp(nm,pol,t0):")]
src = src.replace(old_ul, "UL={(nm,pol):split(XU[(nm,True)]['l'][pol],XU[(nm,True)]['lss'],XU[(nm,True)]['tv']) for nm in ('TFP shock','demand shock') for pol in ('peg','opt')}\nimport os; here=os.getcwd()\n")
src = src.replace("X0={(r['case'],r['ttb']):r for r in pickle.load(open(S+'/exp_cohort.pkl','rb')) if r['L']==8}",
                  "X0={(r['case'],r['ttb']):r for r in pickle.load(open(S+'/exp_cohort.pkl','rb'))}")
src = src.replace("sys.path.insert(0,S); os.chdir(NB); import exp_horizon as eh",
                  f"sys.path.insert(0,{SP!r}); os.chdir(NB); import exp_horizon as eh")
for a, b in (("one period's", "one month's"), ("periods ahead", "months ahead"), ("('nominal rate $i_t$','%')", "('nominal rate $i_t$','% a year')"),
             ("('inflation $\\\\pi_t$','%')", "('inflation $\\\\pi_t$','% a month')"), ("ax.set_xlabel('t',", "ax.set_xlabel('month',"),
             ("for ax in axes[1]: ax.set_xlabel('t',", "for ax in axes[1]: ax.set_xlabel('month',")):
    src = src.replace(a, b)
src = src.replace("ax4[3].step(t,(RHO-pi[1:])[:M],where='post',color=col,lw=3.2)",
                  "ax4[3].step(t,(100*(((1+RM)*np.exp(-pi[1:]/100))**12-1))[:M],where='post',color=col,lw=3.2)")
src = src.replace("('real rate $i_t-\\\\pi_{t+1}$ (nominal rate fixed at $\\\\rho$)','%')", "('real rate, annualised (nominal rate fixed at $\\\\rho$)','% a year')")
src = src.replace("    ax4[3].axhline(RHO,color='0.6',lw=1.6)\n",
                  "    ax4[3].axhline(RHO,color='0.6',lw=1.6)\n    for a_ in ax4[:3]:\n        lo_,hi_=a_.get_ylim()\n        if hi_-lo_<1e-6: a_.set_ylim(-0.5,0.5)\n")
assert "np.exp(-pi[1:]/100))**12-1" in src and "if hi_-lo_<1e-6" in src
open(SP + '/ttb_plots_monthly.py', 'w').write(src)
asrc = open(SP + '/asym_fig.py').read()
asrc = asrc.replace("S=sys.argv[1]; K=13; RHO=1.0", f"S=sys.argv[1]; K=13; RHO={100 * RA!r}; RA={RA!r}; RM={RM!r}")
asrc = re.sub(r",0\.01\)", ",RA)", asrc).replace("1.01**(k-K)", "(1+RM)**(k-K)").replace("one period's", "one month's")
open(SP + '/asym_fig_monthly.py', 'w').write(asrc)
SL = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'slides')
for script in ('ttb_plots_monthly.py', 'asym_fig_monthly.py'):
    p = subprocess.run([sys.executable, SP + '/' + script, OUT], cwd=SL, capture_output=True, text=True)
    print(script, 'exit', p.returncode); print(p.stdout[-2500:]); print(p.stderr[-2500:])
