# All figures of ttb_policy_slides from one place.
#   expected shocks  : exp_cohort_news.pkl  (fixed-wage model, exact Euler, news at t = -K, policy free from the news date)
#   unexpected shocks: ttb_unexp.pkl + labour by firm re-solved from fixed_wage_cohorts.ipynb
import json, sys, io, contextlib, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
S=sys.argv[1]; NB='../'
# time to build = bold red, no time to build = black (names kept from the old blue/green scheme)
BLUE,GREEN,INK,MUTED,RED='#d00000','#000000','#0b0b0b','#52514e','#000000'
COLS=['#56B4E9','#E69F00','#009E73','#CC79A7','#0072B2','#D55E00']; GROUPS=[0,1,2,3,4,5]
RHO=1.0; OPT=dict(lw=3.2); PEG=dict(lw=2.6,ls=(0,(1.2,1.6)))
plt.rcParams.update({'font.size':13})

# ---------------------------------------------------------------- data, shock date = index 0
XE={(r['case'],r['ttb']):r for r in pickle.load(open(S+'/exp_cohort_news.pkl','rb'))}
K=XE[('TFP shock',True)]['K']
XU={(r['case'],r['ttb']):r for r in pickle.load(open(S+'/ttb_unexp.pkl','rb'))}

def infl(P): return np.diff(np.r_[0.0,P])            # P_{-1} = P^ss (before the news / the shock)
def series_exp(nm,ttb,t0):
    r=XE[(nm,ttb)]; T=len(r['Y']['peg']); a=K-t0
    ip=100*np.r_[r['i'],np.full(T-len(r['i']),0.01)]
    return dict(c=(r['Y']['opt'][a:],r['Y']['peg'][a:]), n=(r['Lab']['opt'][a:],r['Lab']['peg'][a:]),
                pi=(infl(r['P']['opt'])[a:],infl(r['P']['peg'])[a:]), i=(ip[a:],np.full(T-a,RHO)))
def series_unexp(nm,ttb,t0):
    r=XU[(nm,ttb)]; T=len(r['Y']['peg']); z=np.zeros(t0)
    ip=np.r_[np.full(t0,RHO),100*np.r_[r['i'],np.full(T-len(r['i']),0.01)]]
    return dict(c=(np.r_[z,r['Y']['opt']],np.r_[z,r['Y']['peg']]), n=(np.r_[z,r['L']['opt']],np.r_[z,r['L']['peg']]),
                pi=(np.r_[z,infl(r['P']['opt'])],np.r_[z,infl(r['P']['peg'])]), i=(ip,np.full(T+t0,RHO)))

def style(ax,title,xr,step,vline=True):
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color('0.75')
    ax.tick_params(colors=MUTED,labelsize=16,length=3); ax.grid(axis='y',color='0.9',lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0,color='0.7',lw=1,zorder=1); ax.set_xlim(*xr); ax.set_title(title,color=INK,fontsize=20,loc='left')
    lo=int(np.ceil(xr[0]/step)*step); ax.set_xticks(range(lo,int(xr[1])+1,step))
    if vline and xr[0]<0: ax.axvline(0,color='0.6',lw=1,ls='--',zorder=0)

def shock_fig(get,fname,t0=0,H=14,cols=None):
    CT,CN=cols if cols else (BLUE,GREEN)
    t=np.arange(-t0,H); M=t0+H
    fig,axes=plt.subplots(2,4,figsize=(20,6.9)); fig.patch.set_facecolor('white')
    for row,(nm,lab) in enumerate((('TFP shock','TFP'),('demand shock','Demand'))):
        for ttb,col in ((True,CT),(False,CN)):
            g=get(nm,ttb,t0)
            for k,key in enumerate(('c','n','pi')):
                axes[row,k].plot(t,g[key][0][:M],color=col,**OPT); axes[row,k].plot(t,g[key][1][:M],color=col,**PEG)
            axes[row,3].step(t,g['i'][0][:M],where='post',color=col,**OPT)
        axes[row,3].step(t,np.full(M,RHO),where='post',color=MUTED,**PEG)
        for k,(ttl,yl) in enumerate((('consumption $c_t$','% from SS'),('employment $n_t$','% from SS'),('inflation $\\pi_t$','%'),('nominal rate $i_t$','%'))):
            style(axes[row,k],f'{lab}: {ttl}',(-t0,H-1),4 if t0<=5 else 5); axes[row,k].set_ylabel(yl,color=MUTED,fontsize=16)
    for ax in axes[1]: ax.set_xlabel('t',color=MUTED)
    fig.tight_layout()
    h=[Line2D([],[],color=CT,lw=3.2,label='time to build'),Line2D([],[],color=CN,lw=3.2,label='no time to build'),
       Line2D([],[],color=MUTED,lw=3.2,label='optimal rate'),Line2D([],[],color=MUTED,lw=2.6,ls=(0,(1.2,1.6)),label=r'peg at $\rho$')]
    fig.legend(handles=h,loc='lower center',ncol=4,frameon=False,fontsize=19,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.09))
    fig.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(fig)

# ---------------------------------------------------------------- labour hired at t, by delivery lag
def split(l_hired,l_ss,tv):
    N=l_ss.sum(); out=np.zeros((l_hired.shape[0],len(GROUPS)))
    for k,gk in enumerate(GROUPS):
        m=(tv>=gk) if gk==GROUPS[-1] else (tv==gk)
        out[:,k]=100*(l_hired[:,m].sum(1)-l_ss[m].sum())/N
    return out
def lab_exp(nm,pol,t0):
    r=XE[(nm,True)]; A=split(r['l'][pol],r['lss'],r['tv'])[K-t0:]
    v0=100*np.expm1(XE[(nm,False)]['Lab'][pol][K-t0:]/100)
    return A,v0
d2=json.load(open(NB+'fixed_wage_cohorts.ipynb')); h={}
import os; here=os.getcwd(); os.chdir(NB)          # the notebook reads its CSV by a bare relative path
with contextlib.redirect_stdout(io.StringIO()):
    for k in (1,3): exec(''.join(d2['cells'][k]['source']), h)
os.chdir(here)
UL={}
for nm in ('TFP shock','demand shock'):
    kw=h['SHOCKS'][nm]; Tu=h['T']; ip=np.r_[XU[(nm,True)]['i'],np.full(Tu-8,h['rho'])]
    for pol,path in (('peg',np.full(Tu,h['rho'])),('opt',ip)):
        s=h['solve'](path,phi=1.0,**kw); UL[(nm,pol)]=split(s['l'],h['SS']['l'],h['TVALS'])
def lab_unexp(nm,pol,t0):
    A=np.vstack([np.zeros((t0,len(GROUPS))),UL[(nm,pol)]])
    v0=np.r_[np.zeros(t0),100*np.expm1(XU[(nm,False)]['L'][pol]/100)]
    return A,v0

def panel(ax,A,v0,title,ylab,t0,H):
    x=np.arange(-t0,H); M=t0+H; pos=np.zeros(M); neg=np.zeros(M)
    for k in range(len(GROUPS)):
        v=A[:M,k]; vp=np.clip(v,0,None); vn=np.clip(v,None,0)
        ax.bar(x,vp,bottom=pos,color=COLS[k],width=0.78,edgecolor='white',lw=0.4); pos+=vp
        ax.bar(x,vn,bottom=neg,color=COLS[k],width=0.78,edgecolor='white',lw=0.4); neg+=vn
    tot=A[:M].sum(1); v0=v0[:M]
    ax.plot(x,tot,color=BLUE,lw=0,marker='D',ms=8,mec='white',mew=0.8,zorder=5)
    nz=np.abs(v0)>1e-2; ax.plot(x[nz],v0[nz],color=RED,lw=0,marker='s',ms=11,mec='white',mew=1.0,zorder=6)
    lo=min(neg.min(),tot.min(),v0.min(),0); hi=max(pos.max(),tot.max(),v0.max(),0); pad=0.08*(hi-lo)
    if hi-lo<0.005:
        ax.set_ylim(-0.5,0.5); ax.text(0.5,0.75,'$\\approx 0$',transform=ax.transAxes,ha='center',color=MUTED,fontsize=18)
    else: ax.set_ylim(lo-pad,hi+pad)
    style(ax,title,(-t0-0.6,H-0.4),2 if t0<=5 else 4,vline=False)
    if t0: ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0)
    if ylab: ax.set_ylabel('change, % of SS\nemployment',color=MUTED,fontsize=15)
def labour_fig(get,fname,t0=0,H=10):
    f,axes=plt.subplots(2,2,figsize=(20,7.2)); f.patch.set_facecolor('white')
    for r,(nm,lab) in enumerate((('TFP shock','TFP'),('demand shock','Demand'))):
        for c,(pol,ttl) in enumerate((('peg','peg at $\\rho$'),('opt','optimal rate'))):
            A,v0=get(nm,pol,t0); panel(axes[r,c],A,v0,f'{lab}, {ttl}: labour hired at $t$',c==0,t0,H)
    for ax in axes[1]: ax.set_xlabel('hiring date t',color=MUTED,fontsize=16)
    f.tight_layout()
    hs=[Patch(color=COLS[k],label=(f'$t+{GROUPS[k]}$' if k<len(GROUPS)-1 else f'$t+{GROUPS[k]}$ or later')) for k in range(len(GROUPS))]
    hs+=[Line2D([],[],lw=0,marker='D',ms=9,color=BLUE,label='total, time to build'),Line2D([],[],lw=0,marker='s',ms=11,color=RED,label='total, no time to build')]
    f.legend(handles=hs,loc='lower center',ncol=len(hs),frameon=False,fontsize=17,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.10),
             columnspacing=1.4,handlelength=1.4,title='time to build, delivered at:',title_fontsize=16)
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)

shock_fig(series_exp,'figs/ttb_expected.png')
shock_fig(series_unexp,'figs/ttb_unexpected.png')
labour_fig(lab_exp,'figs/ttb_expected_labour.png')
labour_fig(lab_unexp,'figs/ttb_unexpected_labour.png')
shock_fig(series_exp,'figs/ttb_expected_pre.png',t0=K)            # appendix: from the news date
labour_fig(lab_exp,'figs/ttb_expected_labour_pre.png',t0=10,H=8)
print('ok')

# appendix: optimal nominal rate, expected vs unexpected, with and without time to build
def rate_fig(fname,H=9):
    t=np.arange(-K,H); M=K+H
    f,axes=plt.subplots(1,2,figsize=(20,5.6)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        ax.step(t,np.full(M,RHO),where='post',color='0.6',lw=1.6)
        for ttb,col in ((True,BLUE),(False,GREEN)):
            ie=100*np.r_[XE[(nm,ttb)]['i'],np.full(M,0.01)][:M]
            iu=np.r_[np.full(K,RHO),100*np.r_[XU[(nm,ttb)]['i'],np.full(H,0.01)]][:M]
            ax.step(t,ie,where='post',color=col,lw=3.2)
            ax.step(t,iu,where='post',color=col,lw=2.6,ls=(0,(4,2.5)))
        style(ax,f'{lab}: optimal nominal rate $i_t$',(-K,H-1),2); ax.set_ylabel('%',color=MUTED,fontsize=16); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Line2D([],[],color=BLUE,lw=3.2,label='time to build'),Line2D([],[],color=GREEN,lw=3.2,label='no time to build'),
       Line2D([],[],color=MUTED,lw=3.2,label='expected (news at $t=-13$)'),Line2D([],[],color=MUTED,lw=2.6,ls=(0,(4,2.5)),label='unexpected (news at $t=0$)'),
       Line2D([],[],color='0.6',lw=1.6,label=r'$\rho$')]
    f.legend(handles=h,loc='lower center',ncol=5,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.12))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
rate_fig('figs/ttb_rates_compare.png')
print('rates ok')

# appendix: optimal nominal rate, expected shock with time to build only
def rate_fig_ttb(fname,H=9):
    t=np.arange(-K,H); M=K+H
    f,axes=plt.subplots(1,2,figsize=(20,5.6)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        ax.step(t,np.full(M,RHO),where='post',color='0.6',lw=1.6)
        y=100*np.r_[XE[(nm,True)]['i'],np.full(M,0.01)][:M]
        ax.step(t,y,where='post',color=BLUE,lw=3.2)
        style(ax,f'{lab}: optimal nominal rate $i_t$',(-K,H-1),2); ax.set_ylabel('%',color=MUTED,fontsize=16); ax.set_xlabel('t',color=MUTED,fontsize=16)
        if y.min()>0.5:                       # nowhere near zero: zoom in on the movement around rho
            pad=0.15*(y.max()-y.min()); ax.set_ylim(y.min()-pad,y.max()+pad)
    f.tight_layout()
    h=[Line2D([],[],color=BLUE,lw=3.2,label='expected, time to build'),Line2D([],[],color='0.6',lw=1.6,label=r'$\rho$')]
    f.legend(handles=h,loc='lower center',ncol=2,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.12))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
rate_fig_ttb('figs/ttb_rates_expected_ttb.png')
print('rates ttb ok')


def stack_cl(ax,t,dC,dL,CC,LC):
    """consumption and labour terms in one bar per date: positive parts stacked up, negative parts down"""
    pos=np.zeros(len(t)); neg=np.zeros(len(t))
    for v,col in ((dC,CC),(dL,LC)):
        vp=np.clip(v,0,None); vn=np.clip(v,None,0)
        ax.bar(t,vp,bottom=pos,width=0.78,color=col,edgecolor='white',lw=0.4); pos+=vp
        ax.bar(t,vn,bottom=neg,width=0.78,color=col,edgecolor='white',lw=0.4); neg+=vn

# welfare gain peg -> optimal, by date: consumption term vs labour term (time to build); totals for both
def gain_fig(info,fname,H=9):
    G=pickle.load(open(S+'/gain_decomp2.pkl','rb')); t0=0
    if info=='expected':   # welfare from the perspective of the shock date: periods t >= 0, discounted to t = 0
        G={k:{q:v[q][K:]*(1+RHO/100)**K for q in ('dC','dL')} for k,v in G.items() if k[0]=='expected'}
    t=np.arange(-t0,H); M=t0+H; CC,LC='#0072B2','#E69F00'
    f,axes=plt.subplots(1,2,figsize=(20,6.2)); f.patch.set_facecolor('white')
    tot=lambda g: (g['dC'].sum(),g['dL'].sum())
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        g=G[(info,nm,True)]; dC,dL=g['dC'][:M],g['dL'][:M]
        stack_cl(ax,t,dC,dL,CC,LC)
        ax.plot(t,dC+dL,color=INK,lw=0,marker='D',ms=7,zorder=5)
        c1,l1=tot(g); c0,l0=tot(G[(info,nm,False)])
        style(ax,f'{lab}\ntime to build: C {c1:+.2f}, L {l1:+.2f}, total {c1+l1:+.3f}\nno time to build: C {c0:+.2f}, L {l0:+.2f}, total {c0+l0:+.3f}',
              (-t0-0.6,H-0.4),2,vline=False)
        ax.title.set_fontsize(17)
        if t0: ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0)
        ax.set_ylabel("gain, % of one period's\nSS consumption",color=MUTED,fontsize=15); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Patch(color=CC,label='from consumption (C)'),Patch(color=LC,label='from labour (L, less disutility)'),
       Line2D([],[],lw=0,marker='D',ms=9,color=INK,label='total, time to build')]
    f.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.11))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
gain_fig('unexpected','figs/ttb_gain_decomp_unexpected.png')
print('gain ok')

# expected shocks: planner acts only from t = 0 vs from the news date
def timing_fig(fname):
    G=pickle.load(open(S+'/gain_timing.pkl','rb')); CC,LC='#0072B2','#E69F00'
    cases=[(True,'from t=0'),(True,'from news'),(False,'from t=0'),(False,'from news')]
    labels=['time to build\nact from $t=0$','time to build\nact from news','no time to build\nact from $t=0$','no time to build\nact from news']
    f,axes=plt.subplots(1,2,figsize=(20,6.4)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        x=np.arange(len(cases))
        for k,(ttb,when) in enumerate(cases):
            g=G[(nm,ttb,when)]; c,l=g['dC'].sum(),g['dL'].sum()
            ax.bar(k-0.18,c,width=0.36,color=CC,edgecolor='white'); ax.bar(k+0.18,l,width=0.36,color=LC,edgecolor='white')
            ax.plot(k,c+l,color=INK,lw=0,marker='D',ms=10,zorder=5)
            ax.annotate(f'{c+l:+.3f}',xy=(k,c+l),xytext=(14,-4),textcoords='offset points',fontsize=16,color=INK)
        for sp in ('top','right'): ax.spines[sp].set_visible(False)
        for sp in ('left','bottom'): ax.spines[sp].set_color('0.75')
        ax.tick_params(colors=MUTED,labelsize=15,length=3); ax.grid(axis='y',color='0.9',lw=0.8); ax.set_axisbelow(True)
        ax.axhline(0,color='0.6',lw=1); ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=14,color=INK)
        ax.axvline(1.5,color='0.8',lw=1)
        ax.set_title(f'{lab}: welfare gain over the peg, by source',color=INK,fontsize=20,loc='left')
        ax.set_ylabel("% of one period's\nSS consumption",color=MUTED,fontsize=15)
    f.tight_layout()
    h=[Patch(color=CC,label='from consumption'),Patch(color=LC,label='from labour (less disutility)'),Line2D([],[],lw=0,marker='D',ms=10,color=INK,label='total')]
    f.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.10))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
def timing_rate_fig(fname,H=9):
    G=pickle.load(open(S+'/gain_timing.pkl','rb')); t=np.arange(-K,H); M=K+H
    f,axes=plt.subplots(1,2,figsize=(20,5.8)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        ax.step(t,np.full(M,RHO),where='post',color='0.6',lw=1.6)
        for ttb,col in ((True,BLUE),(False,GREEN)):
            ax.step(t,G[(nm,ttb,'from news')]['i'][:M],where='post',color=col,lw=3.2)
            ax.step(t,G[(nm,ttb,'from t=0')]['i'][:M],where='post',color=col,lw=2.6,ls=(0,(4,2.5)))
        style(ax,f'{lab}: optimal nominal rate $i_t$',(-K,H-1),2); ax.set_ylabel('%',color=MUTED,fontsize=16); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Line2D([],[],color=BLUE,lw=3.2,label='time to build'),Line2D([],[],color=GREEN,lw=3.2,label='no time to build'),
       Line2D([],[],color=MUTED,lw=3.2,label='planner acts from the news ($t=-13$)'),Line2D([],[],color=MUTED,lw=2.6,ls=(0,(4,2.5)),label='planner acts from $t=0$'),
       Line2D([],[],color='0.6',lw=1.6,label=r'$\rho$')]
    f.legend(handles=h,loc='lower center',ncol=5,frameon=False,fontsize=17,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.12))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
timing_fig('figs/ttb_gain_timing.png')
timing_rate_fig('figs/ttb_rates_timing.png')
print('timing ok')

# expected shocks, planner acts only from t = 0 (rate at rho before the shock)
X0={(r['case'],r['ttb']):r for r in pickle.load(open(S+'/exp_cohort.pkl','rb')) if r['L']==8}
def series_exp0(nm,ttb,t0):
    r=X0[(nm,ttb)]; T=len(r['Y']['peg']); a=K-t0
    ip=100*np.r_[np.full(K,0.01),r['i'],np.full(T,0.01)][:T]
    return dict(c=(r['Y']['opt'][a:],r['Y']['peg'][a:]), n=(r['Lab']['opt'][a:],r['Lab']['peg'][a:]),
                pi=(infl(r['P']['opt'])[a:],infl(r['P']['peg'])[a:]), i=(ip[a:],np.full(T-a,RHO)))
def lab_exp0(nm,pol,t0):
    r=X0[(nm,True)]; A=split(r['l'][pol],r['lss'],r['tv'])[K-t0:]
    v0=100*np.expm1(X0[(nm,False)]['Lab'][pol][K-t0:]/100)
    return A,v0
shock_fig(series_exp0,'figs/ttb_expected0.png')
labour_fig(lab_exp0,'figs/ttb_expected0_labour.png')
print('exp0 ok')

# expected shocks: welfare gain by source, by date from the news (agents react to the news either way)
def gain_fig_exp(when,fname,H=9):
    G=pickle.load(open(S+'/gain_timing.pkl','rb')); t=np.arange(-K,H); M=K+H; CC,LC='#0072B2','#E69F00'
    f,axes=plt.subplots(1,2,figsize=(20,6.2)); f.patch.set_facecolor('white')
    tot=lambda g: (g['dC'].sum(),g['dL'].sum())
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        g=G[(nm,True,when)]; dC,dL=g['dC'][:M],g['dL'][:M]
        stack_cl(ax,t,dC,dL,CC,LC)
        ax.plot(t,dC+dL,color=INK,lw=0,marker='D',ms=7,zorder=5)
        c1,l1=tot(g); c0,l0=tot(G[(nm,False,when)])
        style(ax,f'{lab}\ntime to build: C {c1:+.2f}, L {l1:+.2f}, total {c1+l1:+.3f}\nno time to build: C {c0:+.2f}, L {l0:+.2f}, total {c0+l0:+.3f}',
              (-K-0.6,H-0.4),2,vline=False)
        ax.title.set_fontsize(17); ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0)
        ax.set_ylabel("gain, % of one period's\nSS consumption",color=MUTED,fontsize=15); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Patch(color=CC,label='from consumption (C)'),Patch(color=LC,label='from labour (L, less disutility)'),
       Line2D([],[],lw=0,marker='D',ms=9,color=INK,label='total, time to build')]
    f.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.11))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
gain_fig_exp('from t=0','figs/ttb_gain_expected0.png')
gain_fig_exp('from news','figs/ttb_gain_expected_news.png')
print('gain exp ok')

# welfare loss relative to the flexible-wage allocation: peg (top) vs optimal (bottom), consumption and labour stacked
def loss_fig(case,fname,H=9,from0=False):
    D=pickle.load(open(S+'/loss_decomp.pkl','rb')); t0=0 if (case=='unexp' or from0) else K
    TOT={k:tuple(v.sum() for v in val) for k,val in D.items()}          # totals: always the whole horizon
    if from0:   # bars for t >= 0 only (same discounting as the totals)
        D={k:tuple(v[K:] for v in val) for k,val in D.items() if k[0]==case}
    t=np.arange(-t0,H); M=t0+H; CC,LC='#0072B2','#E69F00'
    f,axes=plt.subplots(2,2,figsize=(20,9.0)); f.patch.set_facecolor('white')
    for col,(nm,lab) in enumerate((('TFP shock','TFP'),('demand shock','Demand'))):
        lims=[]
        for row,(pol,pl) in enumerate((('peg','peg at $\\rho$'),('opt','optimal rate'))):
            ax=axes[row,col]; dC,dL=[v[:M] for v in D[(case,nm,True,pol)]]
            stack_cl(ax,t,dC,dL,CC,LC); ax.plot(t,dC+dL,color=INK,lw=0,marker='D',ms=7,zorder=5)
            c,l=TOT[(case,nm,True,pol)]; n0=sum(TOT[(case,nm,False,pol)])
            style(ax,f'{lab}, {pl}\nC {c:+.2f}, L {l:+.2f}, total {c+l:+.3f}  (no time to build: {n0:+.3f})',(-t0-0.6,H-0.4),2,vline=False)
            ax.title.set_fontsize(16)
            if t0: ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0)
            if col==0: ax.set_ylabel("vs flexible wage, % of one\nperiod's SS consumption",color=MUTED,fontsize=14)
            lims.append(ax.get_ylim())
        lo=min(a for a,b in lims); hi=max(b for a,b in lims)
        for row in range(2): axes[row,col].set_ylim(lo,hi)       # same scale for peg and optimal
    for ax in axes[1]: ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Patch(color=CC,label='consumption term'),Patch(color=LC,label='labour term'),Line2D([],[],lw=0,marker='D',ms=9,color=INK,label='net, time to build')]
    f.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.08))
    if from0: f.text(0.5,-0.115,'totals include the periods before $t=0$',ha='center',fontsize=16,color=MUTED)
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
loss_fig('exp0','figs/ttb_loss_expected0.png',from0=True)
loss_fig('expnews','figs/ttb_loss_expected_news.png')
loss_fig('unexp','figs/ttb_loss_unexpected.png')
print('loss ok')

# gain from optimal policy (optimal - peg), by date, consumption and labour stacked
def policy_gain_fig(case,fname,H=9,from0=False):
    D=pickle.load(open(S+'/loss_decomp.pkl','rb')); TOT={k:tuple(v.sum() for v in val) for k,val in D.items()}
    t0=0 if (case=='unexp' or from0) else K; a=0 if case=='unexp' else K-t0
    t=np.arange(-t0,H); M=t0+H; CC,LC='#0072B2','#E69F00'
    f,axes=plt.subplots(1,2,figsize=(20,6.2)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        o,p_=D[(case,nm,True,'opt')],D[(case,nm,True,'peg')]
        gC=(o[0]-p_[0])[a:a+M]; gL=(o[1]-p_[1])[a:a+M]
        stack_cl(ax,t,gC,gL,CC,LC); ax.plot(t,gC+gL,color=INK,lw=0,marker='D',ms=7,zorder=5)
        c=TOT[(case,nm,True,'opt')][0]-TOT[(case,nm,True,'peg')][0]; l=TOT[(case,nm,True,'opt')][1]-TOT[(case,nm,True,'peg')][1]
        n0=sum(TOT[(case,nm,False,'opt')])-sum(TOT[(case,nm,False,'peg')])
        style(ax,f'{lab}: optimal $-$ peg\nC {c:+.2f}, L {l:+.2f}, total {c+l:+.3f}  (no time to build: {n0:+.3f})',(-t0-0.6,H-0.4),2,vline=False)
        ax.title.set_fontsize(17)
        if t0: ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0)
        ax.set_ylabel("gain, % of one period's\nSS consumption",color=MUTED,fontsize=15); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Patch(color=CC,label='consumption term'),Patch(color=LC,label='labour term'),Line2D([],[],lw=0,marker='D',ms=9,color=INK,label='net, time to build')]
    f.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.11))
    if from0: f.text(0.5,-0.16,'totals include the periods before $t=0$',ha='center',fontsize=16,color=MUTED)
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
policy_gain_fig('exp0','figs/ttb_policygain_expected0.png',from0=True)
policy_gain_fig('expnews','figs/ttb_policygain_expected_news.png')
policy_gain_fig('unexp','figs/ttb_policygain_unexpected.png')
print('policy gain ok')

# news horizon k: optimal rate paths (time to build, policy from the news) and welfare gain vs k
import glob
HZ={}
for fn in glob.glob(S+'/hz_*.pkl'):
    r=pickle.load(open(fn,'rb')); HZ[(r['case'],r['ttb'],r['K'],r['when'])]=r
def horizon_rate_fig(fname,T0=6,H=9):
    t=np.arange(-T0,H); M=T0+H; ks=[0,1,2,3,4,5,13]; cm=plt.get_cmap('viridis')
    f,axes=plt.subplots(1,2,figsize=(20,6.0)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        ax.step(t,np.full(M,RHO),where='post',color='0.6',lw=1.6)
        for j,k in enumerate(ks):
            if k==0: ip=np.r_[np.full(T0,RHO),100*np.r_[XU[(nm,True)]['i'],np.full(H,0.01)]][:M]
            elif k==13: ip=100*np.r_[XE[(nm,True)]['i'],np.full(M,0.01)][K-T0:K-T0+M]
            else:
                full=100*HZ[(nm,True,k,'news')]['ipath']; ip=np.r_[np.full(max(0,T0-k),RHO),full[max(0,k-T0):]][:M]
            ax.step(t,ip,where='post',color=cm(0.9*j/(len(ks)-1)),lw=2.8 if k in (0,13) else 2.2,
                    ls=(0,(4,2.5)) if k==0 else '-',label=('unexpected ($k=0$)' if k==0 else f'$k={k}$'))
        style(ax,f'{lab}: optimal nominal rate $i_t$, news $k$ periods ahead',(-T0,H-1),2)
        ax.set_ylabel('%',color=MUTED,fontsize=16); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout(); h,l=axes[0].get_legend_handles_labels()
    f.legend(h,l,loc='lower center',ncol=len(ks),frameon=False,fontsize=17,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.11),
             title='time to build, policy from the news',title_fontsize=15)
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
def horizon_gain_fig(fname):
    G=pickle.load(open(S+'/horizon_gains.pkl','rb')); ks=[0,1,2,3,4,5,13]; x=np.arange(len(ks))
    f,axes=plt.subplots(1,2,figsize=(20,6.0)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        for ttb,col in ((True,BLUE),(False,GREEN)):
            for w,ls,mk in (('news','-','o'),('shock',(0,(4,2.5)),'s')):
                ax.plot(x,[G[(nm,ttb,k,w)] for k in ks],color=col,lw=3,ls=ls,marker=mk,ms=9,mfc='white',mew=2)
        for sp in ('top','right'): ax.spines[sp].set_visible(False)
        for sp in ('left','bottom'): ax.spines[sp].set_color('0.75')
        ax.tick_params(colors=MUTED,labelsize=16,length=3); ax.grid(axis='y',color='0.9',lw=0.8); ax.set_axisbelow(True)
        ax.set_xticks(x); ax.set_xticklabels(['0\n(unexp.)','1','2','3','4','5','13'],fontsize=16,color=INK)
        ax.set_xlabel('news horizon $k$ (periods ahead)',color=MUTED,fontsize=16); ax.axhline(0,color='0.7',lw=1)
        ax.set_ylabel("gain over peg, % of one\nperiod's SS consumption",color=MUTED,fontsize=15)
        ax.set_title(f'{lab}: welfare gain from optimal policy',color=INK,fontsize=20,loc='left')
    f.tight_layout()
    h=[Line2D([],[],color=BLUE,lw=3,label='time to build'),Line2D([],[],color=GREEN,lw=3,label='no time to build'),
       Line2D([],[],color=MUTED,lw=3,marker='o',ms=9,mfc='white',mew=2,label='policy from the news'),
       Line2D([],[],color=MUTED,lw=3,ls=(0,(4,2.5)),marker='s',ms=9,mfc='white',mew=2,label='policy from the shock')]
    f.legend(handles=h,loc='lower center',ncol=4,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.13))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
print('horizon ok')

# partially expected shocks: news k = 1..5 periods ahead, policy from the news.  Same three figures as the
# unexpected section, drawn from the news date (t = -k).
sys.path.insert(0,S); os.chdir(NB); import exp_horizon as eh
# horizons with a full set of results (policy from the news, both shocks, with and without time to build)
KS=sorted(k for k in {r[2] for r in HZ} if all((nm,tb,k,'news') in HZ for nm in ('TFP shock','demand shock') for tb in (True,False)))
HM={key:eh.model(*key) for key in [(tb,k) for tb in (True,False) for k in KS]}; os.chdir(here)
print('news horizons:',KS)
def series_hz(k):
    def get(nm,ttb,t0):
        r=HZ[(nm,ttb,k,'news')]; a=k-t0; T=len(r['Y']['peg'])
        return dict(c=(r['Y']['opt'][a:],r['Y']['peg'][a:]), n=(r['Lab']['opt'][a:],r['Lab']['peg'][a:]),
                    pi=(infl(r['P']['opt'])[a:],infl(r['P']['peg'])[a:]), i=(100*r['ipath'][a:],np.full(T-a,RHO)))
    return get
def lab_hz(k):
    def get(nm,pol,t0):
        r=HZ[(nm,True,k,'news')]; A=split(r['l'][pol],r['lss'],r['tv'])[k-t0:]
        v0=100*np.expm1(HZ[(nm,False,k,'news')]['Lab'][pol][k-t0:]/100)
        return A,v0
    return get
def gain_terms(nm,ttb,k):
    r=HZ[(nm,ttb,k,'news')]; g=HM[(ttb,k)]; SSg=g['SS']; gam=g['gamma']; eta=g['eta']; chi=g['chi']; T=g['T']
    xi=np.r_[np.zeros(k),-0.01,np.zeros(T-k-1)] if nm=='demand shock' else np.zeros(T); w=g['DISC']*np.exp(xi); unit=SSg['Y']**(1-gam)/100
    C={q:SSg['Y']*np.exp(r['Y'][q]/100) for q in ('peg','opt')}; Lq={q:SSg['L']*np.exp(r['Lab'][q]/100) for q in ('peg','opt')}
    return (w*(C['opt']**(1-gam)-C['peg']**(1-gam))/(1-gam)/unit, -w*chi*(Lq['opt']**(1+eta)-Lq['peg']**(1+eta))/(1+eta)/unit)
def hz_gain_fig(k,fname,H=9):
    t=np.arange(-k,H); M=k+H; CC,LC='#0072B2','#E69F00'
    f,axes=plt.subplots(1,2,figsize=(20,6.2)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        gC,gL=[v[:M] for v in gain_terms(nm,True,k)]; c,l=[v.sum() for v in gain_terms(nm,True,k)]
        n0=sum(v.sum() for v in gain_terms(nm,False,k))
        stack_cl(ax,t,gC,gL,CC,LC); ax.plot(t,gC+gL,color=INK,lw=0,marker='D',ms=7,zorder=5)
        style(ax,f'{lab}: optimal $-$ peg\nC {c:+.2f}, L {l:+.2f}, total {c+l:+.3f}  (no time to build: {n0:+.3f})',(-k-0.6,H-0.4),2,vline=False)
        ax.title.set_fontsize(17); ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0)
        ax.set_ylabel("gain, % of one period's\nSS consumption",color=MUTED,fontsize=15); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Patch(color=CC,label='consumption term'),Patch(color=LC,label='labour term'),Line2D([],[],lw=0,marker='D',ms=9,color=INK,label='net, time to build')]
    f.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=18,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.11))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
for k in KS:
    shock_fig(series_hz(k),f'figs/ttb_news{k}.png',t0=k)
    labour_fig(lab_hz(k),f'figs/ttb_news{k}_labour.png',t0=k)
    hz_gain_fig(k,f'figs/ttb_news{k}_gain.png')
print('news k ok')

# one shock per figure, 2 x 2 (consumption, employment / inflation, nominal rate)
def shock_fig_one(get,nm,fname,t0=0,H=14):
    t=np.arange(-t0,H); M=t0+H
    f,axes=plt.subplots(2,2,figsize=(20,8.6)); f.patch.set_facecolor('white'); ax4=axes.ravel()
    for ttb,col in ((True,BLUE),(False,GREEN)):
        g=get(nm,ttb,t0)
        for k,key in enumerate(('c','n','pi')):
            ax4[k].plot(t,g[key][0][:M],color=col,**OPT); ax4[k].plot(t,g[key][1][:M],color=col,**PEG)
        ax4[3].step(t,g['i'][0][:M],where='post',color=col,**OPT)
    ax4[3].step(t,np.full(M,RHO),where='post',color=MUTED,**PEG)
    for k,(ttl,yl) in enumerate((('consumption $c_t$','% from SS'),('employment $n_t$','% from SS'),('inflation $\\pi_t$','%'),('nominal rate $i_t$','%'))):
        style(ax4[k],ttl,(-t0,H-1),2 if t0<=5 else 4); ax4[k].set_ylabel(yl,color=MUTED,fontsize=16)
    for ax in axes[1]: ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Line2D([],[],color=BLUE,lw=3.2,label='time to build'),Line2D([],[],color=GREEN,lw=3.2,label='no time to build'),
       Line2D([],[],color=MUTED,lw=3.2,label='optimal rate'),Line2D([],[],color=MUTED,lw=2.6,ls=(0,(1.2,1.6)),label=r'peg at $\rho$')]
    f.legend(handles=h,loc='lower center',ncol=4,frameon=False,fontsize=19,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.07))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
SPLIT={'ttb_expected0':(series_exp0,0),'ttb_expected_pre':(series_exp,K),'ttb_unexpected':(series_unexp,0)}
for k in KS: SPLIT[f'ttb_news{k}']=(series_hz(k),k)
for base,(get,t0) in SPLIT.items():
    for nm,tag in (('TFP shock','tfp'),('demand shock','dem')):
        shock_fig_one(get,nm,f'figs/{base}_{tag}.png',t0=t0)
print('split ok')

# peg only ("no policy change"): one shock per figure; 4th panel = real rate rho - pi_{t+1}
def shock_fig_peg(get,nm,fname,t0=0,H=14):
    t=np.arange(-t0,H); M=t0+H
    f,axes=plt.subplots(2,2,figsize=(20,8.6)); f.patch.set_facecolor('white'); ax4=axes.ravel()
    for ttb,col in ((True,BLUE),(False,GREEN)):
        g=get(nm,ttb,t0)
        for k,key in enumerate(('c','n','pi')):
            ax4[k].plot(t,g[key][1][:M],color=col,lw=3.2)
        pi=np.r_[g['pi'][1],0.0]
        ax4[3].step(t,(RHO-pi[1:])[:M],where='post',color=col,lw=3.2)
    ax4[3].axhline(RHO,color='0.6',lw=1.6)
    for k,(ttl,yl) in enumerate((('consumption $c_t$','% from SS'),('employment $n_t$','% from SS'),('inflation $\\pi_t$','%'),
                                 ('real rate $i_t-\\pi_{t+1}$ (nominal rate fixed at $\\rho$)','%'))):
        style(ax4[k],ttl,(-t0,H-1),2 if t0<=5 else 4); ax4[k].set_ylabel(yl,color=MUTED,fontsize=16)
    for ax in axes[1]: ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Line2D([],[],color=BLUE,lw=3.2,label='time to build'),Line2D([],[],color=GREEN,lw=3.2,label='no time to build'),
       Line2D([],[],color='0.6',lw=1.6,label=r'$\rho$')]
    f.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=19,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.12),
             title=r'peg: nominal rate held at $\rho$',title_fontsize=17)
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
for base,(get,t0) in SPLIT.items():
    for nm,tag in (('TFP shock','tfp'),('demand shock','dem')):
        shock_fig_peg(get,nm,f'figs/{base}_{tag}_peg.png',t0=t0)
print('peg ok')

# short impulse-response deck: bold red (time to build) and black (none)
RB=('#d00000','#000000')
shock_fig(series_exp0,'figs/irf_expected0.png',cols=RB)
shock_fig(series_exp,'figs/irf_expected_pre.png',t0=K,cols=RB)
shock_fig(series_unexp,'figs/irf_unexpected.png',cols=RB)
print('irf ok')

# firm profits paid to households (cash flow): Pi_t / P_t = Y_t - (W/P_t) L_t, change from SS in % of SS output
SSP={tb:HM[(tb,1)]['SS'] for tb in (True,False)}
def real_profit(Ylog,Llog,Plog,tb):
    ss=SSP[tb]; Y=ss['Y']*np.exp(Ylog/100); L=ss['L']*np.exp(Llog/100); P=ss['P']*np.exp(Plog/100)
    return 100*((Y-ss['W']/P*L)-(ss['Y']-ss['W']/ss['P']*ss['L']))/ss['Y']
def pg_exp(R):
    def get(nm,tb,t0):
        r=R[(nm,tb)]; a=K-t0
        return {q:real_profit(r['Y'][q],r['Lab'][q],r['P'][q],tb)[a:] for q in ('opt','peg')}
    return get
def pg_unexp(nm,tb,t0):
    r=XU[(nm,tb)]; z=np.zeros(t0)
    return {q:np.r_[z,real_profit(r['Y'][q],r['L'][q],r['P'][q],tb)] for q in ('opt','peg')}
def pg_hz(k):
    def get(nm,tb,t0):
        r=HZ[(nm,tb,k,'news')]; a=k-t0
        return {q:real_profit(r['Y'][q],r['Lab'][q],r['P'][q],tb)[a:] for q in ('opt','peg')}
    return get
def profit_fig(get,fname,t0=0,H=14):
    t=np.arange(-t0,H); M=t0+H
    f,axes=plt.subplots(1,2,figsize=(20,6.0)); f.patch.set_facecolor('white')
    for ax,(nm,lab) in zip(axes,(('TFP shock','TFP'),('demand shock','Demand'))):
        for tb,col in ((True,BLUE),(False,GREEN)):
            g=get(nm,tb,t0); ax.plot(t,g['opt'][:M],color=col,**OPT); ax.plot(t,g['peg'][:M],color=col,**PEG)
        style(ax,f'{lab}: real profits $Y_t-(\\bar W/P_t)L_t$',(-t0,H-1),2 if t0<=5 else 4)
        ax.set_ylabel('change, % of SS output',color=MUTED,fontsize=16); ax.set_xlabel('t',color=MUTED,fontsize=16)
    f.tight_layout()
    h=[Line2D([],[],color=BLUE,lw=3.2,label='time to build'),Line2D([],[],color=GREEN,lw=3.2,label='no time to build'),
       Line2D([],[],color=MUTED,lw=3.2,label='optimal rate'),Line2D([],[],color=MUTED,lw=2.6,ls=(0,(1.2,1.6)),label=r'peg at $\rho$')]
    f.legend(handles=h,loc='lower center',ncol=4,frameon=False,fontsize=19,labelcolor=MUTED,bbox_to_anchor=(0.5,-0.12))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
profit_fig(pg_exp(X0),'figs/ttb_expected0_profit.png')
profit_fig(pg_exp(XE),'figs/ttb_expected_pre_profit.png',t0=K)
profit_fig(pg_unexp,'figs/ttb_unexpected_profit.png')
for k in KS: profit_fig(pg_hz(k),f'figs/ttb_news{k}_profit.png',t0=k)
# sanity: no time to build -> profits identically zero
print('max |profit| without time to build:', max(np.max(np.abs(pg_unexp(nm,False,0)[q])) for nm in ('TFP shock','demand shock') for q in ('opt','peg')))
print('profit ok')

# summary across news horizons k = 0 (unexpected), 1..12, 13 (fully expected); policy from the news
def gain_hz(nm,tb,k):
    if k==0:
        D=pickle.load(open(S+'/loss_decomp.pkl','rb')); return sum(D[('unexp',nm,tb,'opt')][q].sum()-D[('unexp',nm,tb,'peg')][q].sum() for q in (0,1))
    if k==K:
        T13=pickle.load(open(S+'/gain_timing.pkl','rb')); return sum(T13[(nm,tb,'from news')][q].sum() for q in ('dC','dL'))
    return sum(v.sum() for v in gain_terms(nm,tb,k))
def rate_hz(nm,k,T0,H):          # time to build, policy from the news, on t = -T0..H-1
    M=T0+H
    if k==0: return np.r_[np.full(T0,RHO),100*np.r_[XU[(nm,True)]['i'],np.full(M,0.01)]][:M]
    full=100*(np.r_[XE[(nm,True)]['i'],np.full(M,0.01)] if k==K else HZ[(nm,True,k,'news')]['ipath'])
    return np.r_[np.full(max(0,T0-k),RHO),full[max(0,k-T0):]][:M]
KALL=[0]+KS+[K]
gain13=lambda nm,tb,k: gain_hz(nm,tb,k)*(1+RHO/100)**(k-K)     # all horizons discounted to t = -K (the k = 13 news date)
def horizon_summary_fig(fname,T0=K,H=9):
    t=np.arange(-T0,H); M=T0+H; cm=plt.get_cmap('turbo')
    col=lambda j: cm(0.05+0.9*j/(len(KALL)-1))
    f,axes=plt.subplots(2,2,figsize=(20,9.4),gridspec_kw={'width_ratios':[1.55,1]}); f.patch.set_facecolor('white')
    for row,(nm,lab) in enumerate((('TFP shock','TFP'),('demand shock','Demand'))):
        ax=axes[row,0]; lo_t,hi_t=-6,4                       # grouped bars: i_t - rho, one bar per horizon k
        sel=(t>=lo_t)&(t<=hi_t); tt=t[sel]; nk=len(KALL); bw=0.86/nk
        for j,k in enumerate(KALL):
            dev=(rate_hz(nm,k,T0,H)-RHO)[sel]
            ax.bar(tt-0.43+bw*(j+0.5),dev,width=bw,color=col(j),edgecolor='none',hatch='//' if k==0 else None)
        style(ax,f'{lab}: optimal rate minus $\\rho$ (time to build)',(lo_t-0.6,hi_t+0.6),1,vline=False)
        ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0)
        ax.set_ylabel('percentage points',color=MUTED,fontsize=15)
        ax=axes[row,1]; x=np.array(KALL)
        g1=[gain13(nm,True,k) for k in KALL]; g0=[gain13(nm,False,k) for k in KALL]
        ax.plot(x,g1,color=BLUE,lw=2.6,zorder=2); ax.plot(x,g0,color=GREEN,lw=2.6,zorder=2)
        ax.scatter(x,g1,c=[col(j) for j in range(len(KALL))],s=90,edgecolor=BLUE,lw=1.5,zorder=3)
        ax.scatter(x,g0,c=[col(j) for j in range(len(KALL))],s=90,edgecolor=GREEN,lw=1.5,zorder=3,marker='s')
        style(ax,f'{lab}: welfare gain over the peg (valued at $t=-13$)',(-0.5,K+0.5),1); ax.set_xticks(KALL)
        ax.set_ylabel("% of one period's\nSS consumption",color=MUTED,fontsize=14)
    for ax in axes[1]: ax.set_xlabel('t' if ax is axes[1,0] else 'news horizon $k$ (0 = unexpected)',color=MUTED,fontsize=15)
    f.tight_layout()
    hk=[Patch(facecolor=col(j),hatch='//' if k==0 else None,label=f'{k}') for j,k in enumerate(KALL)]
    l1=f.legend(handles=hk,loc='lower center',ncol=len(KALL),frameon=False,fontsize=15,labelcolor=MUTED,bbox_to_anchor=(0.38,-0.075),
                title='news horizon $k$ (periods ahead; 0 = unexpected)',title_fontsize=14,columnspacing=1.0,handlelength=1.6)
    f.legend(handles=[Line2D([],[],color=BLUE,lw=2.6,marker='o',ms=9,mfc='white',label='time to build'),
                      Line2D([],[],color=GREEN,lw=2.6,marker='s',ms=9,mfc='white',label='no time to build')],
             loc='lower center',ncol=2,frameon=False,fontsize=15,labelcolor=MUTED,bbox_to_anchor=(0.82,-0.055))
    f.savefig(fname,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
horizon_summary_fig('figs/ttb_horizon_summary.png')
# table numbers (TFP, time to build): gains, i_{-1}, i_0, hiring at t = 0 under the optimal rate
rows=[]
for k in KALL:
    if k==0: r=XU[('TFP shock',True)]; im1,i0,n0=np.nan,100*r['i'][0],r['L']['opt'][0]
    elif k==K: r=XE[('TFP shock',True)]; im1,i0,n0=100*r['i'][K-1],100*r['i'][K],r['Lab']['opt'][K]
    else: r=HZ[('TFP shock',True,k,'news')]; im1,i0,n0=100*r['ipath'][k-1],100*r['ipath'][k],r['Lab']['opt'][k]
    rows.append((k,gain13('TFP shock',True,k),gain13('TFP shock',False,k),gain13('demand shock',True,k),gain13('demand shock',False,k),im1,i0,n0))
pickle.dump(rows,open(S+'/horizon_table.pkl','wb'))
for r in rows: print('k=%2d  gainTFP %.3f / %.3f  gainDem %.4f / %.4f  i_-1 %s  i_0 %.2f  n_0 %.2f'%(r[0],r[1],r[2],r[3],r[4],'  -  ' if np.isnan(r[5]) else '%.2f'%r[5],r[6],r[7]))
print('summary ok')
