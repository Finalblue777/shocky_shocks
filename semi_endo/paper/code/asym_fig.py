# slide-109-style summary for the planner-informed case
import sys, pickle, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
S=sys.argv[1]; K=13; RHO=1.0
RED,BLACK,INK,MUTED='#d00000','#000000','#0b0b0b','#52514e'
plt.rcParams.update({'font.size':13})
XU={(r['case'],r['ttb']):r for r in pickle.load(open(S+'/ttb_unexp.pkl','rb'))}
full={r[0]:(r[3],r[6]) for r in pickle.load(open(S+'/horizon_loss_table.pkl','rb'))}     # everyone informed, at t=-13
KA=[1,2,3,5,8,13]
AS={(nm,k):pickle.load(open(f"{S}/asym_ttb_{nm[:3]}_{k}.pkl",'rb')) for nm in ('TFP shock','demand shock') for k in KA}
KALL=list(range(14)); cm=plt.get_cmap('turbo'); col=lambda k: cm(0.05+0.9*k/13)       # same colours as slide 109
def style(ax,title,xr):
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color('0.75')
    ax.tick_params(colors=MUTED,labelsize=15,length=3); ax.grid(axis='y',color='0.9',lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0,color='0.7',lw=1,zorder=1); ax.set_xlim(*xr); ax.set_title(title,color=INK,fontsize=19,loc='left')
def rate(nm,k,lo,hi):
    t=np.arange(lo,hi+1)
    if k==0: path=np.r_[np.full(-lo,RHO),100*np.r_[XU[(nm,True)]['i'],np.full(30,0.01)]][:len(t)]
    else:
        ip=100*AS[(nm,k)]['ipath']; path=np.array([ip[k+s] if k+s>=0 else RHO for s in t])
    return t,path-RHO
f,axes=plt.subplots(2,2,figsize=(20,9.4),gridspec_kw={'width_ratios':[1.55,1]}); f.patch.set_facecolor('white')
ks=[0]+KA; nk=len(ks); bw=0.86/nk
for row,(nm,lab,j,sc,unit) in enumerate((('TFP shock','TFP',0,1,''),('demand shock','Demand',1,1000,r' ($\times10^{-3}$)'))):
    ax=axes[row,0]
    for n,k in enumerate(ks):
        t,dev=rate(nm,k,-6,4)
        ax.bar(t-0.43+bw*(n+0.5),dev,width=bw,color=col(k),edgecolor='none',hatch='//' if k==0 else None)
    style(ax,f'{lab}: optimal rate minus $\\rho$, planner informed',(-6.6,4.6)); ax.set_xticks(range(-6,5))
    ax.axvline(-0.5,color='0.6',lw=1,ls='--',zorder=0); ax.set_ylabel('percentage points',color=MUTED,fontsize=15)
    ax=axes[row,1]
    ge=[sc*full[k][j] for k in KALL]
    gp=[sc*full[0][j]]+[sc*AS[(nm,k)]['gain']*1.01**(k-K) for k in KA]
    ax.plot(KALL,ge,color=RED,lw=2.6,zorder=2,label='everyone informed')
    ax.plot([0]+KA,gp,color=RED,lw=2.6,ls=(0,(4,2.5)),zorder=2,label='planner informed only')
    ax.axhline(sc*full[0][j],color=BLACK,lw=1.6,ls=':',zorder=1,label='no warning (unexpected)')
    ax.scatter(KALL,ge,c=[col(k) for k in KALL],s=80,edgecolor=RED,lw=1.5,zorder=3)
    ax.scatter([0]+KA,gp,c=[col(k) for k in [0]+KA],s=90,edgecolor=RED,lw=1.5,zorder=3,marker='s')
    style(ax,f'{lab}: gain over the peg{unit}',(-0.5,13.5)); ax.set_xticks(KALL)
    ax.set_ylabel("% of one period's SS\nconsumption, at $t=-13$",color=MUTED,fontsize=14)
axes[1,0].set_xlabel('t',color=MUTED,fontsize=15); axes[1,1].set_xlabel('news horizon $k$ (0 = unexpected)',color=MUTED,fontsize=15)
f.tight_layout()
hk=[Patch(facecolor=col(k),hatch='//' if k==0 else None,label=str(k)) for k in ks]
f.legend(handles=hk,loc='lower center',ncol=len(ks),frameon=False,fontsize=15,labelcolor=MUTED,bbox_to_anchor=(0.32,-0.075),
         title='news horizon $k$ (planner informed; 0 = unexpected)',title_fontsize=14)
f.legend(handles=[Line2D([],[],color=RED,lw=2.6,marker='o',ms=8,mfc='white',label='everyone informed'),
                  Line2D([],[],color=RED,lw=2.6,ls=(0,(4,2.5)),marker='s',ms=8,mfc='white',label='planner informed only'),
                  Line2D([],[],color=BLACK,lw=1.6,ls=':',label='no warning')],
         loc='lower center',ncol=3,frameon=False,fontsize=14,labelcolor=MUTED,bbox_to_anchor=(0.80,-0.055))
f.savefig('figs/ttb_horizon_summary_planner.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close(f)
print('ok')
