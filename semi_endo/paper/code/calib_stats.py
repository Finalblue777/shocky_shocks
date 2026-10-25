import sys
exec(open('nodebt.py').read().split('# steady-state weights')[0])
g_=g; c=g_['c']; beta=g_['beta']; beta1=g_['beta1']; N=np.asarray(g_['N_ARRAY']); dt=np.asarray(g_['data_tau'])
print('kappa',c,' delta',beta,' i alt (beta1):',1/beta1-1)
print('data tau (months): mean %.2f median %.2f min %.2f max %.2f'%(dt.mean(),np.median(dt),dt.min(),dt.max()))
print('N: min %.2f median %.2f max %.2f'%(N.min(),np.median(N),N.max()))
print('tau counts by value:',np.bincount(TV)); print('share tau=0 producers',np.mean(TV==0))
print('max tau',TM,' mean tau (unweighted)',TV.mean())
s_j=SS['p']*SS['y']/(SS['P']*SS['Y']); lam=SS['l']/SS['l'].sum()
for nm,w in (('exp',s_j),('emp',lam)):
    print(nm,'shares by lag 0..6+:',np.round(100*np.r_[np.bincount(TV,weights=w)[:6],w[TV>=6].sum()],1))
# gap g_j
D=np.arange(0,40)
z=lambda d: (c*beta)**d*(d+N[:,None])
Z=z(D[None,:]); zt=Z[np.arange(len(TV)),TV]
other=np.where(D[None,:]!=TV[:,None],Z,-np.inf).max(1)
gap=1-other/zt
print('gap: min %.4f  p10 %.4f median %.4f max %.4f'%(gap.min(),np.quantile(gap,.1),np.median(gap),gap.max()))
for th in (0.001,0.005,0.01,0.03,0.05):
    print(' gap<%.3f: producers %.1f%%  expenditure %.1f%%'%(th,100*np.mean(gap<th),100*s_j[gap<th].sum()))
print('Theta/Atil',SS['Theta']/SS['A_til'],' profit share of output',1-SS['A_til']/SS['Theta'])
print('SS L, Y',SS['L'],SS['Y'])
