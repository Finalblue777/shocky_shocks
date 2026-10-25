# Rotemberg union wage setting for the fixed-wage cohort model (robustness to the fixed nominal wage).
#
# Unions set the wage of differentiated labour (elasticity EPS_W) and pay a utility cost
# (PHI_W/2) (pi^w_t - 1)^2 to change it.  An employment subsidy removes the steady-state wage markup, so the
# steady state is the same as in the fixed-wage model.  Symmetric first-order condition:
#   PHI_W (pi^w_t - 1) pi^w_t = EPS_W L_t C_t^{-sigma} (chi L_t^eta C_t^sigma - W_t/P_t)
#                               + beta (theta_{t+1}/theta_t) PHI_W (pi^w_{t+1} - 1) pi^w_{t+1},
# with W_{-1} = W_ss and pi^w = 1 after the horizon.  PHI_W is set so that the linearised slope matches a Calvo
# wage contract lasting DUR months on average (Born and Pfeifer, 2020).  Welfare subtracts the adjustment cost.
import numpy as np
from scipy.optimize import root

EPS_W = 10.0

def add_union(g, dur):
    """Replace g['solve'] with the union version (phi = 1) / flexible wage (phi = 0)."""
    SS, T, beta, gam, eta, chi = g['SS'], g['T'], g['beta_hh'], g['gamma'], g['eta'], g['chi']
    alloc, Y_FLR, DISC, rho = g['alloc'], g['Y_FLR'], g['DISC'], g['rho']
    th = 1 - 1 / dur
    kappa_w = (1 - th) * (1 - beta * th) / (th * (1 + EPS_W * eta))
    scale = EPS_W * SS['L'] * SS['Y'] ** (-gam) * SS['omega']           # linear slope = scale / PHI_W
    PHI_W = scale / kappa_w
    flex_solve = g['solve']

    def solve(i_path=None, A_path=None, xi=None, phi=1.0, key=None):
        if phi == 0.0:
            return flex_solve(i_path, A_path, xi, phi=0.0, key=key)
        i_path = np.full(T, rho) if i_path is None else np.asarray(i_path, float)
        A_path = np.ones(T) if A_path is None else np.asarray(A_path, float)
        xi = np.zeros(T) if xi is None else np.asarray(xi, float)

        def resid(u):
            W = np.exp(u[:T]); Y = Y_FLR + np.exp(u[T:])
            P, _, _, _, L = alloc(W, Y, i_path, A_path)
            Px = np.r_[P, SS['P']]; Yx = np.r_[Y, SS['Y']]; xix = np.r_[xi, 0.0]
            piw = W / np.r_[SS['W'], W[:-1]]; f = PHI_W * (piw - 1) * piw; fx = np.r_[f[1:], 0.0]
            wpc = f - EPS_W * L * Y ** (-gam) * (chi * L ** eta * Y ** gam - W / P) - beta * np.exp(xix[1:] - xix[:T]) * fx
            eu = (xix[:T] - gam * np.log(Yx[:T]) - np.log(Px[:T])) \
               - (np.log(beta) + np.log(1 + i_path) + xix[1:] - gam * np.log(Yx[1:]) - np.log(Px[1:]))
            return np.r_[wpc / scale, eu]

        u0 = np.r_[np.full(T, np.log(SS['W'])), np.log(SS['Y'] - Y_FLR)]
        s = root(resid, u0, method='hybr', tol=1e-13)
        W = np.exp(s.x[:T]); Y = Y_FLR + np.exp(s.x[T:])
        P, p, y, l, L = alloc(W, Y, i_path, A_path)
        piw = W / np.r_[SS['W'], W[:-1]]
        cost = PHI_W / 2 * (piw - 1) ** 2
        U = float(np.sum(DISC * np.exp(xi) * (Y ** (1 - gam) / (1 - gam) - chi * L ** (1 + eta) / (1 + eta) - cost)))
        return dict(W=W, Y=Y, C=Y, P=P, p=p, y=y, l=l, L=L, i=i_path, A=A_path, xi=xi, phi=phi, omega=W / P,
                    ok=bool(np.max(np.abs(s.fun)) < 1e-8), res=float(np.max(np.abs(s.fun))), welfare=U, wcost=cost, piw=piw)

    g['solve'] = solve; g['PHI_W'] = PHI_W; g['kappa_w'] = kappa_w
    return g
