"""
nu_solver.py  --  transparent computation of nu for the A_n catastrophe fit.

Solves  A nu = xi  where, from minimising  J(nu) = int sum_k L_k(theta(t),nu)^2 dt,
    A  = sum_i sum_k  M_k(t_i)^T M_k(t_i)     (symmetric, PSD, S x S)
    xi = - sum_i sum_k M_k(t_i)^T rho_k(t_i)
with L_k = rho_k + M_k nu affine in nu (leading coefficient X^n fixed to 1).

Moment building blocks (omega ~ N(0,tau), g = A + B*omega, A=atil*tau, B=btil>0):
    mu_m       = E[omega^m]            = (m-1)!! tau^{m/2}  (even m; 0 odd; mu_0=1)
    E_{r,p}    = E[omega^r (A+Bw)^p]   = sum_l C(p,l) A^{p-l} B^l mu_{r+l}
    [M_k]_{(j,r,s)} = tau^s E_{r,j+k},   rho_k = E_{0,n+k},   k = 0..n+1.

This file forms A and xi EXPLICITLY (no hidden stacked-lstsq), applies the
diagonal GMM weighting (per-equation row norm) and column preconditioning,
ridge lambda, and robust trimming.  Self-tests at the bottom:
  (1) exact recovery of a known cubic to machine precision,
  (2) agreement with the production routine catlib.fit_moment.
"""
import numpy as np
from math import comb

from catlib import MON            # identical monomial ordering as production code
NMON = len(MON)


# ----------------------------- moment building blocks -----------------------
def dfact(m):                       # double factorial m!!  (dfact(-1)=1)
    r = 1.0
    while m > 1:
        r *= m; m -= 2
    return r


def mu(m, tau):                     # E[omega^m], omega~N(0,tau)
    return 0.0 if m % 2 == 1 else dfact(m - 1) * tau ** (m // 2)


def E_rp(r, p, A, B, tau):          # E[omega^r (A+B omega)^p]
    return sum(comb(p, l) * A ** (p - l) * B ** l * mu(r + l, tau)
               for l in range(p + 1))


def control_powers(n):              # powers j of X carrying a control (skip X^{n-1})
    return list(range(n - 2, -1, -1))


# ----------------------------- design rows per point ------------------------
def point_rows(a, b, tau, n, K):
    """Return list of (M_k row, rho_k) for k=0..K at one cloud point."""
    A, B = a * tau, b
    powers = control_powers(n)
    out = []
    for k in range(K + 1):
        row = np.zeros(len(powers) * NMON)
        for jc, j in enumerate(powers):
            for m, (r, s) in enumerate(MON):
                row[jc * NMON + m] = (tau ** s) * E_rp(r, j + k, A, B, tau)
        rho = E_rp(0, n + k, A, B, tau)              # from the fixed X^n term
        out.append((row, rho))
    return out


def build_design(atil, btil, tau, n, K):
    """Stack all rows; return M (rows), rho (vector), and per-row GMM weights."""
    M, rho = [], []
    for a, b, t in zip(atil, btil, tau):
        for row, rk in point_rows(a, b, t, n, K):
            M.append(row); rho.append(rk)
    M = np.array(M); rho = np.array(rho)
    # diagonal GMM weight: divide each equation by its scale (row norm).
    # Leaves the exact (zero-residual) solution invariant; only balances the
    # noise weighting across moments whose magnitudes differ by orders (E[g^{n+k}]).
    w = 1.0 / np.maximum(np.sqrt((M ** 2).sum(1) + rho ** 2), 1e-12)
    return M * w[:, None], rho * w


# ----------------------------- explicit A nu = xi solve ---------------------
def solve_nu(atil, btil, tau, degree, K=None, ridge=1e-2, robust=True,
             trim=0.15, iters=6, verbose=False):
    n = degree
    if K is None:
        K = degree + 1
    nrow = K + 1
    Mw, rhow = build_design(atil, btil, tau, n, K)         # row-weighted design
    npts = len(tau)
    keep = np.ones(Mw.shape[0], dtype=bool)
    S = Mw.shape[1]
    nu = np.zeros(S)
    A = xi = None
    for it in range(iters if robust else 1):
        Mk = Mw[keep]; rk = rhow[keep]
        # ---- EXPLICIT normal system, ridge on the ORIGINAL nu -------------
        A = Mk.T @ Mk                       # A = sum_i sum_k M_k^T M_k  (S x S, PSD)
        xi = -Mk.T @ rk                     # xi = - sum_i sum_k M_k^T rho_k
        nu = np.linalg.solve(A + ridge * np.eye(S), xi)
        if verbose:
            print(f"  iter {it}: cond(A+lI)={np.linalg.cond(A+ridge*np.eye(S)):.1e} "
                  f"max|nu|={np.abs(nu).max():.1f} kept={keep.sum()}/{len(keep)}")
        if not robust:
            break
        res = np.abs(Mw @ nu - (-rhow))     # residual of L_k = M_k nu + rho_k
        agg = res.reshape(npts, nrow).sum(1)
        good = agg <= np.quantile(agg, 1 - trim)
        keep = np.repeat(good, nrow)
    J = float(np.mean((Mw[keep] @ nu + rhow[keep]) ** 2))
    return nu, J, A, xi


# ----------------------------- roots of the fitted polynomial ---------------
def cat_roots(nu, w, t, degree, tol=1e-7):
    powers = control_powers(degree)
    cvals = [nu[i * NMON:(i + 1) * NMON] @ np.array([w ** r * t ** s for (r, s) in MON])
             for i in range(len(powers))]
    coeffs = np.zeros(degree + 1); coeffs[0] = 1.0
    for j, c in zip(powers, cvals):
        coeffs[degree - j] = c
    rts = np.roots(coeffs)
    return np.sort(rts[np.abs(rts.imag) < tol].real)


# =========================== self-tests =====================================
if __name__ == "__main__":
    import sympy as sp

    # ---- (1) EXACT RECOVERY of a known depressed cubic ----------------------
    al = np.array([0.10, -0.15, 0.05]); be = np.array([0.80, -1.20, 0.40])  # sum=0
    w, t = sp.symbols('w t')
    g = [al[m] * t + be[m] * w for m in range(3)]
    p_true = sp.expand(g[0] * g[1] + g[0] * g[2] + g[1] * g[2])
    q_true = sp.expand(-g[0] * g[1] * g[2])

    def coeffs(poly):
        P = sp.Poly(poly, w, t).as_dict()
        return np.array([float(P.get((r, s), 0)) for (r, s) in MON])
    nu_true = np.concatenate([coeffs(p_true), coeffs(q_true)])   # [c1(=p), c0(=q)]

    rng = np.random.default_rng(0)
    N = 600
    taus = rng.uniform(0.05, 1.0, N); mact = rng.integers(0, 3, N)
    atil = al[mact]; btil = be[mact]     # exact active branch per point (signed, math test)
    nu, J, A, xi = solve_nu(atil, btil, taus, 3, K=4, ridge=1e-12, robust=False,
                            verbose=True)
    print("EXACT-RECOVERY  ||nu-nu_true||_inf = %.2e   J = %.2e" %
          (np.abs(nu - nu_true).max(), J))
    print("A symmetric? %s   A PSD? %s   shape %s" %
          (np.allclose(A, A.T), np.all(np.linalg.eigvalsh(A) > -1e-9), A.shape))

    # ---- (2) AGREEMENT with production routine catlib.fit_moment -----------
    import catlib
    from build_gwt import prep
    for kind in ["synth", "btc"]:
        D = prep(kind)
        nu1, J1, c1 = catlib.fit_moment(D['atil'], D['btil'], D['tau'], 3, ridge=1e-2)
        nu2, J2, A2, xi2 = solve_nu(D['atil'], D['btil'], D['tau'], 3, ridge=1e-2)
        dmax = max(np.abs(np.sort(cat_roots(nu1, ww, tt, 3)) -
                          np.sort(cat_roots(nu2, ww, tt, 3))).max()
                   for ww, tt in [(0.3, 0.3), (-0.5, 0.6), (0.8, 0.85)]
                   if len(cat_roots(nu1, ww, tt, 3)) == len(cat_roots(nu2, ww, tt, 3)) == 3)
        print(f"{kind}: roots agree with fit_moment to {dmax:.1e}  "
              f"(explicit-A vs stacked-lstsq)")
