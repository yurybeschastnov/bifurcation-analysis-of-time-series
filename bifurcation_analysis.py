#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Bifurcation / catastrophe analysis of multiregime time series
 Complete reference implementation: numerics + ALL figure-generating code.

 Companion code for:
   Yu. N. Beschastnov, G. G. Malinetsky,
   "Analysis of multiregime time series by methods of bifurcation and
    catastrophe theory".
   https://github.com/yurybeschastnov/bifurcation-analysis-of-time-series

 METHOD
   Regimes are the branches (roots) of ONE implicit stochastic equation
       f(X_t, W_t, t; nu) = 0,
   and a regime switch is a bifurcation (merging of two roots). The identity
   holds almost surely, so it is replaced by its moment consequences
       L_k = E_W[ f(g, W, t; nu) * g^k ] = 0,   k = 0..K,
   which are LINEAR in the global parameter nu:
       L_k = rho_k + M_k nu    =>    A nu = xi,
       A  =  int sum_k M_k^T M_k dt,      xi = -int sum_k M_k^T rho_k dt.
   nu is found by ridge-regularised least squares, with the ridge applied to the
   ORIGINAL nu (not to column-scaled coordinates); see `fit_moment`.

 IDENTIFIABILITY
   From a single trajectory only the MAGNITUDE b = |b| is identifiable (via the
   quadratic variation). The sign of beta is not: W_t is unobserved and
   symmetric, so beta and -beta induce the same law. Everything uses b >= 0.

 CONTENTS
   Part 0  Plot style (colours, font size 12, 350 dpi)
   Part 1  Synthetic data generator (three branches, one Wiener path)
   Part 2  Core library A: monomials, Gaussian moments, local theta(t),
           product-model fit, silhouette, cubic roots
   Part 3  Core library B: moment conditions, A nu = xi, catastrophe roots
   Part 4  Transparent standalone solver for A nu = xi (+ self-tests)
   Part 5  Figures, Experiment 1 - regime identification
   Part 6  Figures, product model on synthetic data (unsigned b)
   Part 7  Figures, Experiment 2 - catastrophe order selection A3 vs A5
   Part 8  Symbolic verification of the algebra (sympy)
   Part 9  Command-line entry point

 USAGE
   python bifurcation_analysis.py exp1       # Experiment 1 figures
   python bifurcation_analysis.py prod       # product-model figures
   python bifurcation_analysis.py exp2       # Experiment 2 figures
   python bifurcation_analysis.py selftest   # solver self-tests
   python bifurcation_analysis.py verify     # sympy verification
   python bifurcation_analysis.py all        # everything above

 REQUIREMENTS
   numpy, pandas, scikit-learn, matplotlib  (sympy only for `verify`/`selftest`)
   Data file `btc_week_1min.csv` with a `close` column, for the BTC figures.
================================================================================
"""

import os
import sys
from math import comb
from itertools import product as iproduct

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = "figs/"
os.makedirs(OUT, exist_ok=True)



# ==============================================================================
# PART 0.  PLOT STYLE
#   Colours, fonts (12 pt on figures), 350 dpi output.
# ==============================================================================



rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.25
rcParams["grid.linewidth"] = 0.6
rcParams["font.size"] = 12
rcParams["axes.titlesize"] = 13
rcParams["axes.labelsize"] = 12
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["legend.fontsize"] = 12
rcParams["figure.dpi"] = 130
rcParams["savefig.dpi"] = 350
rcParams["savefig.bbox"] = "tight"

# regime colours: 0=calm(blue) 1=mid(orange) 2=turbulent(green)
REG = ["#3b6fb6", "#e07b39", "#3f9b5c"]
ACCENT = "#c0392b"
TRUE = "#1f3b73"


# ==============================================================================
# PART 1.  SYNTHETIC DATA GENERATOR
#   Three linear branches g_n = alpha_n t + beta_n W_t driven by ONE Wiener path.
#   The trajectory switches only where two branches intersect, so Y_t is continuous.
# ==============================================================================


"""Synthetic 3-regime series with CONTINUOUS transitions.

The observed series follows one active linear branch g_n(t)=alpha_n t+beta_n W_t.
Regime switches are allowed ONLY at instants where the active branch crosses
another branch (g_i = g_j). Switching at a crossing makes Y_t continuous:
no artificial jumps. A single common Wiener path W_t drives all branches.
"""


def simulate(alpha, beta, T=30.0, dt=0.01, seed=0,
             p_switch=0.55, min_dwell=2.0):
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    t = np.arange(n + 1) * dt
    dW = rng.normal(0.0, np.sqrt(dt), n)
    W = np.concatenate([[0.0], np.cumsum(dW)])
    N = len(alpha)
    G = np.array([alpha[k] * t + beta[k] * W for k in range(N)])  # N x (n+1)

    active = np.zeros(n + 1, dtype=int)
    cur = 0
    last_switch_t = 0.0
    for i in range(1, n + 1):
        active[i] = cur
        # detect crossing of active branch with any other between i-1 and i
        for j in range(N):
            if j == cur:
                continue
            d_prev = G[cur, i - 1] - G[j, i - 1]
            d_now = G[cur, i] - G[j, i]
            if d_prev == 0 or d_now == 0 or (d_prev < 0) != (d_now < 0):
                if (t[i] - last_switch_t) > min_dwell and rng.random() < p_switch:
                    cur = j
                    last_switch_t = t[i]
                    active[i] = cur
                    break
    Y = G[active, np.arange(n + 1)]
    return dict(t=t, W=W, Y=Y, G=G, active=active,
                alpha=np.asarray(alpha), beta=np.asarray(beta), dt=dt, T=T)


if __name__ == "__main__":
    alpha = [0.30, -0.50, 0.20]
    beta = [0.80, -1.20, 0.40]
    d = simulate(alpha, beta, seed=3)
    jumps = np.abs(np.diff(d["Y"]))
    print("max |dY| step :", jumps.max(), " typical:", np.median(jumps))
    print("regime fractions:",
          [round(np.mean(d["active"] == k), 3) for k in range(3)])
    # count switches
    sw = np.sum(np.diff(d["active"]) != 0)
    print("n switches:", sw)


# ==============================================================================
# PART 2.  CORE LIBRARY A  (formerly a3lib.py)
#   Monomials, Gaussian moments, local pointwise theta(t) = (a, |b|),
#   product-model fit (clustering of the theta-cloud), silhouette, cubic roots.
# ==============================================================================


"""
Core library for the regime-identification model.

Two independent analyses are provided for any (log-)series Y_t:

  (1) product model:   f = prod_n (X - alpha_n t - beta_n w),
      regimes = clusters of the local cloud theta(t)=(a,b).

  (2) A_3 catastrophe: depressed cubic
          f(X,w,t;nu) = X^3 + p(w,t;nu) X + q(w,t;nu),
      with p,q full bivariate polynomials of total degree 3 in (w,t).
      The coefficients {p_jk, q_jk} ARE the unknown vector nu and are found
      by minimising the global moment functional J(nu) (linear least squares
      in nu), NOT plugged in from a closed formula.

Pipeline (always the same):
      step 1 :  estimate theta(t) locally from the series
      step 2 :  minimise J(nu)  ->  nu_hat   (product: cluster; A3: lstsq)
      step 3 :  (A3) solve f(X,w,t;nu_hat)=0 for the roots
"""

# ---------------------------------------------------------------- monomials
def monomials(deg):
    """List of (j,k) with j+k<=deg, j=power of w, k=power of t."""
    out = []
    for s in range(deg + 1):
        for j in range(s + 1):
            out.append((j, s - j))
    return out

P_MON = monomials(3)   # 10 monomials for p(w,t)
Q_MON = monomials(3)   # 10 monomials for q(w,t)
NP_, NQ_ = len(P_MON), len(Q_MON)
NU_DIM = NP_ + NQ_      # = 20

# ------------------------------------------------------------ gauss moments
def gaussmom(m, t):
    """E[w^m], w ~ N(0,t)."""
    if m % 2 == 1:
        return 0.0
    k = m // 2
    return np.prod(np.arange(1, m, 2)) * (t ** k) if m > 0 else 1.0


def _E_wj_gr(j, r, a, b, t):
    """E[ w^j (a t + b w)^r ], w ~ N(0,t)."""
    s = 0.0
    at = a * t
    for l in range(r + 1):
        s += comb(r, l) * (at ** (r - l)) * (b ** l) * gaussmom(j + l, t)
    return s


# ------------------------------------------------------ step 1: local theta
def reconstruct_W(Y, b_local):
    """Single common driving Wiener path from standardised increments:
       dW_j = dY_j / b_local_j ,  W = cumsum.  (Used when W is not observed.)"""
    dY = np.diff(Y)
    dW = dY / np.maximum(b_local, 1e-12)
    return np.concatenate([[0.0], np.cumsum(dW)])


def local_theta(Y, dt, h, hb=None, W=None):
    """Local estimate theta(t)=(a,b).

      a(t) : drift, central difference over the long window 2h
      b(t) : volatility over the short window 2*hb (default hb=h)
             - if W is given: SIGNED slope of dY on dW (recovers sign of beta)
             - else         : sqrt of realised quadratic variation (|beta|>=0)
    """
    n = len(Y)
    dY = np.diff(Y)
    if hb is None:
        hb = h
    Tw = 2 * h * dt
    idx, a, b = [], [], []
    for i in range(h, n - h):
        a.append((Y[i + h] - Y[i - h]) / Tw)
        s0, s1 = i - hb, i + hb
        if W is not None:
            dWseg = np.diff(W)[s0:s1]
            dYseg = dY[s0:s1]
            slope = np.sum(dYseg * dWseg) / max(np.sum(dWseg ** 2), 1e-12)
            b.append(slope)                      # signed
        else:
            seg = dY[s0:s1]
            b.append(np.sqrt(np.sum(seg ** 2) / (2 * hb * dt)))
        idx.append(i)
    return np.array(idx), np.array(a), np.array(b)


def vol_qv(Y, dt, hb):
    """Unsigned local volatility (realised quadratic variation)."""
    n = len(Y)
    dY = np.diff(Y)
    idx, b = [], []
    for i in range(hb, n - hb):
        seg = dY[i - hb:i + hb]
        b.append(np.sqrt(np.sum(seg ** 2) / (2 * hb * dt)))
        idx.append(i)
    return np.array(idx), np.array(b)


# ------------------------------------------- step 2 (A3): linear LSQ for nu
def _design_a3(t_arr, a_arr, b_arr, Kmax):
    rows_A, rows_c, owner = [], [], []
    for ip, (t, a, b) in enumerate(zip(t_arr, a_arr, b_arr)):
        for k in range(Kmax + 1):
            c = _E_wj_gr(0, 3 + k, a, b, t)
            row = np.zeros(NU_DIM)
            for col, (j, kp) in enumerate(P_MON):
                row[col] = (t ** kp) * _E_wj_gr(j, k + 1, a, b, t)
            for col, (j, kp) in enumerate(Q_MON):
                row[NP_ + col] = (t ** kp) * _E_wj_gr(j, k, a, b, t)
            rows_A.append(row)
            rows_c.append(c)
            owner.append(ip)
    return np.array(rows_A), np.array(rows_c), np.array(owner)


def fit_a3(t_arr, a_arr, b_arr, Kmax=2, robust=True, trim=0.15, iters=4):
    """Minimise J(nu) for the depressed cubic (linear in nu).

    Robust trimmed least squares: down-weights cloud points whose moment
    residual is large (e.g. extreme noisy-drift outliers), so the A3 fit
    stays an INDEPENDENT analysis driven only by the local cloud theta(t).
    Inputs assumed already O(1)-scaled (see Scaler).
    Returns nu_hat (20), J, condition number.
    """
    A, c, owner = _design_a3(t_arr, a_arr, b_arr, Kmax)
    col_scale = np.maximum(np.abs(A).max(axis=0), 1e-12)
    As = A / col_scale
    # per-row weight: damp rows with huge base term (heavy-tailed g^{3+k})
    rw = 1.0 / (1.0 + (np.abs(c) / (np.median(np.abs(c)) + 1e-12)))
    keep = np.ones(len(c), dtype=bool)
    nu = np.zeros(NU_DIM)
    for _ in range(iters if robust else 1):
        m = keep
        W = np.sqrt(rw[m])[:, None]
        nu_s, *_ = np.linalg.lstsq(As[m] * W, (-c[m]) * W[:, 0], rcond=None)
        nu = nu_s / col_scale
        if not robust:
            break
        res = np.abs(A @ nu + c)
        # aggregate residual per cloud point, trim worst `trim` fraction
        pts = np.unique(owner)
        agg = np.array([res[owner == p].sum() for p in pts])
        thr = np.quantile(agg, 1 - trim)
        bad_pts = set(pts[agg > thr])
        keep = np.array([o not in bad_pts for o in owner])
    J = float(np.mean((A[keep] @ nu + c[keep]) ** 2))
    cond = np.linalg.cond(As[keep])
    return nu, J, cond


def eval_pq(nu, w, t):
    """Evaluate p(w,t), q(w,t) for given nu."""
    p = sum(nu[i] * (w ** j) * (t ** k) for i, (j, k) in enumerate(P_MON))
    q = sum(nu[NP_ + i] * (w ** j) * (t ** k) for i, (j, k) in enumerate(Q_MON))
    return p, q


def cubic_roots(nu, w, t):
    """Real roots of X^3 + p X + q = 0 at (w,t). Returns sorted real roots."""
    p, q = eval_pq(nu, w, t)
    r = np.roots([1.0, 0.0, p, q])
    real = r[np.abs(r.imag) < 1e-7 * (1 + np.abs(r.real))].real
    return np.sort(real)


def discriminant(nu, w, t):
    p, q = eval_pq(nu, w, t)
    return -4.0 * p ** 3 - 27.0 * q ** 2   # >0 : three distinct real roots


# ------------------------------------------ step 2 (product): cluster cloud
def fit_product(a_arr, b_arr, N, b_weight=4.0, seed=0):
    """Cluster the theta-cloud into N regimes along the robust coordinate b.

    Returns centers (alpha_n, beta_n) [beta = signed via median of b-cluster],
    and the per-point regime label."""
    # cluster mainly along b (robust), lightly along a
    feats = np.column_stack([a_arr, b_weight * b_arr])
    km = KMeans(n_clusters=N, n_init=10, random_state=seed).fit(feats)
    lab = km.labels_
    # order clusters by volatility magnitude (calm -> turbulent)
    order = np.argsort([np.abs(np.median(b_arr[lab == n])) for n in range(N)])
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[x] for x in lab])
    alpha = np.array([np.median(a_arr[lab == n]) for n in range(N)])
    beta = np.array([np.median(b_arr[lab == n]) for n in range(N)])
    return alpha, beta, lab


def silhouette_b(b_arr, Ns=(2, 3, 4), seed=0):
    out = {}
    X = b_arr.reshape(-1, 1)
    for N in Ns:
        lab = KMeans(n_clusters=N, n_init=10, random_state=seed).fit_predict(X)
        out[N] = silhouette_score(X, lab)
    return out


# ------------------------------------------------------------------ scaling
class Scaler:
    """Maps natural (t,Y,w) <-> scaled (tau,x,omega) so the A3 fit is O(1).

        tau = t / T,   x = (Y-Y0)/sY,   omega = w / sqrt(T),  omega~N(0,tau)
        a_tilde = a*T/sY,  b_tilde = b*sqrt(T)/sY
    """
    def __init__(self, T, Y0, sY):
        self.T, self.Y0, self.sY = T, Y0, sY
        self.sw = np.sqrt(T)

    def theta(self, t, a, b):
        return t / self.T, a * self.T / self.sY, b * self.sw / self.sY

    def x_to_Y(self, x):
        return self.Y0 + self.sY * x

    def w_to_omega(self, w):
        return w / self.sw

    def t_to_tau(self, t):
        return t / self.T


# ============================================================================
#  A3 fit, robust formulation: match the cubic identity g^3 + p g + q = 0
#  as a polynomial identity in w  (no high moments of the noisy drift).
#
#  For the active branch at time t,  g(w) = A + B w,  A = a*t,  B = b.
#  Requiring  g^3 + p(w,t) g + q(w,t) = 0  for all w gives, with
#  p(w,t)=sum_j c_j(t) w^j,  q(w,t)=sum_j d_j(t) w^j  (j=0..3):
#       w^0:  A^3        + c0 A           + d0 = 0
#       w^1:  3 A^2 B    + c1 A + c0 B     + d1 = 0
#       w^2:  3 A B^2    + c2 A + c1 B     + d2 = 0
#       w^3:  B^3        + c3 A + c2 B     + d3 = 0
#       w^4:              c3 B             = 0
#  with c_j(t)=sum_k p_{jk} t^k,  d_j(t)=sum_k q_{jk} t^k.  Each row is LINEAR
#  in nu, lowest possible order in the noisy cloud values.  We winsorise the
#  drift, column-scale, and add a small ridge so ill-determined (drift)
#  directions are shrunk while the well-determined volatility/curvature
#  structure (the cusp) is recovered.
# ============================================================================
def _cj_index(j):
    """columns (into nu) and t-powers contributing to c_j(t) (p-part)."""
    out = []
    for col, (jj, k) in enumerate(P_MON):
        if jj == j:
            out.append((col, k))
    return out


def _dj_index(j):
    out = []
    for col, (jj, k) in enumerate(Q_MON):
        if jj == j:
            out.append((NP_ + col, k))
    return out


_C_IDX = [_cj_index(j) for j in range(4)]
_D_IDX = [_dj_index(j) for j in range(4)]


def _design_a3_poly(t_arr, A_arr, B_arr):
    """Build M nu = r from the 5 power-of-w equations per cloud point."""
    rows, rhs = [], []
    for t, A, B in zip(t_arr, A_arr, B_arr):
        tp = [t ** k for k in range(4)]
        # E0 : c0*A + d0 = -A^3
        r0 = np.zeros(NU_DIM)
        for col, k in _C_IDX[0]:
            r0[col] += A * tp[k]
        for col, k in _D_IDX[0]:
            r0[col] += tp[k]
        rows.append(r0); rhs.append(-A ** 3)
        # E1 : c1*A + c0*B + d1 = -3 A^2 B
        r1 = np.zeros(NU_DIM)
        for col, k in _C_IDX[1]:
            r1[col] += A * tp[k]
        for col, k in _C_IDX[0]:
            r1[col] += B * tp[k]
        for col, k in _D_IDX[1]:
            r1[col] += tp[k]
        rows.append(r1); rhs.append(-3 * A ** 2 * B)
        # E2 : c2*A + c1*B + d2 = -3 A B^2
        r2 = np.zeros(NU_DIM)
        for col, k in _C_IDX[2]:
            r2[col] += A * tp[k]
        for col, k in _C_IDX[1]:
            r2[col] += B * tp[k]
        for col, k in _D_IDX[2]:
            r2[col] += tp[k]
        rows.append(r2); rhs.append(-3 * A * B ** 2)
        # E3 : c3*A + c2*B + d3 = -B^3
        r3 = np.zeros(NU_DIM)
        for col, k in _C_IDX[3]:
            r3[col] += A * tp[k]
        for col, k in _C_IDX[2]:
            r3[col] += B * tp[k]
        for col, k in _D_IDX[3]:
            r3[col] += tp[k]
        rows.append(r3); rhs.append(-B ** 3)
        # E4 : c3*B = 0
        r4 = np.zeros(NU_DIM)
        for col, k in _C_IDX[3]:
            r4[col] += B * tp[k]
        rows.append(r4); rhs.append(0.0)
    return np.array(rows), np.array(rhs)


def _winsor(x, k=3.0):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
    return np.clip(x, lo, hi)


#   per-equation weights (E0..E4): downweight the drift-heavy rows E0 (~A^3)
#   and E1 (~A^2 B) so the fit is anchored on the well-determined volatility /
#   curvature structure (E2,E3,E4); the ridge shrinks the ill-determined drift
#   directions toward zero.  This realises the honest limitation Var(a_hat)~b^2.
EQ_WEIGHTS = np.array([0.15, 0.4, 1.0, 1.0, 1.0])


def fit_a3_poly(t_arr, a_arr, b_arr, ridge=1e-2, robust=True, trim=0.15,
                iters=4, winsor_k=2.5, eq_weights=EQ_WEIGHTS):
    """Estimate nu (20) by weighted least squares on the cubic-identity rows.

    t_arr,a_arr,b_arr : scaled cloud (tau, a_tilde, b_tilde), already O(1).
    Returns nu, J (mean sq identity residual), condition number.
    """
    a_w = _winsor(a_arr, winsor_k)
    A_arr = a_w * t_arr
    B_arr = b_arr
    M, r = _design_a3_poly(t_arr, A_arr, B_arr)
    nrow_per = 5
    ew = np.tile(eq_weights, len(t_arr))
    cs = np.maximum(np.abs(M).max(axis=0), 1e-12)
    Ms = M / cs
    npar = NU_DIM
    keep = np.ones(M.shape[0], dtype=bool)
    nu = np.zeros(npar)
    lam = np.sqrt(ridge)
    for _ in range(iters if robust else 1):
        wv = ew[keep]
        Mk = Ms[keep] * wv[:, None]
        rk = r[keep] * wv
        # ridge-augmented least squares (stable; no normal-equation squaring)
        Maug = np.vstack([Mk, lam * np.eye(npar)])
        raug = np.concatenate([rk, np.zeros(npar)])
        nu_s, *_ = np.linalg.lstsq(Maug, raug, rcond=None)
        nu = nu_s / cs
        if not robust:
            break
        res = np.abs(M @ nu - r) * ew
        npts = len(t_arr)
        agg = res.reshape(npts, nrow_per).sum(axis=1)
        thr = np.quantile(agg, 1 - trim)
        good = agg <= thr
        keep = np.repeat(good, nrow_per)
    resid = (M[keep] @ nu - r[keep])
    J = float(np.mean(resid ** 2))
    cond = float(np.linalg.cond(Ms[keep] * ew[keep][:, None]))
    return nu, J, cond


# ============================================================================
#  A3 fit -- value+slope form (drift-free, uses only clean observables).
#  Per observed point we know the scaled value X (=observed centred series),
#  the driver omega and time tau, and the clean local volatility b=dX/domega.
#  The branch X(omega) of f=X^3+pX+q=0 must satisfy
#     (1) value :  X^3 + p X + q = 0
#     (2) slope :  (3X^2+p) b + p_w X + q_w = 0     (since dX/dw=-f_w/f_X=b)
#  Both are LINEAR in nu and involve no drift estimate, so the noisy local
#  drift never enters.  The t-dependence (drift structure) is recovered
#  globally from how (X,b) vary with tau across the cloud.
# ============================================================================
def _pw_index(j):
    """columns/(power) contributing to p_omega = d p/d omega, per omega-power."""
    out = []
    for col, (jj, k) in enumerate(P_MON):
        if jj >= 1:
            out.append((col, jj, jj - 1, k))  # (col, coeff j, new omega-pow, tau-pow)
    return out


def _design_a3_vs(tau, X, b, omega):
    rows, rhs = [], []
    for t, x, bb, w in zip(tau, X, b, omega):
        # ---- Eq1 : p*X + q = -X^3
        r1 = np.zeros(NU_DIM)
        for col, (j, k) in enumerate(P_MON):
            r1[col] += x * (w ** j) * (t ** k)
        for col, (j, k) in enumerate(Q_MON):
            r1[NP_ + col] += (w ** j) * (t ** k)
        rows.append(r1); rhs.append(-x ** 3)
        # ---- Eq2 : p*b + p_w*X + q_w = -3 X^2 b
        r2 = np.zeros(NU_DIM)
        for col, (j, k) in enumerate(P_MON):
            r2[col] += bb * (w ** j) * (t ** k)                     # p*b
            if j >= 1:
                r2[col] += x * j * (w ** (j - 1)) * (t ** k)        # p_w * X
        for col, (j, k) in enumerate(Q_MON):
            if j >= 1:
                r2[NP_ + col] += j * (w ** (j - 1)) * (t ** k)      # q_w
        rows.append(r2); rhs.append(-3 * x ** 2 * bb)
    return np.array(rows), np.array(rhs)


def fit_a3_vs(tau, X, b, omega, ridge=1e-3, robust=True, trim=0.15, iters=5,
              slope_weight=1.0):
    """Drift-free value+slope A3 estimator. Returns nu, J, cond."""
    M, r = _design_a3_vs(tau, X, b, omega)
    # per-equation weight (rows alternate value, slope)
    ew = np.tile([1.0, slope_weight], len(tau))
    cs = np.maximum(np.abs(M).max(axis=0), 1e-12)
    Ms = M / cs
    npar = NU_DIM
    keep = np.ones(M.shape[0], dtype=bool)
    lam = np.sqrt(ridge)
    nu = np.zeros(npar)
    for _ in range(iters if robust else 1):
        wv = ew[keep]
        Mk = Ms[keep] * wv[:, None]
        rk = r[keep] * wv
        Maug = np.vstack([Mk, lam * np.eye(npar)])
        raug = np.concatenate([rk, np.zeros(npar)])
        nu_s, *_ = np.linalg.lstsq(Maug, raug, rcond=None)
        nu = nu_s / cs
        if not robust:
            break
        res = np.abs(M @ nu - r) * ew
        agg = res.reshape(len(tau), 2).sum(axis=1)
        thr = np.quantile(agg, 1 - trim)
        good = agg <= thr
        keep = np.repeat(good, 2)
    J = float(np.mean((M[keep] @ nu - r[keep]) ** 2))
    cond = float(np.linalg.cond(Ms[keep] * ew[keep][:, None]))
    return nu, J, cond


# ==============================================================================
# PART 3.  CORE LIBRARY B  (formerly catlib.py)
#   Moment-condition machinery: E_{r,p}, M_k, rho_k, the normal equations
#   A nu = xi with ridge on the original nu, and the catastrophe roots.
# ==============================================================================


"""
General catastrophe-equilibrium estimator (value + slope, linear in the
control coefficients), for a depressed polynomial of arbitrary degree n:

    A_n  equilibrium:   f(X,w,t) = X^n + sum_{j=0}^{n-2} c_j(w,t) X^j = 0

The X^{n-1} term is removed (depressed form); each control c_j(w,t) is a
full bivariate polynomial of total degree <= 3 in (w,t).

    A_3 (cusp)       -> n=3, controls c_1=p, c_0=q            (2 x 10 = 20 par)
    A_5 (butterfly)  -> n=5, controls c_3,c_2,c_1,c_0         (4 x 10 = 40 par)

Fit uses only clean observables: the observed scaled value X, its driver
omega, time tau, and the clean local volatility b = dX/domega.  Two linear
equations per observation:

    value :  X^n + sum_j c_j X^j = 0
    slope :  f_X * b + f_w = 0,   f_X = n X^{n-1}+sum_j j c_j X^{j-1},
                                  f_w = sum_j c_{j,w} X^j
Both are linear in the coefficients; solved by ridge-regularised least
squares with robust trimming (identical machinery to fit_a3_vs).
"""

MON = monomials(3)          # [(r,s) : r+s<=3], 10 monomials
NMON = len(MON)


def _mon_vec(w, t):
    return np.array([(w ** r) * (t ** s) for (r, s) in MON])


def _mon_dw_vec(w, t):
    return np.array([(r * w ** (r - 1) if r >= 1 else 0.0) * (t ** s)
                     for (r, s) in MON])


def control_powers(degree):
    """Powers j of X carrying a control c_j, high to low (skip X^{n-1})."""
    return list(range(degree - 2, -1, -1))


def _design_cat_vs(X, w, t, b, degree):
    powers = control_powers(degree)
    ncoef = len(powers) * NMON
    rows, rhs = [], []
    for Xi, wi, ti, bi in zip(X, w, t, b):
        m = _mon_vec(wi, ti)
        mdw = _mon_dw_vec(wi, ti)
        # ---- value : sum_j c_j X^j = -X^n
        rv = np.zeros(ncoef)
        for idx, j in enumerate(powers):
            rv[idx * NMON:(idx + 1) * NMON] = (Xi ** j) * m
        rows.append(rv); rhs.append(-(Xi ** degree))
        # ---- slope : sum_j [ b*j*X^{j-1} + c_{j,w}->X^j ] = -n X^{n-1} b
        rs = np.zeros(ncoef)
        for idx, j in enumerate(powers):
            if j >= 1:
                rs[idx * NMON:(idx + 1) * NMON] += bi * j * (Xi ** (j - 1)) * m
            rs[idx * NMON:(idx + 1) * NMON] += (Xi ** j) * mdw
        rows.append(rs); rhs.append(-degree * (Xi ** (degree - 1)) * bi)
    return np.array(rows), np.array(rhs)


def fit_cat_vs(X, w, t, b, degree, ridge=1e-3, robust=True, trim=0.1, iters=6,
               slope_weight=1.0):
    """Return (nu, J, cond). nu is the flat control-coefficient vector."""
    M, r = _design_cat_vs(X, w, t, b, degree)
    n = len(X)
    ew = np.tile([1.0, slope_weight], n)
    cs = np.maximum(np.abs(M).max(axis=0), 1e-12)
    Ms = M / cs
    npar = M.shape[1]
    keep = np.ones(M.shape[0], dtype=bool)
    lam = np.sqrt(ridge)
    nu = np.zeros(npar)
    for _ in range(iters if robust else 1):
        wv = ew[keep]
        Mk = Ms[keep] * wv[:, None]
        rk = r[keep] * wv
        Maug = np.vstack([Mk, lam * np.eye(npar)])
        raug = np.concatenate([rk, np.zeros(npar)])
        nu_s, *_ = np.linalg.lstsq(Maug, raug, rcond=None)
        nu = nu_s / cs
        if not robust:
            break
        res = np.abs(M @ nu - r) * ew
        agg = res.reshape(n, 2).sum(axis=1)
        thr = np.quantile(agg, 1 - trim)
        good = agg <= thr
        keep = np.repeat(good, 2)
    J = float(np.mean((M[keep] @ nu - r[keep]) ** 2))
    cond = float(np.linalg.cond(Ms[keep] * ew[keep][:, None]))
    return nu, J, cond


def _controls_at(nu, w, t, degree):
    """Evaluate control values c_j(w,t) at a point, high power to low."""
    powers = control_powers(degree)
    m = _mon_vec(w, t)
    return [nu[i * NMON:(i + 1) * NMON] @ m for i in range(len(powers))]


def cat_roots(nu, w, t, degree, tol=1e-7):
    """Real roots of the depressed equilibrium polynomial at (w,t)."""
    powers = control_powers(degree)
    cvals = _controls_at(nu, w, t, degree)
    coeffs = np.zeros(degree + 1)
    coeffs[0] = 1.0            # X^n
    # X^{n-1} term absent (depressed); fill controls
    for j, c in zip(powers, cvals):
        coeffs[degree - j] = c
    rts = np.roots(coeffs)
    real = rts[np.abs(rts.imag) < tol].real
    return np.sort(real)


# ============================================================================
#  W-FREE line-root estimator (no realized/reconstructed W_t).
#  Requires the local active-branch LINE  X = A + B*omega  (A=a*tau, B=b)
#  to be a root of the depressed A_n polynomial identically in omega, i.e.
#  every power-of-omega coefficient of f(A+B*omega, omega, tau) vanishes.
#  Uses ONLY theta(t)=(a,b); omega is the abstract Gaussian argument, never a
#  reconstructed trajectory.  Linear in nu -> ridge least squares.
# ============================================================================


def _design_line_root(atil, btil, tau, degree):
    powers = control_powers(degree)          # [degree-2,...,0]
    ncoef = len(powers) * NMON
    maxP = degree + 1
    rows, rhs = [], []
    for ai, bi, ti in zip(atil, btil, tau):
        A = ai * ti          # intercept a*tau
        B = bi               # slope b
        for P in range(maxP + 1):
            row = np.zeros(ncoef)
            for jc, j in enumerate(powers):
                for m, (r, s) in enumerate(MON):
                    l = P - r
                    if 0 <= l <= j:
                        row[jc * NMON + m] += (ti ** s) * comb(j, l) * (A ** (j - l)) * (B ** l)
            rr = -comb(degree, P) * (A ** (degree - P)) * (B ** P) if P <= degree else 0.0
            rows.append(row); rhs.append(rr)
    return np.array(rows), np.array(rhs), maxP + 1


def fit_line_root(atil, btil, tau, degree, ridge=1e-2, robust=True, trim=0.15,
                  iters=6, low_w_weight=0.3):
    """W-free line-root fit. Down-weights the lowest omega-power rows (drift-
    heavy, since they carry A=a*tau) via low_w_weight. Returns nu, J, cond."""
    M, r, nrow = _design_line_root(atil, btil, tau, degree)
    npts = len(tau)
    # per-row weights: low omega-powers (drift-heavy) down-weighted
    w_per = np.array([low_w_weight if P <= 1 else 1.0 for P in range(nrow)])
    ew = np.tile(w_per, npts)
    cs = np.maximum(np.abs(M).max(axis=0), 1e-12)
    Ms = M / cs
    npar = M.shape[1]
    keep = np.ones(M.shape[0], dtype=bool)
    lam = np.sqrt(ridge)
    nu = np.zeros(npar)
    for _ in range(iters if robust else 1):
        wv = ew[keep]
        Mk = Ms[keep] * wv[:, None]; rk = r[keep] * wv
        Maug = np.vstack([Mk, lam * np.eye(npar)])
        raug = np.concatenate([rk, np.zeros(npar)])
        nu_s, *_ = np.linalg.lstsq(Maug, raug, rcond=None)
        nu = nu_s / cs
        if not robust:
            break
        res = np.abs(M @ nu - r) * ew
        agg = res.reshape(npts, nrow).sum(axis=1)
        good = agg <= np.quantile(agg, 1 - trim)
        keep = np.repeat(good, nrow)
    J = float(np.mean((M[keep] @ nu - r[keep]) ** 2))
    cond = float(np.linalg.cond(Ms[keep] * ew[keep][:, None]))
    return nu, J, cond


# ============================================================================
#  FAITHFUL moment functional  J(nu) = int_0^Tmax sum_k L_k(theta(t),nu)^2 dt.
#  L_k = E_omega[ f(g(omega),omega,tau) * g(omega)^k ],  omega ~ N(0,tau),
#  g(omega) = A + B*omega  with A = atil*tau, B = btil (>0, linear ansatz).
#  f = X^n + sum_{j<=n-2} c_j(omega,tau) X^j ,  c_j = sum_{r+s<=3} c_j^{rs} w^r t^s.
#  Each L_k is AFFINE in nu (leading X^n fixed = 1); minimising the discretised
#  J is a ridge least-squares problem.  No W_t trajectory; w is the Gaussian
#  integration variable.
# ============================================================================
def _dfact(m):
    r = 1.0
    while m > 1:
        r *= m; m -= 2
    return r


def _gaussmom(m, tau):
    """E[omega^m], omega~N(0,tau)."""
    if m % 2 == 1:
        return 0.0
    return _dfact(m - 1) * (tau ** (m // 2)) if m > 0 else 1.0


def _E_wr_Ap(r, p, A, B, tau):
    """E[ omega^r (A+B*omega)^p ], omega~N(0,tau)."""
    s = 0.0
    for l in range(p + 1):
        s += comb(p, l) * (A ** (p - l)) * (B ** l) * _gaussmom(r + l, tau)
    return s


def fit_moment(atil, btil, tau, degree, K=None, ridge=1e-2, robust=True,
               trim=0.15, iters=6):
    """Minimise sum_k L_k^2 integrated over t. Returns nu, J, cond."""
    powers = control_powers(degree)
    if K is None:
        K = degree + 1                      # complete independent moment set
    ncoef = len(powers) * NMON
    nrow = K + 1
    rows, rhs = [], []
    for ai, bi, ti in zip(atil, btil, tau):
        A = ai * ti; B = bi
        for k in range(nrow):
            row = np.zeros(ncoef)
            for jc, j in enumerate(powers):
                for m, (r, s) in enumerate(MON):
                    row[jc * NMON + m] = (ti ** s) * _E_wr_Ap(r, j + k, A, B, ti)
            rk = _E_wr_Ap(0, degree + k, A, B, ti)      # from X^degree term
            rows.append(row); rhs.append(-rk)
    M = np.array(rows); r = np.array(rhs)
    npts = len(tau)
    # balance moments: divide each equation by its scale (row norm) so that
    # sum_k L_k^2 is not dominated by the largest-k moments (E[g^{n+k}] grows).
    rownorm = np.maximum(np.sqrt((M ** 2).sum(axis=1) + r ** 2), 1e-12)
    M = M / rownorm[:, None]; r = r / rownorm
    npar = M.shape[1]
    keep = np.ones(M.shape[0], dtype=bool)
    lam = np.sqrt(ridge)
    nu = np.zeros(npar)
    for _ in range(iters if robust else 1):
        # ridge on the ORIGINAL nu (penalise all coefficients equally, so
        # poorly-identified high-order directions are shrunk toward 0 and nu
        # stays bounded); solved by stable SVD least squares (no normal eqs).
        Maug = np.vstack([M[keep], lam * np.eye(npar)])
        raug = np.concatenate([r[keep], np.zeros(npar)])
        nu, *_ = np.linalg.lstsq(Maug, raug, rcond=None)
        if not robust:
            break
        res = np.abs(M @ nu - r)
        agg = res.reshape(npts, nrow).sum(axis=1)
        good = agg <= np.quantile(agg, 1 - trim)
        keep = np.repeat(good, nrow)
    J = float(np.mean((M[keep] @ nu - r[keep]) ** 2))
    cond = float(np.linalg.cond(M[keep]))
    return nu, J, cond


# ==============================================================================
# PART 4.  TRANSPARENT STANDALONE SOLVER FOR  A nu = xi
#   Builds A and xi explicitly and cross-checks against fit_moment.
#   Its local helpers are prefixed sv_ to avoid clashing with Part 3.
# ==============================================================================


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


def sv_control_powers(n):              # powers j of X carrying a control (skip X^{n-1})
    return list(range(n - 2, -1, -1))


# ----------------------------- design rows per point ------------------------
def point_rows(a, b, tau, n, K):
    """Return list of (M_k row, rho_k) for k=0..K at one cloud point."""
    A, B = a * tau, b
    powers = sv_control_powers(n)
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
def sv_cat_roots(nu, w, t, degree, tol=1e-7):
    powers = sv_control_powers(degree)
    cvals = [nu[i * NMON:(i + 1) * NMON] @ np.array([w ** r * t ** s for (r, s) in MON])
             for i in range(len(powers))]
    coeffs = np.zeros(degree + 1); coeffs[0] = 1.0
    for j, c in zip(powers, cvals):
        coeffs[degree - j] = c
    rts = np.roots(coeffs)
    return np.sort(rts[np.abs(rts.imag) < tol].real)


# =========================== self-tests =====================================

def run_selftest():
    """Exact-recovery test + agreement with fit_moment. Needs sympy."""
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
    for kind in ["synth", "btc"]:
        D = prep_gwt(kind)
        nu1, J1, c1 = fit_moment(D['atil'], D['btil'], D['tau'], 3, ridge=1e-2)
        nu2, J2, A2, xi2 = solve_nu(D['atil'], D['btil'], D['tau'], 3, ridge=1e-2)
        dmax = max(np.abs(np.sort(sv_cat_roots(nu1, ww, tt, 3)) -
                          np.sort(sv_cat_roots(nu2, ww, tt, 3))).max()
                   for ww, tt in [(0.3, 0.3), (-0.5, 0.6), (0.8, 0.85)]
                   if len(sv_cat_roots(nu1, ww, tt, 3)) == len(sv_cat_roots(nu2, ww, tt, 3)) == 3)
        print(f"{kind}: roots agree with fit_moment to {dmax:.1e}  "
              f"(explicit-A vs stacked-lstsq)")


# ==============================================================================
# PART 5.  FIGURES - EXPERIMENT 1 (regime identification)
#   Synthetic: series, true branches, theta-cloud + b-histogram, A3 roots.
#   BTC/USD: series, theta-cloud, volatility levels, A3 roots.
# ==============================================================================


"""
Experiment 1 (regime finding) figures, English labels, 350 dpi.

Clustering method (stated explicitly): regimes differ by volatility, and the
drift a is weakly identified (Var(a_hat) ~ b^2), so we cluster on the reliably
identified coordinate |b(t)| (1-D).  Standardise |b|, run K-means with N=3,
order clusters by increasing centre (calm/normal/turbulent), assign each time
its cluster -> segmentation.  theta(t)=(a,|b|) is the RAW per-point estimate.
"""

OUT = "figs/"
ALPHA_EXP1 = [0.10, -0.15, 0.05]; BETA_EXP1 = [0.80, -1.20, 0.40]; SEED_EXP1 = 314


def cluster_b_exp1(b, N=3):
    """K-means on standardised |b|; return labels ordered calm->turbulent,
    ordered centres (in |b| units), silhouette."""
    v = np.abs(b).reshape(-1, 1)
    vs = (v - v.mean()) / (v.std() + 1e-12)
    km = KMeans(N, n_init=10, random_state=0).fit(vs)
    order = np.argsort(km.cluster_centers_.ravel())
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[x] for x in km.labels_])
    centres = np.array([np.abs(b)[lab == k].mean() for k in range(N)])
    sil = silhouette_score(vs, km.labels_)
    return lab, centres, sil


def prep_exp1(kind):
    if kind == "synth":
        d = simulate(ALPHA_EXP1, BETA_EXP1, T=30, dt=0.01, seed=SEED_EXP1, p_switch=0.6, min_dwell=2.5)
        Y, dt, T, W = d['Y'], d['dt'], d['T'], d['W']
        idx, a, b = local_theta(Y, dt, h=120, hb=30, W=None)
        xax, xlabel, title = d['t'][idx], "time $t$", "Synthetic"
        extra = dict(t=d['t'], W=W, active=d['active'])
    else:
        c = pd.read_csv('btc_week_1min.csv')['close'].values[::5]
        Y = np.log(c); T = float(len(Y)); dt = 1.0
        idx, a, b = local_theta(Y, dt, h=48, hb=12, W=None)
        xax, xlabel, title = idx.astype(float), "5-min bars", "BTC/USD"
        extra = dict()
    sb = np.median(np.abs(b)); sX = np.sqrt(T) * sb; Y0 = np.median(Y)
    tau = idx * dt / T
    atil = a * T / sX; btil = np.abs(b) * np.sqrt(T) / sX
    lab, centres, sil = cluster_b_exp1(b, 3)
    return dict(idx=idx, a=a, b=b, Y=Y, Y0=Y0, T=T, dt=dt, sX=sX, tau=tau,
                atil=atil, btil=btil, lab=lab, centres=centres, sil=sil,
                xax=xax, xlabel=xlabel, title=title, n=len(Y), **extra)


def fill_exp1(n, idx, lab):
    f = np.full(n, -1); f[idx] = lab; last = lab[0]
    for i in range(n):
        if f[i] < 0:
            f[i] = last
        else:
            last = f[i]
    return f


# ---------------------------------------------------------------- series
def fig_series(D, fname, ann=None):
    labf = fill_exp1(D['n'], D['idx'], D['lab'])
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    Y = D['Y']
    for k in range(3):
        ax.fill_between(np.arange(D['n']), Y.min(), Y.max(), where=(labf == k),
                        color=REG[k], alpha=0.13, step='mid')
    ax.plot(np.arange(D['n']), Y, color='#222', lw=0.7)
    ax.set_xlim(0, D['n'] - 1); ax.set_ylim(Y.min(), Y.max())
    ax.set_ylabel("$Y_t$" if D['title'] == "Synthetic" else "BTC log-price")
    ax.set_xlabel("time index" if D['title'] == "Synthetic" else "5-min bars (Feb 21--28, 2025)")
    ax.set_title(f"{D['title']}: series and identified regimes (clustering of $b(t)$)")
    lbl = ([f"regime {k+1}" for k in range(3)] if ann is None else
           [f"regime {k+1}: {ann(D['centres'][k]):.0f}\\%" for k in range(3)])
    ax.legend(handles=[Patch(color=REG[k], alpha=0.4, label=lbl[k]) for k in range(3)],
              fontsize=12, ncol=3, loc='upper right')
    plt.tight_layout(); plt.savefig(OUT + fname, dpi=350); plt.close()


# ---------------------------------------------------------------- true branches (synth)
def fig_true_branches(D, fname):
    t, W = D['t'], D['W']
    G = np.array([ALPHA_EXP1[k] * t + BETA_EXP1[k] * W for k in range(3)])   # true branches g_n(t)
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    names = ["calm ($\\beta=0.4$)", "mid ($\\beta=0.8$)", "turbulent ($\\beta=1.2$)"]
    ord_by_absbeta = [2, 0, 1]        # map plotted colour calm/mid/turb -> branch idx
    for col, bi in enumerate(ord_by_absbeta):
        ax.plot(t, G[bi], color=REG[col], lw=1.3, alpha=0.9, label=names[col])
    ax.plot(t, D['Y'], color='#111', lw=0.8, label="observed $Y_t$ (active branch)")
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("time $t$"); ax.set_ylabel("$X$")
    ax.set_title("Synthetic: true regime branches $g_n(t)=\\alpha_n t+\\beta_n W_t$ "
                 "(all three shown; $Y_t$ rides the active one)")
    ax.legend(fontsize=12, ncol=2, loc='upper left')
    plt.tight_layout(); plt.savefig(OUT + fname, dpi=350); plt.close()


# ---------------------------------------------------------------- cloud + |b| histogram
def fig_cloud(D, fname, ann=None):
    a, b, lab, ctr = D['a'], np.abs(D['b']), D['lab'], D['centres']
    scale = (1e4 if D['title'] != "Synthetic" else 1.0)
    ascaled = a * scale
    yb = b * (100 if ann else 1.0)
    fig = plt.figure(figsize=(10.0, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0]); axh = fig.add_subplot(gs[1], sharey=ax)
    for k in range(3):
        m = lab == k
        ax.scatter(ascaled[m], yb[m], s=6, color=REG[k], alpha=0.35, edgecolors='none',
                   label=(f"regime {k+1}" if ann is None else f"regime {k+1}: {ann(ctr[k]):.0f}\\%"))
        axh.hist(yb[m], bins=40, orientation='horizontal', color=REG[k], alpha=0.6)
    for k in range(3):
        ax.scatter(np.median(ascaled[lab == k]), ctr[k] * (100 if ann else 1.0),
                   marker='X', s=130, color=REG[k], edgecolors='k', lw=1.1, zorder=5)
    if D['title'] == "Synthetic":
        tb = np.abs(np.array(BETA_EXP1))[np.argsort(np.abs(BETA_EXP1))]
        ta = np.array(ALPHA_EXP1)[np.argsort(np.abs(BETA_EXP1))]
        for k in range(3):
            ax.scatter(ta[k], tb[k], marker='o', s=80, facecolors='none',
                       edgecolors=TRUE, lw=2, zorder=6,
                       label="true centres" if k == 0 else None)
    ax.set_xlabel("drift $a$" + (r" ($\times10^{-4}$/bar)" if D['title'] != "Synthetic" else ""))
    ax.set_ylabel("volatility $b$" + (r" (\%, ann.)" if ann else " (from quad. variation)"))
    ax.set_title(f"{D['title']}: cloud $\\theta(t)=(a,b)$ clustered on $b$  "
                 f"(silhouette {D['sil']:.2f})")
    ax.legend(fontsize=12, loc='lower left' if ann else 'upper left')
    axh.tick_params(labelleft=False); axh.set_xlabel("count")
    axh.set_title("$b$ hist.", fontsize=12)
    plt.savefig(OUT + fname, dpi=350); plt.close()


# ---------------------------------------------------------------- A3 roots on t-slices
def fig_roots_A3(D, fname, taus_show=(0.22, 0.5, 0.82)):
    nu, J, cond = fit_moment(D['atil'], D['btil'], D['tau'], 3, ridge=1e-2)
    true_lines = None
    if D['title'] == "Synthetic":
        ats = np.array(ALPHA_EXP1) * D['T'] / D['sX']; bts = np.array(BETA_EXP1) * np.sqrt(D['T']) / D['sX']
        true_lines = (ats, bts)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for a_, t0 in zip(ax, taus_show):
        wmax = 2.0 * np.sqrt(t0); om = np.linspace(-wmax, wmax, 260)
        for o in om:
            r = cat_roots(nu, o, t0, 3)
            a_.plot([o] * len(r), r, '.', color=ACCENT, ms=2.4)
        if true_lines is not None:
            for k in range(3):
                a_.plot(om, true_lines[0][k] * t0 + true_lines[1][k] * om,
                        color=TRUE, lw=1.6, alpha=0.5,
                        label="true branches" if k == 0 else None)
        a_.set_title(f"$\\tau={t0}$"); a_.set_xlabel("$w$")
    ax[0].set_ylabel("roots $X=g_i(w,t)$")
    if true_lines is not None:
        ax[0].legend(fontsize=12, loc='best')
    fig.suptitle(f"{D['title']}: branches $X=g_i(w,t)$ of the $A_3$ cusp on time slices "
                 f"(fit by minimising $\\int\\sum_k L_k^2\\,dt$)", y=1.02)
    plt.tight_layout(); plt.savefig(OUT + fname, dpi=350); plt.close()



def run_exp1():
    """Build all Experiment-1 figures."""
    BPY = 288 * 365
    ann = lambda bt: bt * np.sqrt(BPY) * 100
    S = prep_exp1("synth"); B = prep_exp1("btc")
    print(f"synth silhouette={S['sil']:.3f} centres|b|={np.round(S['centres'],3)}")
    print(f"btc   silhouette={B['sil']:.3f} centres ann%={np.round(ann(B['centres']),1)}")
    # Experiment 1 figures
    fig_series(S, "fig_synth_series.png")
    fig_true_branches(S, "fig_synth_true.png")
    fig_cloud(S, "fig_synth_cloud.png")
    fig_roots_A3(S, "fig_synth_roots_A3.png")
    fig_series(B, "fig_btc_series.png", ann=ann)
    fig_cloud(B, "fig_btc_cloud.png", ann=ann)
    fig_roots_A3(B, "fig_btc_roots_A3.png")
    # BTC volatility levels over time
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.plot(B['idx'], ann(np.abs(B['b'])), color='#888', lw=0.6, alpha=0.6)
    for k in range(3):
        m = B['lab'] == k
        ax.scatter(B['idx'][m], ann(np.abs(B['b'])[m]), s=5, color=REG[k], alpha=0.6)
        ax.axhline(ann(B['centres'][k]), color=REG[k], lw=1.6, ls='--')
    ax.set_xlabel("5-min bars"); ax.set_ylabel("annualized volatility, \\%")
    ax.set_title("BTC/USD: local volatility $b(t)$ and regime levels")
    plt.tight_layout(); plt.savefig(OUT + "fig_btc_levels.png", dpi=350); plt.close()
    print("Experiment 1 figures saved")


# ==============================================================================
# PART 6.  FIGURES - PRODUCT MODEL ON SYNTHETIC DATA
#   Unsigned b from quadratic variation; true centres at (alpha_n, |beta_n|).
# ==============================================================================


"""
Product-model illustration for the SYNTHETIC data (English labels, 350 dpi).

UNSIGNED version: volatility b is estimated from realised quadratic variation,
so b = |b| >= 0 everywhere (identical formula to every other figure and to the
real-data case). The sign of beta is NOT identifiable from a single path and is
irrelevant since W_t is symmetric (P(W_t>0)=1/2); a branch with beta and one
with -beta give the same observable law. True centres are therefore placed at
(alpha_n, |beta_n|).

Produces (only the cloud is regenerated here):
  fig_synth_prod_cloud.png  -- cloud (a,b) with estimated (x) and true (o) centres
  fig_synth_prod_series.png -- series coloured by regime (unchanged content)
"""

OUT = "figs/"
ALPHA_PROD = [0.10, -0.15, 0.05]; BETA_PROD = [0.80, -1.20, 0.40]; SEED_PROD = 314


def fill_prod(n, idx, lab):
    f = np.full(n, -1); f[idx] = lab; last = lab[0]
    for i in range(n):
        if f[i] < 0:
            f[i] = last
        else:
            last = f[i]
    return f


def cluster_b_prod(a, b, N=3, seed=0):
    """Cluster the theta-cloud on the robust coordinate |b| (1-D KMeans)."""
    ab = np.abs(b)
    z = (ab - ab.mean()) / (ab.std() + 1e-12)
    km = KMeans(n_clusters=N, n_init=10, random_state=seed).fit(z.reshape(-1, 1))
    lab = km.labels_
    order = np.argsort([ab[lab == k].mean() for k in range(N)])   # calm -> turbulent
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[x] for x in lab])
    ac = np.array([np.median(a[lab == k]) for k in range(N)])
    bc = np.array([ab[lab == k].mean() for k in range(N)])
    return ac, bc, lab



def run_prod():
    """Build the product-model figures (unsigned b)."""
    d = simulate(ALPHA_PROD, BETA_PROD, T=30, dt=0.01, seed=SEED_PROD, p_switch=0.6, min_dwell=2.5)
    t, W, Y, dt, T = d['t'], d['W'], d['Y'], d['dt'], d['T']
    idx, a, b = local_theta(Y, dt, h=120, hb=30, W=None)          # UNSIGNED |b|
    alh, beh, lab = cluster_b_prod(a, b, N=3, seed=0)

    # true centres ordered by |beta| (ascending), sign of beta dropped
    order = np.argsort(np.abs(BETA_PROD))
    ta = np.array(ALPHA_PROD)[order]; tb = np.abs(np.array(BETA_PROD))[order]

    bb = np.abs(b)

    # ---------------- cloud ----------------
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for k in range(3):
        m = lab == k
        ax.scatter(a[m], bb[m], s=6, color=REG[k], alpha=0.35, edgecolors='none',
                   label=f"regime {k+1}")
    for k in range(3):
        ax.scatter(alh[k], beh[k], marker='X', s=150, color=REG[k],
                   edgecolors='k', lw=1.3, zorder=5)
    for k in range(3):
        ax.scatter(ta[k], tb[k], marker='o', s=110, facecolors='none',
                   edgecolors=TRUE, lw=2.2, zorder=6,
                   label="true centres" if k == 0 else None)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("drift $a$")
    ax.set_ylabel("volatility $b$ (from quad. variation)")
    ax.set_title("Synthetic, product model: cloud $\\theta(t)=(a,b)$ and regime centres")
    ax.legend(fontsize=12, loc='center right')
    plt.tight_layout(); plt.savefig(OUT + "fig_synth_prod_cloud.png", dpi=350); plt.close()

    # ---------------- series ----------------
    labf = fill_prod(len(t), idx, lab)
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    for k in range(3):
        ax.fill_between(t, Y.min(), Y.max(), where=(labf == k), color=REG[k],
                        alpha=0.15, step='mid')
    ax.plot(t, Y, color='#111', lw=0.8)
    ax.set_xlim(t.min(), t.max()); ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("time $t$"); ax.set_ylabel("$Y_t$")
    ax.set_title("Synthetic: series $Y_t$ and regimes (product model, $\\min J$)")
    ax.legend(handles=[Patch(color=REG[k], alpha=0.4, label=f"regime {k+1}")
                       for k in range(3)], fontsize=12, ncol=3, loc='upper right')
    plt.tight_layout(); plt.savefig(OUT + "fig_synth_prod_series.png", dpi=350); plt.close()

    print("|b|_hat:", np.round(beh, 3), " true |b|:", np.round(tb, 3))
    print("a_hat  :", np.round(alh, 3), " true a  :", np.round(ta, 3))
    print("saved fig_synth_prod_cloud.png (unsigned |b|, English, 350 dpi)")


# ==============================================================================
# PART 7.  FIGURES - EXPERIMENT 2 (order selection A3 vs A5)
#   Branch roots on tau-slices, pairwise C-norm matrices, summary bars.
# ==============================================================================


"""
X = g(w,t) branch graphs and catastrophe-order selection by root separation.

The catastrophe polynomial f(X,w,t;nu)=X^n + sum c_j(w,t) X^j is fitted W-FREE
(line-root moment conditions, using only theta(t)=(a,|b|); w is the abstract
Gaussian argument, NOT a reconstructed trajectory).  Its real roots g_i(w,t)
are plotted over the (w,t) domain of existence -- these are the X=g(w,t) graphs.

Order selection: over the region where all n roots are real we measure the
uniform-norm distance between branches
      D_inf(i,j) = max_{(w,t) in Omega_n} | g_i(w,t) - g_j(w,t) | .
If two branches stay too close (small D_inf), the extra branch is redundant and
a lower-degree catastrophe suffices.  A_5 fitted to a 3-branch structure yields a
near-coincident pair; A_3 does not.  Figures at 350 dpi.
"""

OUT = "figs/"
CB = ["#3b6fb6", "#e07b39", "#3f9b5c", "#9b59b6", "#c0392b"]   # branch colours


# ----------------------------------------------------------------- data prep_gwt
def prep_gwt(kind):
    if kind == "synth":
        alpha = [0.10, -0.15, 0.05]; beta = [0.80, -1.20, 0.40]
        d = simulate(alpha, beta, T=30, dt=0.01, seed=314, p_switch=0.6, min_dwell=2.5)
        Y, dt, T = d['Y'], d['dt'], d['T']
        idx, a, b = local_theta(Y, dt, h=120, hb=30, W=None)
        title = "Synthetic"
    else:
        c = pd.read_csv('btc_week_1min.csv')['close'].values[::5]
        Y = np.log(c); T = float(len(Y)); dt = 1.0
        idx, a, b = local_theta(Y, dt, h=48, hb=12, W=None)
        title = "BTC/USD"
    sb = np.median(np.abs(b)); sX = np.sqrt(T) * sb
    tau = idx * dt / T                       # actual time / horizon in [0,1]
    atil = a * T / sX
    btil = np.abs(b) * np.sqrt(T) / sX      # |b|, unsigned -> W-free
    return dict(tau=tau, atil=atil, btil=btil, title=title,
                tmin=float(tau.min()), tmax=float(tau.max()))


# ----------------------------------------------------------------- root grid
def root_grid(nu, degree, tmin, tmax, nt=70, nw=90, cw=2.0):
    """Return list over tau of (omega_array, list-of-real-root-arrays)."""
    taus = np.linspace(max(tmin, 1e-3), tmax, nt)
    data = []
    for t in taus:
        wmax = cw * np.sqrt(t)
        om = np.linspace(-wmax, wmax, nw)
        roots = [cat_roots(nu, o, t, degree) for o in om]
        data.append((t, om, roots))
    return taus, data


def separation(nu, degree, tmin, tmax):
    """min pairwise uniform-norm branch distance over the all-real region,
    the full D_inf matrix, and coverage of the all-real region."""
    _, data = root_grid(nu, degree, tmin, tmax)
    full = []
    tot = 0
    for _, om, roots in data:
        for r in roots:
            tot += 1
            if len(r) == degree:
                full.append(np.sort(r))
    full = np.array(full)
    if len(full) == 0:
        return np.nan, np.full((degree, degree), np.nan), 0.0
    D = np.zeros((degree, degree))
    for i in range(degree):
        for j in range(degree):
            D[i, j] = np.abs(full[:, i] - full[:, j]).max()
    off = D + np.eye(degree) * 1e9
    return off.min(), D, len(full) / tot


# ----------------------------------------------------------------- plotting
def sheets_row(axes, nu, degree, tmin, tmax, taus_show, true_lines=None):
    for ax, t0 in zip(axes, taus_show):
        wmax = 2.0 * np.sqrt(t0)
        om = np.linspace(-wmax, wmax, 240)
        for o in om:
            r = cat_roots(nu, o, t0, degree)
            ax.plot([o] * len(r), r, '.', color=ACCENT, ms=2.2)
        if true_lines is not None:
            ats, bts = true_lines
            for k in range(len(ats)):
                ax.plot(om, ats[k] * t0 + bts[k] * om, color=TRUE, lw=1.8,
                        alpha=0.5, label="true branches" if k == 0 else None)
        ax.set_title(f"$\\tau={t0:.2f}$"); ax.set_xlabel("$w$")


def gwt_figure(kind, degrees=(3, 5)):
    D = prep_gwt(kind)
    taus_show = np.quantile(D['tau'], [0.2, 0.5, 0.85])
    true_lines = None
    if kind == "synth":
        # true branches in the SAME scaled coords (shown for reference only)
        alpha = np.array([0.10, -0.15, 0.05]); beta = np.array([0.80, -1.20, 0.40])
        # rebuild scaling
        d = simulate(list(alpha), list(beta), T=30, dt=0.01, seed=314,
                     p_switch=0.6, min_dwell=2.5)
        idx, a, b = local_theta(d['Y'], d['dt'], h=120, hb=30, W=None)
        sX = np.sqrt(d['T']) * np.median(np.abs(b))
        true_lines = (alpha * d['T'] / sX, beta * np.sqrt(d['T']) / sX)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9.4))
    res = {}
    for row, deg in zip((0, 1), degrees):
        nu, J, cond = fit_moment(D['atil'], D['btil'], D['tau'], deg, ridge=1e-2)
        msep, Dm, cov = separation(nu, deg, D['tmin'], D['tmax'])
        res[deg] = (msep, cov, J)
        sheets_row(ax[row], nu, deg, D['tmin'], D['tmax'], taus_show,
                   true_lines if kind == "synth" else None)
        ax[row, 0].set_ylabel(f"$A_{deg}$\n roots $X=g(w,t)$")
    if kind == "synth":
        ax[0, 0].legend(fontsize=12, loc='upper left')
    fig.suptitle(f"{D['title']}: branches $g_i(w,t)$ (roots $A_n$) over the "
                 f"domain of existence -- $A_3$ (top), $A_5$ (bottom)", y=1.02)
    plt.tight_layout(); plt.savefig(OUT + f"fig_gwt_{kind}.png", dpi=350); plt.close()
    return res, D['tmin'], D['tmax']


def sep_figure(kind):
    D = prep_gwt(kind)
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 12.5))
    for col, deg in zip((0, 1), (3, 5)):
        nu, J, cond = fit_moment(D['atil'], D['btil'], D['tau'], deg, ridge=1e-2)
        msep, Dm, cov = separation(nu, deg, D['tmin'], D['tmax'])
        im = ax[col].imshow(Dm, cmap='viridis', vmin=0)
        for i in range(deg):
            for j in range(deg):
                ax[col].text(j, i, f"{Dm[i,j]:.2f}", ha='center', va='center',
                             color='w' if Dm[i, j] < Dm.max() * 0.6 else 'k', fontsize=12)
        ax[col].set_xticks(range(deg)); ax[col].set_yticks(range(deg))
        ax[col].set_xticklabels([f"$g_{i+1}$" for i in range(deg)])
        ax[col].set_yticklabels([f"$g_{i+1}$" for i in range(deg)])
        ax[col].set_title(f"$A_{deg}$:  $\\|g_i-g_j\\|_\\infty$\n"
                          f"min$={msep:.2f}$, coverage$={cov:.2f}$")
        plt.colorbar(im, ax=ax[col], fraction=0.046, pad=0.04)
    fig.suptitle(f"{D['title']}: pairwise uniform norm between branches", y=1.02)
    plt.tight_layout(); plt.savefig(OUT + f"fig_sep_{kind}.png", dpi=350); plt.close()



def run_exp2():
    """Build all Experiment-2 figures."""
    summary = {}
    for kind in ("synth", "btc"):
        res, tmin, tmax = gwt_figure(kind)
        sep_figure(kind)
        summary[kind] = res
        print(f"{kind}: A3 minSep={res[3][0]:.2f} cov={res[3][1]:.2f} | "
              f"A5 minSep={res[5][0]:.2f} cov={res[5][1]:.2f}")

    # cross-dataset summary
    fig, ax = plt.subplots(2, 1, figsize=(9.0, 9.0))
    labs = ["Synthetic", "BTC/USD"]; xx = np.arange(2); w = 0.35
    for col, (metric, ttl, yl) in enumerate([
            (0, "Min branch separation\n$\\min_{i\\neq j}\\|g_i-g_j\\|_\\infty$",
             r"$\min\|g_i-g_j\|_\infty$"),
            (1, "Domain of existence of all branches", "fraction of domain")]):
        a3 = [summary["synth"][3][metric], summary["btc"][3][metric]]
        a5 = [summary["synth"][5][metric], summary["btc"][5][metric]]
        ax[col].bar(xx - w/2, a3, w, color=TRUE, label="$A_3$")
        ax[col].bar(xx + w/2, a5, w, color=ACCENT, label="$A_5$")
        ax[col].set_xticks(xx); ax[col].set_xticklabels(labs)
        ax[col].set_title(ttl); ax[col].set_ylabel(yl); ax[col].legend()
    fig.suptitle("Order selection: $A_5$ yields close (redundant) branches and small "
                 "domain of existence of all branches", y=1.02)
    plt.tight_layout(); plt.savefig(OUT + "fig_sep_summary.png", dpi=350); plt.close()
    print("saved g(w,t) and separation figures")


# ==============================================================================
# PART 8.  SYMBOLIC VERIFICATION OF THE ALGEBRA (sympy)
#   Checks mu_m, E_{r,p}, M_k, rho_k against symbolic computation.
# ==============================================================================


def run_verification():
    """Symbolic verification of the moment algebra. Needs sympy."""
    import sympy as sp
    w, t, A, B = sp.symbols('omega tau A B', real=True)

    def gauss_E(expr):
        """E[expr], expr polynomial in omega, omega~N(0,tau): omega^m -> mu_m."""
        p = sp.Poly(sp.expand(expr), w)
        res = sp.Integer(0)
        for (m,), c in p.terms():
            if m % 2 == 0:                       # odd moments vanish
                mu = sp.factorial2(m-1) * t**(m//2)     # (m-1)!! tau^{m/2}, mu_0=1
                res += c*mu
        return sp.expand(res)

    print("="*70)
    print("STEP 1  Gaussian central moments  mu_m = E[omega^m], omega~N(0,tau)")
    print("="*70)
    for m in range(0,7):
        print(f"  mu_{m} =", gauss_E(w**m))

    print()
    print("="*70)
    print("STEP 2  Building block  E_{r,p}(tau) = E[ omega^r (A+B omega)^p ]")
    print("="*70)
    for (r,p) in [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
                  (1,1),(1,2),(1,3),(1,4),(2,0),(2,2),(2,3),(3,1),(3,2)]:
        sym = gauss_E(w**r*(A+B*w)**p)
        num = _E_wr_Ap(r,p,0.7,1.3,0.6)                 # code
        ref = float(sym.subs({A:0.7,B:1.3,t:0.6}))      # analytic
        ok = abs(num-ref)<1e-12
        print(f"  E_({r},{p}) = {sym}    [code vs analytic: {'OK' if ok else 'MISMATCH %.2e'%abs(num-ref)}]")

    print()
    print("="*70)
    print("STEP 3  A_3 :  f = X^3 + c1(w,t) X + c0(w,t)   (depressed cubic)")
    print("        L_k = E[ f(A+Bw, w, t) (A+Bw)^k ],  k = 0..4")
    print("        L_k = rho_k(tau) + sum_{j,r,s} [tau^s E_{r,j+k}] c_j^{rs}")
    print("        => row M_k entry for c_j^{rs} is  tau^s * E_{r,j+k},  rho_k=E_{0,3+k}")
    print("="*70)
    # symbolic controls c1,c0 as degree<=3 polynomials
    MONs=[(r,s) for r in range(4) for s in range(4) if r+s<=3]
    c1={(r,s):sp.Symbol(f'c1_{r}{s}') for (r,s) in MONs}
    c0={(r,s):sp.Symbol(f'c0_{r}{s}') for (r,s) in MONs}
    c1p=sum(c1[(r,s)]*w**r*t**s for (r,s) in MONs)
    c0p=sum(c0[(r,s)]*w**r*t**s for (r,s) in MONs)
    X=A+B*w
    f=X**3+c1p*X+c0p
    for k in [0,1,2]:
        Lk=gauss_E(f*(A+B*w)**k)
        print(f"\n  L_{k} = "+str(sp.collect(sp.expand(Lk),list(c1.values())+list(c0.values())))[:400]+" ...")

    # ---- numeric verification of M_k, rho_k vs code, A3 (all k, random point) ----
    def code_row(j_powers,degree,k,r,s,Av,Bv,Tv):
        return (Tv**s)*_E_wr_Ap(r,j+k if False else 0,Av,Bv,Tv)  # placeholder
    print("\n  Numeric check M_k / rho_k  (code vs sympy), A_3, random (A,B,tau):")
    Av,Bv,Tv=0.7,1.3,0.6
    powers=control_powers(3)  # [1,0]
    maxerr=0.0
    for k in range(5):
        Lk=gauss_E(f*(A+B*w)**k).subs({A:Av,B:Bv,t:Tv})
        # sympy: coefficient of each param + constant
        for jc,j in zip([1,0],[c1,c0]):  # j=1 -> c1, j=0 -> c0
            for (r,s) in MONs:
                sym_coef=float(sp.diff(Lk,j[(r,s)]))
                code_coef=(Tv**s)*_E_wr_Ap(r,(1 if j is c1 else 0)+k,Av,Bv,Tv)
                maxerr=max(maxerr,abs(sym_coef-code_coef))
        const_sym=float(Lk.subs({v:0 for v in list(c1.values())+list(c0.values())}))
        rho_code=_E_wr_Ap(0,3+k,Av,Bv,Tv)
        maxerr=max(maxerr,abs(const_sym-rho_code))
    print(f"    max |code - analytic| over all M_k entries & rho_k (k=0..4): {maxerr:.2e}")

    # ---- A5 numeric verification ----
    print("\n  Numeric check M_k / rho_k, A_5 : f=X^5+c3 X^3+c2 X^2+c1 X+c0, k=0..6:")
    cs={j:{(r,s):sp.Symbol(f'c{j}_{r}{s}') for (r,s) in MONs} for j in [3,2,1,0]}
    f5=X**5+sum((sum(cs[j][(r,s)]*w**r*t**s for (r,s) in MONs))*X**j for j in [3,2,1,0])
    maxerr5=0.0
    for k in range(7):
        Lk=gauss_E(f5*(A+B*w)**k).subs({A:Av,B:Bv,t:Tv})
        for j in [3,2,1,0]:
            for (r,s) in MONs:
                sym_coef=float(sp.diff(Lk,cs[j][(r,s)]))
                code_coef=(Tv**s)*_E_wr_Ap(r,j+k,Av,Bv,Tv)
                maxerr5=max(maxerr5,abs(sym_coef-code_coef))
        const_sym=float(Lk.subs({v:0 for jj in cs for v in cs[jj].values()}))
        maxerr5=max(maxerr5,abs(const_sym-_E_wr_Ap(0,5+k,Av,Bv,Tv)))
    print(f"    max |code - analytic| over all M_k entries & rho_k (k=0..6): {maxerr5:.2e}")

    print()
    print("="*70)
    print("STEP 4  End-to-end: exact recovery of a KNOWN cubic from its branches")
    print("        (tests the whole A nu = -h logic; signed beta, sum=0)")
    print("="*70)
    al=np.array([0.10,-0.15,0.05]); be=np.array([0.80,-1.20,0.40])   # sum=0
    assert abs(al.sum())<1e-12 and abs(be.sum())<1e-12
    # true p,q as polynomials in (w,t): p=sum_{i<j} g_i g_j, q=-g1 g2 g3
    g=[al[m]*t+be[m]*w for m in range(3)]
    p_true=sp.expand(g[0]*g[1]+g[0]*g[2]+g[1]*g[2])
    q_true=sp.expand(-g[0]*g[1]*g[2])
    def coeffs(poly):
        P=sp.Poly(poly,w,t).as_dict(); return {(r,s):float(P.get((r,s),0)) for (r,s) in MONs}
    c1_true=coeffs(p_true); c0_true=coeffs(q_true)
    # assemble true nu in code's param order: powers=[1,0], each over MON (catlib order)
    nu_true=np.array([c1_true[(r,s)] for (r,s) in MON]+[c0_true[(r,s)] for (r,s) in MON])

    # build exact cloud: many tau, cycle active branch, feed EXACT (atil=alpha_m, btil=beta_m)
    rng=np.random.default_rng(0)
    N=600; taus=rng.uniform(0.05,1.0,N); mact=rng.integers(0,3,N)
    atil=al[mact]; btil=be[mact]
    nu_hat,J,cond=fit_moment(atil,btil,taus,3,K=4,ridge=1e-12,robust=False)
    print(f"  recovered J={J:.2e}  cond={cond:.1e}")
    print(f"  ||nu_hat - nu_true||_inf = {np.abs(nu_hat-nu_true).max():.2e}")
    # check roots equal true branches at random (w,t)
    err=0
    for _ in range(200):
        ww=rng.normal(0,np.sqrt(0.5)); tt=rng.uniform(0.1,0.9)
        r_hat=np.sort(cat_roots(nu_hat,ww,tt,3))
        r_true=np.sort([al[m]*tt+be[m]*ww for m in range(3)])
        if len(r_hat)==3: err=max(err,np.abs(r_hat-r_true).max())
    print(f"  max |roots_hat - true branches| over random (w,t) = {err:.2e}")
    print()
    print("  => moment machinery + A nu=-h recover the exact cubic to machine eps.")


# ==============================================================================
# PART 9.  COMMAND-LINE ENTRY POINT
# ==============================================================================


_TASKS = {
    "exp1":     run_exp1,
    "prod":     run_prod,
    "exp2":     run_exp2,
    "selftest": run_selftest,
    "verify":   run_verification,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    task = argv[0] if argv else "all"
    if task == "all":
        for name in ("verify", "selftest", "exp1", "prod", "exp2"):
            print("\n=== %s ===" % name)
            _TASKS[name]()
    elif task in _TASKS:
        _TASKS[task]()
    else:
        print(__doc__)
        print("Unknown task %r. Choose one of: %s, all"
              % (task, ", ".join(_TASKS)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())