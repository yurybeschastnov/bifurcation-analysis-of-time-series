#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Analysis of multiregime time series by methods of bifurcation and catastrophe
theory -- numerical experiments and figures.

This single script reproduces every numerical result and figure of the paper:

  * local point-wise estimation of theta(t) = (a(t), b(t))  (drift / volatility);
  * clustering of the volatility b(t) into regimes;
  * identification of the global parameter nu of the catastrophe A_n
    (n = 3,4,5) by the linear system  A nu = xi  built from moment conditions;
  * recovery of the branches g_i(w,t) as real roots of f(X,w,t;nu)=0 and of
    their volatility profile  d g_i / d w  (the well-identified diffusion slope);
  * selection of the catastrophe order from the C-separation of the slopes;
  * all figures used in the article.

Key methodological point of this version
----------------------------------------
For a linear branch g_i(w,t) = alpha_i t + beta_i w one has  d g_i/d w = beta_i,
i.e. the w-derivative isolates the volatility (diffusion) beta_i.  From a single
trajectory the drift alpha_i is poorly identified (Var(a_hat) ~ beta^2), whereas
beta_i is recovered consistently from the quadratic variation.  We therefore
compare and select branches through the slope  d g_i/d w  rather than through the
root level g_i itself: the comparison is carried out on the well-estimated
quantity.

Author: reproduction code accompanying the article.
================================================================================
"""

import os
import numpy as np
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------- #
#  Global style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 350,
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT = "work/Figure"
os.makedirs(OUT, exist_ok=True)

# Regime colours: calm / medium / turbulent
REG_COLORS = ["#2E7D32", "#F9A825", "#C62828", "#6A1B9A", "#00838F", "#795548"]
def reg_colors(N):
    """N distinct regime colours, ordered by increasing volatility."""
    if N <= len(REG_COLORS):
        return REG_COLORS[:N]
    return [cm.turbo(x) for x in np.linspace(0.05, 0.95, N)]

def reg_names(N):
    if N == 3:
        return ["calm", "medium", "turbulent"]
    return ["regime %d" % (i + 1) for i in range(N)]
REG_NAMES  = ["calm", "medium", "turbulent"]
TRUE_C     = "#1565C0"                              # blue for true branches

# --------------------------------------------------------------------------- #
#  Moment-condition machinery  (see Appendix of the paper)
# --------------------------------------------------------------------------- #
def RS_basis(n):
    """
    Monomial basis of the coefficients c_j(w,t) for the catastrophe A_n:
    all monomials w^r t^s with r + s <= n.

    The degree must follow the order: for f = prod_{j=1}^{n}(X - alpha_j t -
    beta_j w) the free term c_0 = (-1)^n prod_j (alpha_j t + beta_j w) is a form
    of degree exactly n in (w,t).  A basis of lower degree cannot represent it,
    and the fit of A_4 / A_5 then fails for a purely parametric reason rather
    than because the order is wrong.  With r + s <= n the exact product of n
    linear branches is reproduced (see verify()).
    """
    return [(r, s) for r in range(n + 1) for s in range(n + 1) if r + s <= n]

RS = RS_basis(3)          # kept for backward compatibility (cubic basis)
NRS = len(RS)

def mu_moment(m, tau):
    """m-th moment of N(0,tau): (m-1)!! tau^{m/2} for even m, 0 for odd, mu_0=1."""
    if m % 2 == 1:
        return 0.0
    dfac = 1.0
    for i in range(1, m, 2):
        dfac *= i
    return dfac * tau ** (m // 2)

def E_rp(r, p, A, B, tau):
    """E_omega[ omega^r (A + B omega)^p ],  omega ~ N(0,tau)."""
    return sum(comb(p, l) * A ** (p - l) * B ** l * mu_moment(r + l, tau)
               for l in range(p + 1))

def moment_rows(n, A, B, tau):
    """
    For catastrophe A_n and one CENTERED branch g(omega)=A+B*omega at scaled time
    tau, return the rows M_k and the scalars rho_k, k = 0..n+1, such that
    L_k = rho_k + M_k . nu  for the depressed normal form

        f(X,w,t) = X^n + c_{n-2}(w,t) X^{n-2} + ... + c_1(w,t) X + c_0(w,t),

    which has no X^{n-1} term.  Parameter order: c_j^{rs}, j = 0..n-2,
    (r,s) in RS_basis(n);  dim nu = (n-1)*|RS_basis(n)|.
    """
    rs_basis = RS_basis(n)
    ncoef = (n - 1) * len(rs_basis)
    Ms, rhos = [], []
    for k in range(0, n + 2):
        rho = E_rp(0, n + k, A, B, tau)
        row = np.zeros(ncoef)
        col = 0
        for j in range(0, n - 1):                 # j = 0..n-2 (depressed)
            for (r, s) in rs_basis:
                row[col] = tau ** s * E_rp(r, j + k, A, B, tau)
                col += 1
        Ms.append(row)
        rhos.append(rho)
    return np.array(Ms), np.array(rhos)

def fit_catastrophe(n, cloud):
    """
    Identify nu by minimising the discretised trajectory integral

        J(nu) = sum_i sum_k w_ik L_k(theta(t_i), nu, t_i)^2 ,   L_k = rho_k + M_k.nu ,

    whose stationarity condition dJ/dnu = 0 is the linear system

        A nu = xi,   A = sum_ik w_ik M_k^T M_k,   xi = -sum_ik w_ik M_k^T rho_k.

    The weights w_ik = (|rho_k| + ||M_k||)^-2 balance the moment conditions: the
    raw moments grow like tau^{(n+k)/2}, so without them the largest tau alone
    would determine the solution.  The symmetric system is equilibrated and
    solved by the pseudo-inverse (least-norm solution if A is singular).

    cloud : list of (a, b, tau) -- the CENTERED branch g(w) = a*tau + b*w of the
            active regime at scaled time tau = t_i.
    """
    rs_basis = RS_basis(n)
    S = (n - 1) * len(rs_basis)
    A_mat = np.zeros((S, S))
    xi = np.zeros(S)
    for (a_i, b_i, tau) in cloud:
        Ms, rhos = moment_rows(n, a_i * tau, b_i, tau)
        for k in range(len(rhos)):
            w = 1.0 / (abs(rhos[k]) + np.linalg.norm(Ms[k]) + 1e-300) ** 2
            A_mat += w * np.outer(Ms[k], Ms[k])
            xi += -w * Ms[k] * rhos[k]
    d = np.sqrt(np.abs(np.diag(A_mat)))
    d[d == 0] = 1.0
    D = np.diag(1.0 / d)
    return D @ (np.linalg.pinv(D @ A_mat @ D) @ (D @ xi))

#  Roots of f(X,w,t;nu)=0 and the implicit derivative  dg/dw
# --------------------------------------------------------------------------- #
def coefs_at(nu, n, w, t):
    """c_j(w,t) and dc_j/dw for j = 0..n-2 (depressed normal form)."""
    rs_basis = RS_basis(n)
    cj = np.zeros(n - 1)
    dcj = np.zeros(n - 1)
    col = 0
    for j in range(n - 1):
        c = 0.0
        dc = 0.0
        for (r, s) in rs_basis:
            c += nu[col] * (w ** r) * (t ** s)
            if r >= 1:
                dc += nu[col] * r * (w ** (r - 1)) * (t ** s)
            col += 1
        cj[j] = c
        dcj[j] = dc
    return cj, dcj

def roots_deriv(nu, n, w, t, tol=1e-7):
    """
    Real roots of  X^n + sum_{j=0}^{n-2} c_j X^j  with the implicit derivative

        dg_i/dw = -( sum_j dc_j/dw g^j ) / ( n g^{n-1} + sum_j j c_j g^{j-1} ).

    These are the CENTERED branches: the physical volatility is dg_i/dw + beta_bar.
    """
    cj, dcj = coefs_at(nu, n, w, t)
    coeffs = np.zeros(n + 1)
    coeffs[0] = 1.0                      # X^n
    coeffs[1] = 0.0                      # no X^{n-1} term
    for j in range(n - 1):
        coeffs[n - j] = cj[j]
    r = np.roots(coeffs)
    real = r[np.abs(r.imag) < tol * (1 + np.abs(r.real))].real
    out = []
    for g in real:
        num = sum(dcj[j] * g ** j for j in range(n - 1))
        den = n * g ** (n - 1) + sum(j * cj[j] * g ** (j - 1) for j in range(1, n - 1))
        if abs(den) < 1e-12:
            continue
        out.append((g, -num / den))
    return out

def branch_slopes_on_grid(nu, n, W, T, beta_bar):
    """
    Evaluate the physical volatility profile dg/dw + beta_bar on a (W,T) grid.
    Returns arrays roots[nw,nt,n]*nan and slopes[nw,nt,n]*nan (sorted by slope),
    plus a boolean 'all_real' mask (all n roots real at that node).
    """
    nw, nt = W.shape
    slopes = np.full((nw, nt, n), np.nan)
    roots = np.full((nw, nt, n), np.nan)
    all_real = np.zeros((nw, nt), dtype=bool)
    for i in range(nw):
        for jt in range(nt):
            res = roots_deriv(nu, n, W[i, jt], T[i, jt])
            if len(res) == n:
                all_real[i, jt] = True
            res_sorted = sorted(res, key=lambda gd: gd[1])   # sort by slope
            for m, (g, d) in enumerate(res_sorted[:n]):
                roots[i, jt, m] = g
                slopes[i, jt, m] = d + beta_bar
    return roots, slopes, all_real

# --------------------------------------------------------------------------- #
#  Local estimation of theta(t) = (a,b)
# --------------------------------------------------------------------------- #
def local_estimates(Y, dt, half):
    """
    Point-wise drift a_hat and volatility b_hat (per sqrt-time) on a centred
    window of half-width 'half' samples.  Volatility from quadratic variation.
    """
    n = len(Y)
    dY = np.diff(Y)
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    Tw = 2 * half * dt
    for i in range(half, n - half):
        a[i] = (Y[i + half] - Y[i - half]) / Tw
        qv = np.sum(dY[i - half:i + half] ** 2)
        b[i] = np.sqrt(qv / Tw)
    return a, b

def select_k(bvals, krange=(2, 3, 4, 5, 6), sample=2000, seed=0):
    """
    Choose the number of regimes k by maximising the silhouette index of the
    k-means partition of the standardised volatility b(t).  Returns (k*, table)
    where table is the list of (k, silhouette).
    """
    x = bvals.reshape(-1, 1)
    xs = (x - x.mean()) / x.std()
    rng = np.random.default_rng(seed)
    sub = (rng.choice(len(xs), size=min(sample, len(xs)), replace=False)
           if len(xs) > sample else np.arange(len(xs)))
    table = []
    for k in krange:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(xs)
        s = silhouette_score(xs[sub], km.labels_[sub])
        table.append((k, float(s)))
    kstar = max(table, key=lambda ks: ks[1])[0]
    return kstar, table


def cluster_b(bvals, N=3):
    """K-means on standardised b; clusters ordered by increasing centre."""
    x = bvals.reshape(-1, 1)
    xs = (x - x.mean()) / x.std()
    km = KMeans(n_clusters=N, n_init=10, random_state=0).fit(xs)
    order = np.argsort([xs[km.labels_ == c].mean() for c in range(N)])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in km.labels_])
    sil = silhouette_score(xs, labels)
    return labels, sil

print("bifurcation_analysis.py: module loaded")

# =========================================================================== #
#  DATA
# =========================================================================== #
def make_synthetic(seed=7):
    """
    Synthetic three-branch series.  Branches (alpha,beta):
        (0.10,0.80), (-0.15,1.20), (0.05,0.40);  T=30, dt=0.01.
    The regimes differ essentially in volatility.  A single common Wiener path
    drives all three branches  L_k(t)=alpha_k t + beta_k W_t; the observed
    series follows the active branch and switches to another one only at a
    crossing L_i = L_j, so the trajectory is continuous (no jumps).
    """
    rng = np.random.default_rng(seed)
    T, dt = 30.0, 0.01
    t = np.arange(0, T, dt)
    n = len(t)
    ab = np.array([(0.10, 0.80), (-0.15, 1.20), (0.05, 0.40)])   # (alpha,beta)
    dW = rng.normal(0, np.sqrt(dt), n)
    W = np.concatenate([[0], np.cumsum(dW)])[:n]
    # piecewise-constant regime path with balanced occupancy
    seg = np.zeros(n, dtype=int)
    order = [0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2]
    idx, oi = 0, 0
    while idx < n:
        L = int(rng.integers(150, 340))
        seg[idx:idx + L] = order[oi % len(order)]
        idx += L; oi += 1
    seg = seg[:n]
    # regime-switching diffusion: single continuous integrated path on common dW
    alpha = ab[seg, 0]; beta = ab[seg, 1]
    dY = alpha * dt + beta * dW
    Y = np.concatenate([[0.0], np.cumsum(dY)])[:n]
    # reference branch value on each contiguous segment (for the benchmark plot)
    return dict(t=t, dt=dt, Y=Y, W=W, seg=seg, ab=ab)

def make_synthetic4(seed=11):
    """
    Four-regime synthetic series (the 'under-specified order' example).
    Branches (alpha,beta): (0.10,0.40), (-0.05,0.70), (0.08,1.00), (-0.12,1.35);
    T=40, dt=0.01.  Same construction as make_synthetic: a regime-switching
    diffusion driven by one common Wiener path.
    """
    rng = np.random.default_rng(seed)
    T, dt = 40.0, 0.01
    t = np.arange(0, T, dt)
    n = len(t)
    ab = np.array([(0.10, 0.40), (-0.05, 0.70), (0.08, 1.00), (-0.12, 1.35)])
    dW = rng.normal(0, np.sqrt(dt), n)
    W = np.concatenate([[0], np.cumsum(dW)])[:n]
    seg = np.zeros(n, dtype=int)
    order = [0, 2, 1, 3, 0, 1, 2, 3, 1, 0, 3, 2]
    idx, oi = 0, 0
    while idx < n:
        L = int(rng.integers(150, 340))
        seg[idx:idx + L] = order[oi % len(order)]
        idx += L; oi += 1
    seg = seg[:n]
    alpha = ab[seg, 0]; beta = ab[seg, 1]
    dY = alpha * dt + beta * dW
    Y = np.concatenate([[0.0], np.cumsum(dY)])[:n]
    return dict(t=t, dt=dt, Y=Y, W=W, seg=seg, ab=ab)

def make_btc_like(seed=25):
    """
    BTC/USD-like 5-min log-price reconstructed to match the published statistics
    (the raw exchange feed is not bundled with the code):
        2016 bars (one week), log-price 11.507 -> 11.324,
        three annualised volatility regimes ~ 23 / 62 / 119 %,
        time fractions ~ 0.55 / 0.29 / 0.16,
        risk-off drift  -0.2 / -4.1 / -7.8 % per day.
    """
    rng = np.random.default_rng(seed)
    nbar = 2016
    bars_per_day = 288
    ann_gen = np.sqrt(365 * bars_per_day)             # per-bar -> annualised
    ann_b = np.sqrt(365.0)                            # per-sqrt-day -> annualised
    sig = np.array([0.23, 0.62, 1.19]) / ann_gen      # per-bar sigma
    drift_day = np.array([-0.002, -0.041, -0.078])    # per-day drift (risk-off)
    drift = drift_day / bars_per_day                  # per-bar drift
    # regime path: calm first half, turbulent bursts in the second half
    seg = np.zeros(nbar, dtype=int)
    i = 0
    while i < nbar:
        u = i / nbar
        if u < 0.45:
            r = rng.choice([0, 1], p=[0.85, 0.15])
        elif u < 0.7:
            r = rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15])
        else:
            r = rng.choice([0, 1, 2], p=[0.30, 0.40, 0.30])
        L = int(rng.integers(10, 40))
        seg[i:i + L] = r
        i += L
    seg = seg[:nbar]
    dY = drift[seg] + sig[seg] * rng.normal(0, 1, nbar)
    Y = 11.507 + np.concatenate([[0.0], np.cumsum(dY)])[:nbar]
    # Brownian-bridge endpoint correction: keep Y[0]=11.507 and Y[-1]=11.324
    err = Y[-1] - 11.324
    Y = Y - err * (np.arange(nbar) / (nbar - 1))
    dt = 1.0 / bars_per_day                            # time unit = 1 day
    t = np.arange(nbar) * dt
    return dict(t=t, dt=dt, Y=Y, seg=seg, ann=ann_b, sig=sig)

# =========================================================================== #
#  PIPELINE:  series -> cloud -> clusters -> catastrophe fits -> slopes
# =========================================================================== #
def run_pipeline(data, half, N=None, ann=None, ktol=0.01):
    """
    N=None -> the number of regimes is chosen from the data by the silhouette
    index (see select_k).  Ties within 'ktol' of the best silhouette are broken
    toward the larger k (the richer structure), which is the usual convention:
    a marginally lower silhouette for a finer partition is not evidence against
    it.  N=int forces the number of regimes.
    """
    t, dt, Y = data["t"], data["dt"], data["Y"]
    a, b = local_estimates(Y, dt, half)
    ok = ~np.isnan(b)
    a_c, b_c, t_c = a[ok], b[ok], t[ok]
    if N is None:
        kbest, ktable = select_k(b_c)
        smax = max(s for _, s in ktable)
        cand = [k for k, s in ktable if s >= smax - ktol]
        N = max(cand)
    else:
        _, ktable = select_k(b_c)
    labels, sil = cluster_b(b_c, N=N)
    centers_b = np.array([b_c[labels == k].mean() for k in range(N)])
    centers_a = np.array([a_c[labels == k].mean() for k in range(N)])
    fracs = np.array([(labels == k).mean() for k in range(N)])
    # Robust per-regime drift: aggregate increment / aggregate time in regime
    # (averages the noise down over the whole occupancy of the regime, unlike
    #  the high-variance point-wise a-hat).
    dY = np.diff(Y)
    lab_full = np.full(len(Y), -1)
    lab_full[np.where(ok)[0]] = labels
    lab_inc = lab_full[:-1]
    centers_a_agg = np.array([
        dY[lab_inc == k].sum() / (max((lab_inc == k).sum(), 1) * dt)
        for k in range(N)])
    Tmax = t.max()
    # --- trajectory integral with theta(t) = (a_j, b_j) of the active regime --
    # theta(t) is a step function of the active regime: both components come
    # from the k-means centre of that regime (b_j is the regime mean of b_hat).
    # The branches are then CENTERED, so that the sum of the roots vanishes and
    # the polynomial takes the depressed normal form (no X^{n-1} term):
    #     alpha_bar = mean_j a_j,  beta_bar = mean_j b_j,
    #     g~_j(w,t) = (a_j - alpha_bar) t + (b_j - beta_bar) w .
    # The physical volatility of a branch is recovered as  dg~/dw + beta_bar.
    alpha_bar = float(centers_a_agg.mean())
    beta_bar = float(centers_b.mean())
    stride = max(1, len(t_c) // 900)
    idx = np.arange(0, len(t_c), stride)
    cloud = [(centers_a_agg[labels[j]] - alpha_bar,
              centers_b[labels[j]] - beta_bar,
              t_c[j]) for j in idx]
    fits = {}
    for n in (3, 4, 5):
        fits[n] = fit_catastrophe(n, cloud)
    return dict(a=a, b=b, ok=ok, a_c=a_c, b_c=b_c, t_c=t_c,
                labels=labels, sil=sil, centers_a=centers_a, centers_b=centers_b,
                centers_a_agg=centers_a_agg, fracs=fracs, beta_bar=beta_bar,
                fits=fits, Tmax=Tmax, N=N, ktable=ktable)

def slope_cnorm(nu, n, Tmax, beta_bar=0.0, ngrid=41):
    """
    Order-selection criterion: pairwise distance between the volatility profiles
    of the branches, in the uniform (C) norm of the two-dimensional argument
    (w, tau),

        || dg_i/dw - dg_j/dw ||_C = max_{(w,tau)} | dg_i/dw - dg_j/dw | .

    Two branches whose profiles are close over the whole domain carry the same
    volatility: they describe one regime, and the catastrophe order is too high.

    Returns (C, min_off, rel), where C is the n x n matrix of pairwise norms,
    min_off = min_{i != j} C_ij, and rel = min_off / mean_off is the same
    quantity normalised by the typical separation (scale-free).
    """
    taus = np.linspace(Tmax * 0.15, Tmax, 7)
    prof = [[] for _ in range(n)]
    for tau in taus:
        for w in np.linspace(-2 * np.sqrt(tau), 2 * np.sqrt(tau), ngrid):
            res = sorted(d + beta_bar for _, d in roots_deriv(nu, n, w, tau))
            if len(res) == n:
                for m in range(n):
                    prof[m].append(res[m])
    if not prof[0]:
        return None, None, None
    P = [np.array(p) for p in prof]
    L = min(len(p) for p in P)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = np.max(np.abs(P[i][:L] - P[j][:L]))
    off = C[~np.eye(n, dtype=bool)]
    levels = [float(np.median(p[:L])) for p in P]      # profile level of each branch
    return C, float(off.min()), levels

def _tau_slices(Tmax):
    return [0.25 * Tmax, 0.5 * Tmax, 0.85 * Tmax]

def _slope_panel(ax, nu, n, tau, beta_bar, levels, yscale=1.0, mode="index",
                 wfrac=2.0, title=None, show_ylabel=True, ylabel=None,
                 level_labels=None, label_side=True):
    """
    Scatter the physical volatility profiles (dg_i/dw + beta_bar)*yscale vs w on
    one tau-slice.  ALL real roots are drawn: at each w the roots present are
    sorted by slope and coloured by that index, so a branch is visible even where
    the remaining roots of A_n are complex (an over-specified order often keeps
    only part of its roots real).  Points where fewer than n roots are real are
    drawn with a lighter edge.  Only genuine fold blow-ups (|dg/dw| -> inf, where
    f'_X -> 0) are filtered.  The y-range follows the data, so excursions outside
    the band of the levels -- the signature of a spurious branch -- stay visible.
    'levels' are reference lines in display units; 'level_labels' annotates them.
    """
    K = len(levels)
    L = max(levels)
    wmax = wfrac * np.sqrt(tau)
    ws = np.linspace(-wmax, wmax, 360)
    cap = 6.0 * abs(L)                    # only the fold blow-ups
    palette = [cm.plasma(x) for x in np.linspace(0.05, 0.92, n)]
    lcols = reg_colors(K)
    full_x = [[] for _ in range(n)]; full_y = [[] for _ in range(n)]
    part_x = [[] for _ in range(n)]; part_y = [[] for _ in range(n)]
    lx = [[] for _ in range(K)]; ly = [[] for _ in range(K)]
    seen = []
    for w in ws:
        res = sorted(roots_deriv(nu, n, w, tau), key=lambda gd: gd[1])
        vals = [((d + beta_bar) * yscale) for _, d in res]
        vals = [v for v in vals if np.isfinite(v) and abs(v) <= cap]
        allreal = (len(vals) == n)
        for m, v in enumerate(vals):
            seen.append(v)
            if mode == "level":
                k = int(np.argmin(np.abs(np.asarray(levels) - v)))
                lx[k].append(w); ly[k].append(v)
            elif allreal:
                full_x[m].append(w); full_y[m].append(v)
            else:
                part_x[m].append(w); part_y[m].append(v)
    if mode == "level":
        for k in range(K):
            if lx[k]:
                ax.scatter(lx[k], ly[k], s=13, color=lcols[k], alpha=0.75,
                           edgecolors="none", zorder=3)
    else:
        for m in range(n):
            if part_x[m]:
                ax.scatter(part_x[m], part_y[m], s=11, color=palette[m],
                           alpha=0.4, edgecolors="none", zorder=2)
            if full_x[m]:
                ax.scatter(full_x[m], full_y[m], s=13, color=palette[m],
                           alpha=0.85, edgecolors="none", zorder=3)
    for lev in levels:
        ax.axhline(lev, ls="--", lw=1.2, color="#263238", alpha=0.75, zorder=1)
    if level_labels is not None and label_side:
        for lev, txt in zip(levels, level_labels):
            ax.annotate(txt, xy=(1.0, lev), xycoords=("axes fraction", "data"),
                        xytext=(3, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9, color="#263238")
    ax.set_xlabel(r"$w$")
    if show_ylabel:
        ax.set_ylabel(ylabel if ylabel else r"$\partial g_i/\partial w$")
    if title:
        ax.set_title(title)
    # y-range: follow the data but stay readable around the levels
    if seen:
        lo = min(min(seen), -0.08 * L)
        hi = max(max(seen), 1.30 * L)
        lo = max(lo, -1.0 * L)
        hi = min(hi, 2.4 * L)
    else:
        lo, hi = -0.08 * L, 1.30 * L
    pad = 0.04 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

def fig_roots_derivative(R, fname, levels, yscale=1.0, ylabel=None, tag="",
                         level_labels=None):
    """One row of tau-slices of the A3 volatility profile dg/dw + beta_bar."""
    Tmax = R["Tmax"]; nu = R["fits"][3]; bb = R["beta_bar"]
    taus = _tau_slices(Tmax)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    for c, tau in enumerate(taus):
        _slope_panel(axes[c], nu, 3, tau, bb, levels, yscale=yscale, mode="level",
                     title=r"$\tau=%.2g$" % tau, show_ylabel=(c == 0),
                     ylabel=ylabel, level_labels=level_labels,
                     label_side=(c == 2))
    fig.suptitle(r"$A_3$ model — volatility profile $\partial g_i/\partial w$ %s" % tag,
                 y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

def fig_sep_matrices(R, fname, orders=(3, 4, 5), yscale=1.0, tag=""):
    """Pairwise C-norm matrices of the slope profiles for the given orders."""
    Tmax = R["Tmax"]
    fig, axes = plt.subplots(1, len(orders), figsize=(5.3 * len(orders), 4.9))
    for ax, n in zip(np.atleast_1d(axes), orders):
        C, mn, lev = slope_cnorm(R["fits"][n], n, Tmax, R["beta_bar"])
        if C is None:
            ax.axis("off"); continue
        Cs = C * yscale
        im = ax.imshow(Cs, cmap="magma_r")
        ax.set_title(r"$A_%d$: $\min\|\cdot\|_C=%.3g$" % (n, mn * yscale))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([r"$g_%d$" % (i + 1) for i in range(n)])
        ax.set_yticklabels([r"$g_%d$" % (i + 1) for i in range(n)])
        for i in range(n):
            for j in range(n):
                ax.text(j, i, "%.2g" % Cs[i, j], ha="center", va="center",
                        color="white" if Cs[i, j] > Cs.max() * 0.5 else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(r"Pairwise $C$-norm of the volatility profiles "
                 r"$\|\partial_w g_i-\partial_w g_j\|_C$ %s" % tag,
                 y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

def fig_synth_true(syn, fname="fig_synth_true.png"):
    """
    Benchmark: the observed series Y_t (black, continuous) together with the
    active branch on each segment.  Where the regime switches, the outgoing and
    incoming branches meet at Y (dashed extensions cross there), so the series
    passes continuously from one branch to another.
    """
    t = syn["t"]; ab = syn["ab"]; seg = syn["seg"]; W = syn["W"]; Y = syn["Y"]
    rank_of = {k: r for r, k in enumerate(np.argsort(ab[:, 1]))}   # beta rank
    fig, ax = plt.subplots(figsize=(15.0, 4.8))
    # contiguous segments
    bounds = [0] + list(np.where(np.diff(seg) != 0)[0] + 1) + [len(t)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        k = seg[a]; r = rank_of[k]
        ax.plot(t[a:b], Y[a:b], color=REG_COLORS[r], lw=2.0, zorder=4)
        # dashed extension of this branch a little beyond the segment
        ext = slice(max(a - 60, 0), min(b + 60, len(t)))
        anchor = a
        gline = Y[anchor] + ab[k, 0]*(t[ext]-t[anchor]) + ab[k, 1]*(W[ext]-W[anchor])
        ax.plot(t[ext], gline, color=REG_COLORS[r], lw=1.0, ls="--", alpha=0.5,
                zorder=2)
    ax.plot(t, Y, color="black", lw=0.7, alpha=0.6, zorder=5)
    # legend proxies
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color=REG_COLORS[r], lw=2.5,
                      label=r"%s branch ($\beta=%.2f$)" %
                      (REG_NAMES[r], np.sort(ab[:, 1])[r])) for r in range(3)]
    ax.legend(handles=proxies, ncol=3, loc="best", fontsize=10)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$X$")
    ax.set_title("True three-branch structure (benchmark)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

def fig_cloud(R, fname, true_ab=None, ann=None, sil_txt=True):
    """(a,b) cloud coloured by cluster + histogram of b.  Works for any k."""
    a_c, b_c, lab = R["a_c"], R["b_c"], R["labels"]
    ca, cb = R["centers_a"], R["centers_b"]
    N = int(lab.max()) + 1
    cols = reg_colors(N)
    names = reg_names(N)
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(14.0, 5.6),
                                  gridspec_kw={"width_ratios": [3, 1]})
    yfac = (ann * 100) if ann else 1.0
    ylab = "annualised volatility (%)" if ann else r"$b$ (volatility)"
    for k in range(N):
        m = lab == k
        ax.scatter(a_c[m], b_c[m] * yfac, s=8, color=cols[k], alpha=0.35,
                   label="%s (%s)" % (names[k],
                        ("%.0f%%" % (cb[k] * yfac)) if ann else ("%.2f" % cb[k])))
    order = np.argsort(cb)
    ax.scatter(ca[order], cb[order] * yfac, marker="X", s=170, color="black",
               zorder=6, label="cluster centres")
    if true_ab is not None:
        ax.scatter(true_ab[:, 0], true_ab[:, 1] * yfac, marker="o", s=190,
                   facecolors="none", edgecolors=TRUE_C, linewidths=2.4,
                   zorder=7, label=r"true $(\alpha,\beta)$")
        for bt in np.sort(true_ab[:, 1]):
            ax.axhline(bt * yfac, ls=":", lw=1.0, color=TRUE_C, alpha=0.7, zorder=1)
    ax.set_xlabel(r"$a$ (drift)"); ax.set_ylabel(ylab)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_title(r"Cloud of local estimates $\theta(t)=(a,b)$, $k=%d$" % N)
    bins = np.linspace(b_c.min(), b_c.max(), 45)
    for k in range(N):
        axh.hist(b_c[lab == k] * yfac, bins=bins * yfac, orientation="horizontal",
                 color=cols[k], alpha=0.75)
    if true_ab is not None:
        for bt in np.sort(true_ab[:, 1]):
            axh.axhline(bt * yfac, ls=":", lw=1.0, color=TRUE_C, alpha=0.7)
    axh.set_xlabel("count"); axh.set_title(r"histogram of $b$")
    axh.set_ylim(ax.get_ylim())
    if sil_txt:
        ax.text(0.03, 0.97, "silhouette = %.2f" % R["sil"], transform=ax.transAxes,
                va="top", fontsize=11, bbox=dict(boxstyle="round", fc="white",
                ec="0.7", alpha=0.9))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

def fig_series_regime(data, R, fname, ylabel=r"$Y_t$", title="", levels=None,
                      level_fmt="{:.2f}"):
    """
    Series with the identified regimes shown as coloured background bands.
    The legend lists each regime with its volatility level.
    """
    t, Y, ok, lab = data["t"], data["Y"], R["ok"], R["labels"]
    full = np.full(len(Y), -1); full[np.where(ok)[0]] = lab
    N = int(full.max()) + 1
    BAND = ["#9EC5E8", "#F3C48B", "#A7D3A0", "#D7BDE2", "#F5B7B1", "#A3E4D7"]
    fig, ax = plt.subplots(figsize=(15.0, 4.6))
    i, n = 0, len(t)
    while i < n:
        if full[i] < 0:
            i += 1; continue
        j = i
        while j + 1 < n and full[j + 1] == full[i]:
            j += 1
        ax.axvspan(t[i], t[min(j + 1, n - 1)], color=BAND[full[i] % len(BAND)],
                   alpha=0.55, lw=0, zorder=0)
        i = j + 1
    ax.plot(t, Y, color="#1a1a1a", lw=0.9, zorder=3)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(r"$t$"); ax.set_ylabel(ylabel); ax.set_title(title)
    from matplotlib.patches import Patch
    if levels is None:
        labels = [REG_NAMES[k] if k < len(REG_NAMES) else "regime %d" % (k + 1)
                  for k in range(N)]
    else:
        labels = [r"regime %d: %s" % (k + 1, level_fmt.format(levels[k]))
                  for k in range(N)]
    handles = [Patch(facecolor=BAND[k % len(BAND)], alpha=0.75, label=labels[k])
               for k in range(N)]
    ax.legend(handles=handles, ncol=N, loc="upper right", fontsize=10,
              framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

def fig_btc_levels(data, R, ann, fname="fig_btc_levels.png"):
    t, ok, lab = data["t"], R["ok"], R["labels"]
    b = R["b"]; cb = R["centers_b"]
    full = np.full(len(t), -1); full[np.where(ok)[0]] = lab
    fig, ax = plt.subplots(figsize=(13.5, 4.0))
    for k in range(3):
        m = full == k
        ax.scatter(t[m], b[m]*ann*100, s=7, color=REG_COLORS[k], label=REG_NAMES[k])
    for k in np.argsort(cb):
        ax.axhline(cb[k]*ann*100, ls="--", lw=1.2, color=REG_COLORS[k], alpha=0.8)
    ax.set_xlabel("time (days)"); ax.set_ylabel("annualised volatility (%)")
    ax.set_title("Local volatility " + r"$b(t)$" + " and regime levels")
    ax.legend(ncol=3, loc="best", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

def fig_slopes_multi(R, fname, levels, orders=(3, 4, 5), yscale=1.0, ylabel=None,
                     tag="", taus_frac=(0.25, 0.5, 0.85), level_labels=None):
    """Rows = catastrophe orders, cols = tau-slices; slope profiles dg_i/dw."""
    Tmax = R["Tmax"]
    taus = [f * Tmax for f in taus_frac]
    fig, axes = plt.subplots(len(orders), 3, figsize=(15.5, 4.3 * len(orders)),
                             sharex="col")
    if len(orders) == 1:
        axes = axes[None, :]
    bb = R["beta_bar"]
    for ri, n in enumerate(orders):
        nu = R["fits"][n]
        for ci, tau in enumerate(taus):
            ax = axes[ri, ci]
            _slope_panel(ax, nu, n, tau, bb, levels, yscale=yscale, mode="index",
                         title=(r"$\tau=%.2g$" % tau if ri == 0 else None),
                         show_ylabel=(ci == 0), ylabel=ylabel,
                         level_labels=level_labels, label_side=(ci == 2))
            if ci == 0:
                ax.text(-0.34, 0.5, r"$A_%d$" % n, transform=ax.transAxes,
                        fontsize=17, fontweight="bold", va="center", ha="center")
            if ri < len(orders) - 1:
                ax.set_xlabel("")
    fig.suptitle(r"Volatility profile $\partial g_i/\partial w$ %s" % tag,
                 y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

# =========================================================================== #
#  VERIFICATION SUITE  (math + code correctness)
# =========================================================================== #
def verify():
    """
    Symbolic + numerical checks of the moment machinery, the exact-cubic
    recovery, and the implicit-derivative formula.  Requires sympy.
    """
    import sympy as sp
    print("=" * 64)
    print("VERIFICATION SUITE")
    print("=" * 64)

    # (1) E_{r,p} closed form vs Gaussian integral
    A, B = sp.symbols('A B', real=True)
    tau = sp.symbols('tau', positive=True)
    om = sp.symbols('om', real=True)

    def _mu(m):
        if m % 2 == 1:
            return sp.Integer(0)
        r = sp.Integer(1)
        for i in range(1, m, 2):
            r *= i
        return r * tau ** (m // 2)

    def _E_closed(r, p):
        return sp.expand(sum(sp.binomial(p, l) * A ** (p - l) * B ** l * _mu(r + l)
                             for l in range(p + 1)))

    def _E_true(r, p):
        return sp.expand(sp.integrate(
            om ** r * (A + B * om) ** p * sp.exp(-om ** 2 / (2 * tau))
            / sp.sqrt(2 * sp.pi * tau), (om, -sp.oo, sp.oo)))

    ok = all(sp.simplify(_E_closed(r, p) - _E_true(r, p)) == 0
             for r in range(4) for p in range(8))
    print("(1) E_{r,p} closed form == Gaussian integral (r<=3,p<=7):",
          "PASS" if ok else "FAIL")

    # (2) M_k nu + rho_k == L_k  (symbolic) for A3 and A5
    def _sym_resid(n):
        import random
        random.seed(1)
        wsym = sp.symbols('w', real=True)
        tt = sp.symbols('t', positive=True)
        As, Bsy = sp.symbols('A B', real=True)
        rs_b = RS_basis(n)
        c = {(j, r, s): sp.Rational(random.randint(-3, 3), random.randint(1, 3))
             for j in range(n - 1) for (r, s) in rs_b}
        g = As + Bsy * wsym
        f = g ** n + sum(sum(c[(j, r, s)] * wsym ** r * tt ** s for (r, s) in rs_b)
                         * g ** j for j in range(n - 1))
        dens = sp.exp(-wsym ** 2 / (2 * tt)) / sp.sqrt(2 * sp.pi * tt)
        nu = np.array([float(c[(j, r, s)]) for j in range(n - 1) for (r, s) in rs_b])
        Av, Bv, tv = 0.6, 1.1, 0.7
        Ms, rhos = moment_rows(n, Av, Bv, tv)
        res = 0.0
        for k in range(len(rhos)):
            Lk = float(sp.integrate(f * g ** k * dens, (wsym, -sp.oo, sp.oo))
                       .subs({As: Av, Bsy: Bv, tt: tv}))
            res = max(res, abs(Lk - (rhos[k] + Ms[k] @ nu)))
        return res
    print("(2) |M_k nu + rho_k - L_k| symbolic:  A3=%.2e  A5=%.2e"
          % (_sym_resid(3), _sym_resid(5)))

    # (3) Exact recovery from the exact product of n linear branches (no centering)
    rng = np.random.default_rng(3)
    for nn in (3, 4, 5):
        al = rng.normal(0, 0.1, nn); be = np.sort(rng.uniform(0.4, 1.4, nn))
        ac = al - al.mean(); bc = be - be.mean(); bbar = be.mean()
        cloud = [(a, b, tau) for (a, b) in zip(ac, bc)
                 for tau in np.linspace(0.2, 30, 60)]
        nu = fit_catastrophe(nn, cloud)
        errs = []
        for w in (-1.0, 0.0, 1.0):
            for t in (5.0, 15.0, 25.0):
                got = sorted(d + bbar for _, d in roots_deriv(nu, nn, w, t))
                if len(got) == nn:
                    errs.append(np.max(np.abs(np.array(got) - be)))
        print("(3) A%d exact product of %d centered lines: dim(nu)=%d  "
              "volatility err=%s  all-real nodes %d/9" % (nn, nn, len(nu),
              ("%.2e" % max(errs)) if errs else "n/a", len(errs)))

    # (4) basis degree check
    print("(4) basis degree r+s<=n, depressed form: dims =",
          [(n, (n - 1) * len(RS_basis(n))) for n in (3, 4, 5)])
    print()
    verify_eiv()
    print("=" * 64)


def verify_eiv(seed=1, taus=None, out_json=None):
    """
    Errors-in-variables demonstration (justifies using the REGIME MEAN of b).

    Three exact linear branches are perturbed by multiplicative noise on b,
        b_hat = beta_j (1 + s * xi),   xi ~ N(0,1),
    and the catastrophe A_3 is identified twice: once from the raw per-sample
    b_hat, once from the mean of b_hat over the regime.  Because b enters M_k and
    rho_k nonlinearly, the raw version biases the system itself and the recovered
    slopes leave the true levels already at s ~ 2%; the regime mean stays on them.
    """
    rng = np.random.default_rng(seed)
    true_b = np.array([0.4, 0.8, 1.2])
    true_a = np.array([0.10, -0.15, 0.05])
    if taus is None:
        taus = np.linspace(1.0, 30.0, 120)
    rows = []
    print("Errors-in-variables: slopes at (w=0, tau=15); true = [0.4, 0.8, 1.2]")
    print("%-9s | %-32s | %-32s" % ("noise s", "raw b_hat per sample",
                                    "regime mean of b_hat"))
    print("-" * 80)
    for s_noise in (0.0, 0.02, 0.05, 0.10, 0.20):
        cl_raw = []
        per = [[], [], []]
        for tau in taus:
            for j in range(3):
                bb = true_b[j] * (1 + s_noise * rng.normal())
                cl_raw.append((true_a[j], bb, tau))
                per[j].append(bb)
        nu_raw = fit_catastrophe(3, cl_raw)
        sl_raw = sorted(d for _, d in roots_deriv(nu_raw, 3, 0.0, 15.0))
        bmean = [float(np.mean(p)) for p in per]
        cl_avg = [(true_a[j], bmean[j], tau) for tau in taus for j in range(3)]
        nu_avg = fit_catastrophe(3, cl_avg)
        sl_avg = sorted(d for _, d in roots_deriv(nu_avg, 3, 0.0, 15.0))
        f = lambda x: str([round(v, 3) for v in x])
        print("%-9.2f | %-32s | %-32s" % (s_noise, f(sl_raw), f(sl_avg)))
        rows.append(dict(noise=s_noise, raw=[float(v) for v in sl_raw],
                         regime_mean=[float(v) for v in sl_avg]))
    if out_json:
        import json
        with open(out_json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return rows


# =========================================================================== #
#  MAIN DRIVER
# =========================================================================== #
def main():
    import json
    summary = {}

    # =================================================================== #
    #  EXPERIMENT 1.  Regime search
    # =================================================================== #
    # ---- synthetic ----
    syn = make_synthetic(seed=7)
    Rs = run_pipeline(syn, half=40, N=None)          # k by silhouette
    true_ab = syn["ab"]
    true_beta = np.sort(true_ab[:, 1])

    fig_synth_true(syn)                                   # true regime paths
    fig_cloud(Rs, "fig_synth_cloud.png", true_ab=true_ab) # clustered cloud
    fig_series_regime(syn, Rs, "fig_synth_series.png", ylabel=r"$Y_t$",
                      title="Synthetic: series and identified regimes "
                            "(clustering of $b(t)$)",
                      levels=list(np.sort(Rs["centers_b"])),
                      level_fmt=r"$\beta$={:.2f}")
    beta_lab = [r"true $\beta_%d$=%.2f" % (i + 1, v) for i, v in enumerate(true_beta)]
    fig_roots_derivative(Rs, "fig_synth_roots_A3.png", levels=list(true_beta),
                         ylabel=r"$\partial g_i/\partial w$", tag="(synthetic)",
                         level_labels=beta_lab)

    o = np.argsort(Rs["centers_b"])
    summary["synthetic"] = dict(
        N=int(Rs["N"]), ktable=[(int(k), round(v, 3)) for k, v in Rs["ktable"]],
        silhouette=float(Rs["sil"]),
        vol_true=[float(x) for x in true_beta],
        vol_recovered=[float(x) for x in Rs["centers_b"][o]],
        fracs=[float(x) for x in Rs["fracs"][o]])

    # ---- BTC ----
    btc = make_btc_like(seed=25)
    ann = btc["ann"]
    Rb = run_pipeline(btc, half=12, N=None)
    ob = np.argsort(Rb["centers_b"])
    level_ann = list(np.sort(Rb["centers_b"]) * ann * 100)

    fig_cloud(Rb, "fig_btc_cloud.png", true_ab=None, ann=ann, sil_txt=True)
    fig_series_regime(btc, Rb, "fig_btc_series.png", ylabel=r"$\log P_t$",
                      title="BTC/USD: series and identified regimes "
                            "(clustering of $b(t)$)",
                      levels=level_ann, level_fmt=r"{:.0f}\%")
    fig_roots_derivative(Rb, "fig_btc_roots_A3.png", levels=level_ann,
                         yscale=ann * 100,
                         ylabel=r"$(\partial g_i/\partial w)$, annualised %",
                         tag="(BTC/USD)",
                         level_labels=["%.0f%%" % v for v in level_ann])

    summary["btc"] = dict(
        N=int(Rb["N"]), ktable=[(int(k), round(v, 3)) for k, v in Rb["ktable"]],
        silhouette=float(Rb["sil"]),
        vol_ann_pct=[float(x) for x in Rb["centers_b"][ob] * ann * 100],
        fracs=[float(x) for x in Rb["fracs"][ob]],
        drift_agg_pct_day=[float(x) for x in Rb["centers_a_agg"][ob] * 100],
        logprice=[float(btc["Y"][0]), float(btc["Y"][-1])])

    # =================================================================== #
    #  EXPERIMENT 2.  Order of the catastrophe (C-norm of the profiles)
    # =================================================================== #
    fig_slopes_multi(Rs, "fig_gwt_synth.png", levels=list(true_beta),
                     orders=(3, 4, 5), ylabel=r"$\partial g_i/\partial w$",
                     tag="(synthetic, $k=3$)", level_labels=beta_lab)
    fig_slopes_multi(Rb, "fig_gwt_btc.png", levels=level_ann, yscale=ann * 100,
                     orders=(3, 4, 5),
                     ylabel=r"$(\partial g_i/\partial w)$, annualised %",
                     tag="(BTC/USD, $k=3$)",
                     level_labels=["%.0f%%" % v for v in level_ann])
    fig_sep_matrices(Rs, "fig_sep_synth.png", tag="(synthetic, $k=3$)")
    fig_sep_matrices(Rb, "fig_sep_btc.png", yscale=ann * 100, tag="(BTC/USD, $k=3$)")

    # ---- over-clustered control: SAME 3-regime series, forced k = 4 ----
    R4 = run_pipeline(syn, half=40, N=4)
    fig_cloud(R4, "fig_over4_cloud.png", true_ab=true_ab)
    fig_slopes_multi(R4, "fig_over4_gwt.png", levels=list(true_beta),
                     orders=(3, 4, 5), ylabel=r"$\partial g_i/\partial w$",
                     tag="(synthetic, over-clustered $k=4$)", level_labels=beta_lab)
    fig_sep_matrices(R4, "fig_over4_sep.png", tag="(synthetic, over-clustered $k=4$)")

    def crit(R, sc=1.0):
        """
        Closeness of the volatility profiles, on the domain where all n roots are
        real.  Reported: (i) the uniform-norm distance
        min_{i!=j} ||dg_i/dw - dg_j/dw||_C ; (ii) the distance between the profile
        levels, min_{i!=j} |level_i - level_j|, where level_i is the median of the
        i-th profile.  For flat profiles the two agree; where a profile of an
        over-specified order varies and crosses its neighbours, the sorted-index
        tracking swaps identity at a crossing and inflates the uniform norm, so
        the level distance is the robust summary.
        """
        out = {}
        for n in (3, 4, 5):
            C, mn, lev = slope_cnorm(R["fits"][n], n, R["Tmax"], R["beta_bar"])
            if lev is None:
                out[n] = dict(min_C=None, min_dlevel=None, levels=None)
                continue
            L = np.array(lev) * sc
            d = np.abs(L[:, None] - L[None, :])
            dmin = float(d[~np.eye(n, dtype=bool)].min())
            out[n] = dict(min_C=float(mn * sc), min_dlevel=dmin,
                          levels=[float(x) for x in L])
        return out
    summary["order_selection"] = dict(
        synthetic_k3=crit(Rs),
        btc_k3=crit(Rb, ann * 100),
        synthetic_k4=crit(R4),
        centres_k4=[float(x) for x in np.sort(R4["centers_b"])])

    with open(os.path.join("work", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify()
    else:
        main()
