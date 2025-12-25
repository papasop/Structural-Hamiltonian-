# ============================================================
# Colab 单 cell（完整版·参数漂移噪声版·统计SNR + 可用工作区 + 分布诊断 + 阈值指数检验）
# ============================================================
# 你要的功能全部合并：
#   1) 扩大 s 扫描范围：0.45 ~ 0.4995
#   2) 漂移强度 ds_std 做 log 扫：1e-5, 3e-5, 1e-4, 3e-4（可改）
#   3) 提高 u0_fixed（默认 0.2），让 Eb 到更“好测”的量级
#   4) SNR 定义：多次 trial 的输出能量分布 -> mean/std, CV
#   5) 增益插值：u_at(t) 内对 G_path 线性插值（减少离散误差）
#   6) 漂移扰动能量：对称度量 Epert = E[|E_drift - E_base|]
#   7) 重新定义“可用工作区”：SNRdist>3 且 Rpert<0.5（可改）
#   8) 搜索内部峰：在可用区内选 best s*（默认 J=SNR*mu）
#   9) 分布形状检查：对指定 s_hist（默认 0.4990）画 E_drift 直方图 + median/MAD
#  10) 阈值定律：E_target 设为更大（默认 1e-10），拟合幂律指数（期望 -2）
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.special import gammaln, loggamma

np.random.seed(1234)

# ============================================================
# 0) Veneziano pole-kernel (saturating) and analytic derivative
# ============================================================
def pole_kernel_abs_width(s, t_vow=-0.51, alpha0=0.5, alphaprime=1.0, n=1, delta=1e-3):
    """
    G(s) = |R_n(t)| / sqrt((alpha(s)-n)^2 + delta^2)
    alpha(s) = alpha0 + alphaprime*s
    """
    alpha_s = alpha0 + alphaprime * s
    alpha_t = alpha0 + alphaprime * t_vow

    # residue magnitude; for n=1, |1+alpha_t| is a stable closed-form in your parameterization
    if n == 1:
        Rmag = abs(1.0 + alpha_t)
    else:
        logfac = -gammaln(n + 1)
        z_num = complex(-alpha_t)
        z_den = complex(-n - alpha_t)
        logratio = float(np.real(loggamma(z_num) - loggamma(z_den)))
        logr = float(np.clip(logfac + logratio, -700.0, 700.0))
        Rmag = float(np.exp(logr))

    denom = float(np.sqrt((alpha_s - n)**2 + delta**2))
    return float(Rmag / denom)

def dG_ds_analytic(s, t_vow=-0.51, alpha0=0.5, alphaprime=1.0, n=1, delta=1e-3):
    """
    If G(s)=R/sqrt((a-n)^2+delta^2), a=alpha0+alphaprime*s,
    dG/ds = -R*alphaprime*(a-n)/((a-n)^2+delta^2)^(3/2)
    """
    alpha_s = alpha0 + alphaprime * s
    alpha_t = alpha0 + alphaprime * t_vow
    if n == 1:
        R = abs(1.0 + alpha_t)
    else:
        eps = 1e-7
        return (pole_kernel_abs_width(s+eps, t_vow, alpha0, alphaprime, n, delta)
              - pole_kernel_abs_width(s-eps, t_vow, alpha0, alphaprime, n, delta)) / (2*eps)

    x = (alpha_s - n)
    den = (x*x + delta*delta)**1.5
    return float(-R * alphaprime * x / den)

# ============================================================
# 1) Non-Hermitian corner-gain 2D lattice
# ============================================================
class NonHermitianCornerGain:
    """
    2D L×L lattice:
      H = sum_{<i,j>} Ω (c_i^† c_j + h.c.)
          + i η n_0
          - i γ sum_{j≠0} n_j
    """
    def __init__(self, L=6, Omega=0.02, eta=0.05, gamma=1.0):
        self.L = L
        self.N = L * L
        self.Omega = Omega
        self.eta = eta
        self.gamma = gamma
        self.H = self.build_hamiltonian()

    def site_index(self, x, y):
        return y * self.L + x

    def build_hamiltonian(self):
        L, N = self.L, self.N
        Omega, eta, gamma = self.Omega, self.eta, self.gamma
        H = np.zeros((N, N), dtype=complex)

        # diagonal gain/loss
        for y in range(L):
            for x in range(L):
                idx = self.site_index(x, y)
                if x == 0 and y == 0:
                    H[idx, idx] += 1j * eta
                else:
                    H[idx, idx] += -1j * gamma

        # reciprocal hopping
        for y in range(L):
            for x in range(L):
                i = self.site_index(x, y)
                if x < L - 1:
                    j = self.site_index(x + 1, y)
                    H[i, j] += Omega
                    H[j, i] += Omega
                if y < L - 1:
                    j = self.site_index(x, y + 1)
                    H[i, j] += Omega
                    H[j, i] += Omega
        return H

# ============================================================
# 2) Drive template u_sig(t): windowed sinusoid
# ============================================================
def smoothstep(x):
    return 0.5*(1 + np.tanh(x))

def drive_env(t, T_on, T_off, ramp):
    return smoothstep((t - T_on)/ramp) - smoothstep((t - T_off)/ramp)

def drive_sig(t, u0, omega, T_on, T_off, ramp):
    return u0 * drive_env(t, T_on, T_off, ramp) * np.sin(omega*t)

def compute_Ein(u0, omega, T, dt, T_on, T_off, ramp):
    ts = np.arange(0.0, T + 1e-12, dt)
    us = np.array([drive_sig(t, u0, omega, T_on, T_off, ramp) for t in ts])
    return float(np.sum(np.abs(us)**2) * dt)

# ============================================================
# 3) OU parameter drift: s(t)=s0 + OU noise
# ============================================================
def generate_s_ou_path(s0, dt, T, ds_std=2e-4, tau=2.0, seed=0, s_min=0.0, s_max=0.4999995):
    rng = np.random.default_rng(seed)
    n = int(T/dt) + 1
    s = np.empty(n, dtype=float)
    s[0] = s0

    # OU: dx = -(1/tau)*x dt + sigma dW; stationary std = ds_std
    sig = np.sqrt(2.0*(ds_std**2) / max(tau, 1e-12))
    for k in range(1, n):
        x = s[k-1] - s0
        dW = rng.normal(0.0, np.sqrt(dt))
        x_new = x + (-(1.0/tau)*x)*dt + sig*dW
        s[k] = float(np.clip(s0 + x_new, s_min, s_max))
    return s

# ============================================================
# 4) Forced evolution (RK4): dψ/dt = -i( H ψ + u(t) f )
# ============================================================
def rk4_step_forced(psi, t, dt, H, f, u_at):
    def rhs(psi_, t_):
        u = u_at(t_)
        return (-1j) * (H @ psi_ + u * f)
    k1 = rhs(psi, t)
    k2 = rhs(psi + 0.5*dt*k1, t + 0.5*dt)
    k3 = rhs(psi + 0.5*dt*k2, t + 0.5*dt)
    k4 = rhs(psi + dt*k3, t + dt)
    return psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def evolve_forced(H, psi0, f, u_at, T=60.0, dt=0.02):
    steps = int(T/dt)
    psi = psi0.astype(complex).copy()
    for k in range(steps):
        t = k*dt
        psi = rk4_step_forced(psi, t, dt, H, f, u_at)
    return psi

# ============================================================
# 5) Injection vector at center & outputs
# ============================================================
def build_injection_vector(L):
    N = L*L
    cx, cy = L//2, L//2
    inj_index = cy*L + cx
    f = np.zeros(N, dtype=complex)
    f[inj_index] = 1.0
    return f, inj_index

def corner_layer_indices(L, layer_d=0):
    idxs = []
    for y in range(L):
        for x in range(L):
            if x + y <= layer_d:
                idxs.append(y*L + x)
    return np.array(idxs, dtype=int)

# ============================================================
# SETTINGS (你可在这里改核心参数)
# ============================================================
# Lattice
L = 6
Omega = 0.02
gamma = 1.0
eta0 = 0.05

# Kernel
kernel_kwargs = dict(t_vow=-0.51, alpha0=0.5, alphaprime=1.0, n=1, delta=1e-3)

# Drive
omega = 1.0
T = 60.0
dt = 0.02
T_on, T_off, ramp = 5.0, 55.0, 1.0

# Measurement
layer_d = 0
layer_idx = corner_layer_indices(L, layer_d=layer_d)

# Scan ranges
s_list = np.linspace(0.45, 0.4995, 25)  # 扩大扫描范围
ds_std_list = [1e-5, 3e-5, 1e-4, 3e-4]  # 漂移强度 log扫（可改）
tau_corr = 2.0
ntrials = 30

# Fixed drive amplitude for drift statistics (提高到更“好测”)
u0_fixed = 0.2

# Usable region definition
SNR_min = 3.0
Rpert_max = 0.5

# Score to pick best point inside usable region
# (推荐：J = SNRdist * mu ；你也可以换成 J=SNRdist 或 J=mu/(CV+eps))
def score_func(mu, sd, snr):
    return snr * mu

# Histogram diagnostic target
s_hist = 0.4990

# Threshold law (absolute target energy at corner)
E_target = 1e-10

# Threshold search brackets
u0_lo = 1e-6
u0_hi = 0.5

# ============================================================
# Core runner
# ============================================================
def run_one(s0, u0, ds_std=0.0, tau_corr=2.0, seed=0):
    model = NonHermitianCornerGain(L=L, Omega=Omega, eta=eta0, gamma=gamma)
    H = model.H

    f, inj_index = build_injection_vector(L)
    psi0 = np.zeros(L*L, dtype=complex)

    t_grid = np.arange(0.0, T + 1e-12, dt)

    if ds_std > 0:
        s_path = generate_s_ou_path(s0, dt, T, ds_std=ds_std, tau=tau_corr, seed=seed, s_max=float(s_list.max()))
    else:
        s_path = np.full_like(t_grid, float(s0))

    G_path = np.array([pole_kernel_abs_width(s, **kernel_kwargs) for s in s_path], dtype=float)

    # linear interpolation of G_path in time
    def u_at(t):
        Gt = float(np.interp(t, t_grid, G_path))
        return Gt * drive_sig(t, u0, omega, T_on, T_off, ramp)

    psiT = evolve_forced(H, psi0, f, u_at, T=T, dt=dt)

    Etot = float(np.sum(np.abs(psiT)**2))
    Elayer = float(np.sum(np.abs(psiT[layer_idx])**2))
    Ecorner = float(np.abs(psiT[0])**2)
    focus = float(Elayer / (Etot + 1e-300))

    Ein = compute_Ein(u0, omega, T, dt, T_on, T_off, ramp)
    relGjit = float(np.std(G_path) / (np.mean(G_path) + 1e-300))

    return dict(
        s0=float(s0),
        G0=float(pole_kernel_abs_width(s0, **kernel_kwargs)),
        dGds=float(dG_ds_analytic(s0, **kernel_kwargs)),
        u0=float(u0),
        Ein=float(Ein),
        Etot=float(Etot),
        Elayer=float(Elayer),
        Ecorner=float(Ecorner),
        focus=float(focus),
        relGjit=float(relGjit),
    )

def find_threshold_u0(s0, E_target, u0_lo=1e-6, u0_hi=0.5, max_iter=28, seed=0):
    """
    Find u0_th such that Ecorner(T) >= E_target (no drift).
    Uses geometric/binary search in amplitude u0.
    """
    r_hi = run_one(s0, u0_hi, ds_std=0.0, seed=seed)
    if r_hi["Ecorner"] < E_target:
        u = u0_hi
        for _ in range(12):
            u *= 2.0
            r_hi = run_one(s0, u, ds_std=0.0, seed=seed)
            if r_hi["Ecorner"] >= E_target:
                u0_hi = u
                break
        else:
            return dict(success=False, s0=float(s0), u0_th=np.nan, Ein_th=np.nan, G=r_hi["G0"], achieved=r_hi["Ecorner"])

    r_lo = run_one(s0, u0_lo, ds_std=0.0, seed=seed)
    if r_lo["Ecorner"] >= E_target:
        Ein_lo = compute_Ein(u0_lo, omega, T, dt, T_on, T_off, ramp)
        return dict(success=True, s0=float(s0), u0_th=float(u0_lo), Ein_th=float(Ein_lo), G=r_lo["G0"], achieved=r_lo["Ecorner"])

    lo, hi = u0_lo, u0_hi
    for _ in range(max_iter):
        mid = np.sqrt(lo*hi)
        r_mid = run_one(s0, mid, ds_std=0.0, seed=seed)
        if r_mid["Ecorner"] >= E_target:
            hi = mid
        else:
            lo = mid

    u0_th = hi
    Ein_th = compute_Ein(u0_th, omega, T, dt, T_on, T_off, ramp)
    r_final = run_one(s0, u0_th, ds_std=0.0, seed=seed)
    return dict(success=True, s0=float(s0), u0_th=float(u0_th), Ein_th=float(Ein_th), G=r_final["G0"], achieved=r_final["Ecorner"])

# ============================================================
# EXPERIMENT 1: THRESHOLD LAW (no drift)
# ============================================================
print("="*120)
print("EXPERIMENT 1: THRESHOLD LAW (no drift, absolute target)")
print("="*120)
print("Definition: u0_th(s) = min u0 such that E_corner(T) >= E_target")
print(f"E_target (absolute) = {E_target:.3e}")
print("-"*120)

th = []
for s0 in s_list:
    row = find_threshold_u0(s0, E_target, u0_lo=u0_lo, u0_hi=u0_hi, max_iter=28, seed=0)
    th.append(row)
    if row["success"]:
        print(f"s={row['s0']:.6f}  G={row['G']:.3e}  u0_th={row['u0_th']:.3e}  Ein_th={row['Ein_th']:.3e}  achieved={row['achieved']:.3e}")
    else:
        print(f"s={row['s0']:.6f}  FAILED (achieved {row['achieved']:.3e})")

Gs = np.array([r["G"] for r in th if r["success"] and np.isfinite(r["Ein_th"]) and r["Ein_th"]>0], dtype=float)
Ein_ths = np.array([r["Ein_th"] for r in th if r["success"] and np.isfinite(r["Ein_th"]) and r["Ein_th"]>0], dtype=float)

slope_th = np.nan
coef = None
if len(Gs) >= 3:
    coef = np.polyfit(np.log10(Gs), np.log10(Ein_ths), 1)
    slope_th = coef[0]
    print("\n[THRESHOLD FIT]")
    print(f"log10(Ein_th) = a + p*log10(G),   p ≈ {slope_th:.3f}   (expect ~ -2)")

plt.figure(figsize=(7.5,5))
plt.loglog(Gs, Ein_ths, "o-", label="measured")
if coef is not None:
    xref = np.array([Gs.min(), Gs.max()])
    yref = 10**(coef[1]) * xref**(slope_th)
    plt.loglog(xref, yref, "k--", label=f"fit slope p≈{slope_th:.2f} (expect -2)")
plt.xlabel("G(s)")
plt.ylabel("Ein_th = ∫|u|^2 dt at threshold")
plt.title("Threshold law: Ein_th vs G(s) (absolute E_target)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# EXPERIMENT 2: PARAMETER-DRIFT OUTPUT DISTRIBUTION (SNR by mean/std)
#   - Sweep ds_std
#   - Compute usable region
#   - Find best internal peak
#   - Histogram at s_hist
# ============================================================
print("="*120)
print("EXPERIMENT 2: PARAMETER-DRIFT OUTPUT DISTRIBUTION (SNR by mean/std) + USABLE REGION + PEAK SEARCH")
print("="*120)
print(f"u0_fixed={u0_fixed:.3e} | T={T}, dt={dt} | OU tau_corr={tau_corr:.2f} | ntrials={ntrials}")
print(f"Usable criteria: SNRdist > {SNR_min} AND Rpert < {Rpert_max}")
print(f"Histogram diagnostic at s_hist={s_hist:.6f}")
print("-"*120)

for ds_std in ds_std_list:
    print("\n" + "="*120)
    print(f"DRIFT LEVEL: ds_std = {ds_std:.2e}")
    print("="*120)
    print("   s0      G0       |dG/ds|      Eb(base)     mu          std         CV         SNRdist     Epert       Rpert      relGjit")

    rows = []
    Ed_hist = None
    Eb_hist = None

    for s0 in s_list:
        base = run_one(s0, u0_fixed, ds_std=0.0, seed=0)
        Eb = base["Ecorner"]

        Ed = []
        relG = []
        for k in range(ntrials):
            r = run_one(s0, u0_fixed, ds_std=ds_std, tau_corr=tau_corr, seed=1000+k)
            Ed.append(r["Ecorner"])
            relG.append(r["relGjit"])
        Ed = np.array(Ed, dtype=float)

        mu = float(np.mean(Ed))
        sd = float(np.std(Ed, ddof=1)) if len(Ed) > 1 else 0.0
        cv = float(sd / (mu + 1e-300))
        snr_dist = float(mu / (sd + 1e-300))

        Epert = float(np.mean(np.abs(Ed - Eb)))
        Rpert = float(Epert / (Eb + 1e-300))
        relGjit = float(np.mean(relG))

        G0 = float(pole_kernel_abs_width(s0, **kernel_kwargs))
        dG = float(abs(dG_ds_analytic(s0, **kernel_kwargs)))

        rows.append(dict(s0=float(s0), G0=G0, dG=dG, Eb=Eb, mu=mu, sd=sd, cv=cv, snr=snr_dist,
                         Epert=Epert, Rpert=Rpert, relGjit=relGjit))

        print(f"{s0:0.6f}  {G0:0.3e}  {dG:0.3e}  {Eb:0.3e}  {mu:0.3e}  {sd:0.3e}  {cv:0.3e}  {snr_dist:0.3e}  {Epert:0.3e}  {Rpert:0.3e}  {relGjit:0.3e}")

        if abs(s0 - s_hist) < 1e-9:
            Ed_hist = Ed.copy()
            Eb_hist = Eb

    # ---- Usable region + best point
    usable = [(r["snr"] > SNR_min) and (r["Rpert"] < Rpert_max) for r in rows]
    idxs = [i for i,u in enumerate(usable) if u]

    print("\n[USABLE REGION RESULT]")
    if not idxs:
        print("  none under this ds_std. Try smaller ds_std or adjust u0_fixed / eta0.")
    else:
        s_usable = [rows[i]["s0"] for i in idxs]
        print(f"  usable s-range: [{min(s_usable):.6f}, {max(s_usable):.6f}]  (count={len(s_usable)})")

        J = [score_func(rows[i]["mu"], rows[i]["sd"], rows[i]["snr"]) for i in idxs]
        i_best = idxs[int(np.argmax(J))]
        rb = rows[i_best]
        print(f"  best s*={rb['s0']:.6f} | mu={rb['mu']:.3e} std={rb['sd']:.3e} CV={rb['cv']:.3f} "
              f"SNR={rb['snr']:.3f} Rpert={rb['Rpert']:.3f} | G0={rb['G0']:.2e} |dG/ds|={rb['dG']:.2e}")

    # ---- Plots: SNR, CV, Rpert, mu
    svals = np.array([r["s0"] for r in rows], dtype=float)
    muvals = np.array([r["mu"] for r in rows], dtype=float)
    sdvals = np.array([r["sd"] for r in rows], dtype=float)
    snrvals = np.array([r["snr"] for r in rows], dtype=float)
    cvvals  = np.array([r["cv"] for r in rows], dtype=float)
    rpvals  = np.array([r["Rpert"] for r in rows], dtype=float)
    dGvals  = np.array([r["dG"] for r in rows], dtype=float)

    plt.figure(figsize=(7.5,5))
    plt.semilogy(svals, np.maximum(snrvals, 1e-300), "o-")
    plt.axhline(SNR_min, ls="--", color="k", alpha=0.6, label=f"SNR_min={SNR_min}")
    plt.xlabel("s0")
    plt.ylabel("SNRdist = mean/std (log)")
    plt.title(f"SNRdist vs s0 (ds_std={ds_std:.1e})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.5,5))
    plt.semilogy(svals, np.maximum(cvvals, 1e-300), "o-")
    plt.axhline(1.0, ls="--", color="k", alpha=0.6, label="CV=1")
    plt.xlabel("s0")
    plt.ylabel("CV = std/mean (log)")
    plt.title(f"Relative jitter (CV) vs s0 (ds_std={ds_std:.1e})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.5,5))
    plt.semilogy(svals, np.maximum(rpvals, 1e-300), "o-")
    plt.axhline(Rpert_max, ls="--", color="k", alpha=0.6, label=f"Rpert_max={Rpert_max}")
    plt.xlabel("s0")
    plt.ylabel("Rpert = E|E-Eb| / Eb (log)")
    plt.title(f"Symmetric perturbation ratio vs s0 (ds_std={ds_std:.1e})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.5,5))
    plt.semilogy(svals, np.maximum(muvals, 1e-300), "o-", label="mu(E_drift)")
    plt.semilogy(svals, np.maximum(sdvals, 1e-300), "o--", label="std(E_drift)")
    plt.xlabel("s0")
    plt.ylabel("Energy (log)")
    plt.title(f"Output distribution scale vs s0 (ds_std={ds_std:.1e})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.5,5))
    plt.semilogy(svals, np.maximum(dGvals, 1e-300), "o-")
    plt.xlabel("s0")
    plt.ylabel("|dG/ds| (log)")
    plt.title(f"Analytic |dG/ds| vs s0 (ds_std={ds_std:.1e})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- Histogram / distribution diagnostics at s_hist
    if Ed_hist is not None:
        plt.figure(figsize=(6.5,4.5))
        plt.hist(Ed_hist, bins=20)
        plt.xlabel("E_corner(T) under drift")
        plt.ylabel("count")
        plt.title(f"E_drift distribution at s0={s_hist:.4f} (ds_std={ds_std:.1e})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        med = float(np.median(Ed_hist))
        mad = float(np.median(np.abs(Ed_hist - med)))
        q05, q50, q95 = np.quantile(Ed_hist, [0.05, 0.50, 0.95])
        print("\n[DISTRIBUTION DIAGNOSTICS at s_hist]")
        print(f"  Eb(base)   = {Eb_hist:.3e}")
        print(f"  mean       = {np.mean(Ed_hist):.3e}")
        print(f"  std        = {np.std(Ed_hist, ddof=1):.3e}")
        print(f"  median     = {med:.3e}")
        print(f"  MAD        = {mad:.3e}")
        print(f"  quantiles  = q05 {q05:.3e} | q50 {q50:.3e} | q95 {q95:.3e}")
        print("  note: if (q95/q50) is huge or mean >> median, likely heavy-tail near pole.")

print("\nDONE.")
