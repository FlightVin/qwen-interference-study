"""Figures for 00-initial-theory.md. Toy models and reported numbers only; no project model or data.

Run: python build/theory_figs.py   (from reports/<ID>/; needs numpy and matplotlib). Writes figures/t*.png and
tables/theory_numbers.json (every number that the document quotes from these simulations).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5, "legend.frameon": False})
NUM = {}

# ---------------------------------------------------------------- ray interference, 2 x 2 bandit
V = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])          # v_1, v_2 (Schaul et al. 2019, Section 2.1)
sig = lambda u: 1 / (1 + np.exp(-u))


def f_rl(u): return u * (1 - u)
def f_sup(u): return 1 - u
def f_mix(u, d=0.1): return ((1 - d) * u + d / 2) * (1 - u)


def Jdot(J1, J2, f):
    """dJ_k/dt under theta' = sum_k f(J_k) v_k, with grad J_k = J_k (1 - J_k) v_k."""
    return J1 * (1 - J1) * (2 * f(J1) - f(J2)), J2 * (1 - J2) * (2 * f(J2) - f(J1))


def integrate(J0, f, dt=0.02, T=4000.0, stop=1.9):
    th = np.array([np.log(J0[0] / (1 - J0[0])), np.log(J0[1] / (1 - J0[1])), 0.0])
    out, t = [], 0.0
    while t < T:
        J = np.array([sig(th[0] + th[2]), sig(th[1] - th[2])])
        out.append((t, J[0], J[1]))
        if J.sum() >= stop:
            break
        th = th + dt * (f(J[0]) * V[0] + f(J[1]) * V[1])
        t += dt
    return np.array(out)


fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.5))
g = np.linspace(0.02, 0.98, 22)
G1, G2 = np.meshgrid(g, g)
starts = [(0.02, 0.18), (0.05, 0.15), (0.08, 0.12), (0.1, 0.1), (0.12, 0.08), (0.15, 0.05), (0.18, 0.02)]
for ax, (f, title) in zip(axs, [(f_rl, "On-policy REINFORCE\n(coupled: f(u) = u(1-u))"),
                               (f_sup, "Supervised learning\n(not coupled: f(u) = 1-u)"),
                               (f_mix, "REINFORCE, 10% uniform actions\n(f(0) > 0)")]):
    U, W = Jdot(G1, G2, f)
    n = np.hypot(U, W) + 1e-12
    ax.quiver(G1, G2, U / n, W / n, np.log10(n), cmap="Greys", alpha=0.75, scale=38, width=0.003)
    for i, s in enumerate(starts):
        tr = integrate(s, f)
        ax.plot(tr[:, 1], tr[:, 2], color=C[0] if i != 3 else C[1], lw=1.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("J1 (success on context 1)")
axs[0].set_ylabel("J2 (success on context 2)")
fig.suptitle("Flow of the two-context bandit in the (J1, J2) plane. All starts have J1 + J2 = 0.2; orange = balanced start.",
             fontsize=9.5, y=1.02)
fig.savefig(os.path.join(FIG, "t1_ray_flow.png")); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1))
imb = [0.5, 0.4, 0.25, 0.1, 0.05]                          # share of J1 in the initial total 0.2
times = {"rl": [], "sup": [], "mix": []}
for name, f, ax in (("rl", f_rl, axs[0]), ("sup", f_sup, axs[1])):
    for i, sh in enumerate(imb):
        tr = integrate((0.2 * sh, 0.2 * (1 - sh)), f)
        ax.plot(tr[:, 0], tr[:, 1] + tr[:, 2], color=plt.cm.Blues(0.95 - 0.16 * i), lw=1.6,
                label=f"J1(0) : J2(0) = {sh / (1 - sh):.2f}")
    ax.set_xscale("log"); ax.set_xlabel("time (gradient-flow units)"); ax.set_ylim(0, 2.02)
axs[0].set_title("On-policy REINFORCE: J1 + J2"); axs[1].set_title("Supervised: J1 + J2")
axs[0].set_ylabel("total performance J"); axs[0].legend(fontsize=7, loc="upper left")
shares = np.linspace(0.02, 0.5, 49)
for name, f in (("rl", f_rl), ("sup", f_sup), ("mix", f_mix)):
    times[name] = [integrate((0.2 * s, 0.2 * (1 - s)), f)[-1, 0] for s in shares]
axs[2].plot(shares / (1 - shares), times["rl"], color=C[0], lw=1.8, label="on-policy REINFORCE")
axs[2].plot(shares / (1 - shares), times["mix"], color=C[2], lw=1.8, label="10% uniform actions")
axs[2].plot(shares / (1 - shares), times["sup"], color=C[1], lw=1.8, label="supervised")
axs[2].set_yscale("log"); axs[2].set_xlabel("initial ratio J1(0) / J2(0)"); axs[2].set_ylabel("time to reach J = 1.9")
axs[2].set_title("Slowdown from an imbalanced start"); axs[2].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "t2_ray_curves.png")); plt.close(fig)
NUM["ray_time_to_1.9"] = {name: {"ratio_1.00": float(times[name][-1]), "ratio_0.33": float(np.interp(0.25, shares, times[name])),
                                 "ratio_0.11": float(np.interp(0.1, shares, times[name])),
                                 "ratio_0.02": float(times[name][0])} for name in times}
# minimum speed along the path (flatness) for the on-policy case
NUM["ray_min_speed"] = {}
for sh in (0.5, 0.25, 0.1, 0.05):
    tr = integrate((0.2 * sh, 0.2 * (1 - sh)), f_rl)
    Jt = tr[:, 1] + tr[:, 2]
    sp = np.gradient(Jt, tr[:, 0])
    mid = (Jt > 0.3) & (Jt < 1.8)
    NUM["ray_min_speed"][f"share_{sh}"] = float(sp[mid].min())

# ---------------------------------------------------------------- K = 8 contexts


def bandit_K(K=8, shared_bias=True, steps=4000000, dt=0.5, on_policy=True):
    J0 = np.logspace(np.log10(0.02), np.log10(0.3), K)          # unequal initial success per context
    W = np.zeros((K, K)); b = np.zeros(K)
    W[np.arange(K), np.arange(K)] = np.log(J0 * (K - 1) / (1 - J0))
    out = []
    for t in range(steps):
        L = W + b[:, None]
        P = np.exp(L - L.max(0)); P /= P.sum(0)              # column k = policy in context k
        Jk = np.diag(P).copy()
        if t % 50 == 0 or t < 50:
            out.append((t * dt, *Jk))
        if Jk.sum() > K - 0.1:
            break
        w = Jk if on_policy else np.ones(K)                   # on-policy: weight J_k; supervised: weight 1
        Gl = -P * w[None, :]
        Gl[np.arange(K), np.arange(K)] += w                   # d/d logits of J_k (on-policy) or log J_k (supervised)
        W += dt * Gl
        if shared_bias:
            b += dt * Gl.sum(1)
    return np.array(out)


fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
for ax, (kw, title) in zip(axs, [(dict(shared_bias=True), "On-policy, shared action bias"),
                                (dict(shared_bias=False), "On-policy, tabular (no shared parameter)"),
                                (dict(shared_bias=True, on_policy=False), "Supervised, shared action bias")]):
    tr = bandit_K(**kw)
    for k in range(8):
        ax.plot(tr[:, 0] + 1, tr[:, 1 + k], color=MUTED, lw=0.7, alpha=0.8)
    ax.plot(tr[:, 0] + 1, tr[:, 1:].sum(1) / 8, color=C[0], lw=2.0, label="mean over 8 contexts")
    ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("time (log scale)")
    NUM.setdefault("K8_time_to_done", {})[title] = float(tr[-1, 0])
axs[0].set_ylabel("success per context (grey), mean (blue)"); axs[0].legend(fontsize=7, loc="upper left")
fig.savefig(os.path.join(FIG, "t3_ray_K8.png")); plt.close(fig)

# ---------------------------------------------------------------- finite step: first and second order
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
eta = np.linspace(0, 2, 200)
for i, cosv in enumerate([0.5, 0.1, 0.0, -0.1, -0.5]):
    # unit gradients, curvature of task i along g_j: h = g_j^T H_i g_j = 1
    axs[0].plot(eta, -eta * cosv + 0.5 * eta ** 2, color=plt.cm.RdBu(0.9 - 0.2 * i), lw=1.7, label=f"cos = {cosv:+.1f}")
axs[0].axhline(0, color=INK, lw=0.7)
axs[0].set_xlabel("step size x curvature  (eta * h, with |g_i| = |g_j| = 1)"); axs[0].set_ylabel("change in loss of task i")
axs[0].set_title("A step on task j: first-order gain, second-order cost"); axs[0].legend(fontsize=7); axs[0].set_ylim(-0.3, 1.2)
cs = np.linspace(-1, 1, 200)
axs[1].fill_between(cs, 0, np.clip(2 * cs, 0, None), color=C[2], alpha=0.25, label="task i improves")
axs[1].fill_between(cs, np.clip(2 * cs, 0, None), 3, color=C[1], alpha=0.25, label="task i gets worse")
axs[1].plot(cs, np.clip(2 * cs, 0, None), color=INK, lw=1.2)
axs[1].set_ylim(0, 3); axs[1].set_xlabel("cosine between the two task gradients"); axs[1].set_ylabel("eta * h * |g_j| / |g_i|")
axs[1].set_title("Boundary: eta * h * |g_j| / |g_i| = 2 cos"); axs[1].legend(fontsize=7, loc="upper left")
fig.savefig(os.path.join(FIG, "t4_step.png")); plt.close(fig)

# ---------------------------------------------------------------- near-orthogonality and shared subspaces
rng = np.random.default_rng(1)
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
ds = [10, 100, 1000, 10000, 100000]
sd = []
for d in ds:
    a = rng.normal(size=(400, d)); b = rng.normal(size=(400, d))
    sd.append(float(np.std((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))))
axs[0].loglog(ds, sd, "o", color=C[0], ms=6, label="simulated (400 pairs)")
dd = np.logspace(1, 9.3, 50)
axs[0].loglog(dd, dd ** -0.5, color=MUTED, lw=1.2, label="1 / sqrt(d)")
axs[0].axvline(1.72e9, color=C[1], lw=1, ls="--"); axs[0].text(1.2e9, 0.1, "Qwen3-1.7B\nparameters", ha="right", fontsize=7)
axs[0].set_xlabel("dimension d"); axs[0].set_ylabel("std of cosine, independent vectors"); axs[0].legend(fontsize=7)
axs[0].set_title("Null model: independent directions")
NUM["cos_std_sim"] = dict(zip(map(str, ds), sd)); NUM["cos_std_null_qwen"] = float(1.72e9 ** -0.5)
# two task gradients = shared component (weight s) + independent components, in d = 10,000
d = 10000
ss = np.linspace(0, 1, 21)
sh = rng.normal(size=d); sh /= np.linalg.norm(sh)
means = []
for s in ss:
    a = rng.normal(size=(300, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(300, d)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    ga = np.sqrt(s) * sh + np.sqrt(1 - s) * a; gb = np.sqrt(s) * sh + np.sqrt(1 - s) * b
    means.append(float(np.mean((ga * gb).sum(1) / np.linalg.norm(ga, axis=1) / np.linalg.norm(gb, axis=1))))
axs[1].plot(ss, means, "o-", color=C[0], ms=4, lw=1.4, label="simulated")
axs[1].plot(ss, ss, color=MUTED, lw=1, ls="--", label="cos = s")
axs[1].set_xlabel("share s of squared gradient norm in a common direction"); axs[1].set_ylabel("mean cosine between the two tasks")
axs[1].set_title("A shared component sets the cosine"); axs[1].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "t5_orthogonality.png")); plt.close(fig)

# ---------------------------------------------------------------- forgetting in over-parameterised linear regression
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.0))
th = np.linspace(0, np.pi / 2, 91)
sim = []
for t in th:                                                  # rank-one tasks in R^50, worst case over unit w*
    x1 = np.zeros(50); x1[0] = 1
    x2 = np.zeros(50); x2[0] = np.cos(t); x2[1] = np.sin(t)
    P1, P2 = np.outer(x1, x1), np.outer(x2, x2)
    M = x1 @ (np.eye(50) - P2) @ (np.eye(50) - P1)            # task-1 residual after fitting 1 then 2 = M w*
    sim.append(float(M @ M))
axs[0].plot(np.degrees(th), sim, color=C[0], lw=1.8, label="simulated, worst case")
axs[0].plot(np.degrees(th), np.cos(th) ** 2 * np.sin(th) ** 2, color=INK, lw=0.8, ls="--", label="cos^2 sin^2")
axs[0].set_xlabel("angle between the two task inputs (degrees)"); axs[0].set_ylabel("forgetting of task 1")
axs[0].set_title("Two rank-one tasks (after Evron et al. 2022)"); axs[0].legend(fontsize=7)
NUM["evron_max_err"] = float(np.max(np.abs(np.array(sim) - np.cos(th) ** 2 * np.sin(th) ** 2)))
al = np.linspace(0, 1, 101)


def gold(a, b):
    return a * (2 + b * (a ** 3 - 6 * a ** 2 + 11 * a - 8) + b ** 2 * (-5 * a ** 3 + 22 * a ** 2 - 30 * a + 12)
                + b ** 3 * (5 * a ** 3 - 18 * a ** 2 + 20 * a - 6))


for i, b in enumerate([0.0, 0.5, 0.8, 0.95, 1.0]):
    axs[1].plot(al, gold(al, b), color=plt.cm.Blues(0.35 + 0.15 * i), lw=1.7, label=f"beta = {b:g}")
axs[1].set_yscale("symlog", linthresh=0.05); axs[1].set_xlabel("dissimilarity alpha (share of rotated dimensions)")
axs[1].set_ylabel("normalised forgetting"); axs[1].legend(fontsize=7, title="over-parameterisation", title_fontsize=7)
axs[1].set_title("Similarity x over-parameterisation\n(formula of Goldfarb et al. 2024)")
NUM["goldfarb_beta1_max"] = [float(al[np.argmax(gold(al, 1.0))]), float(gold(al, 1.0).max())]
NUM["goldfarb_check_beta1"] = float(np.max(np.abs(gold(al, 1.0) - al ** 2 * (1 - al) ** 2)))
NUM["goldfarb_check_beta0"] = float(np.max(np.abs(gold(al, 0.0) - 2 * al)))
pn = np.logspace(0.05, 3, 100)                                # p / n
r = 1 - 1 / pn
for i, dist in enumerate([0.0, 0.5, 1.0, 2.0]):               # ||w2* - w1*||^2 with ||w1*|| = 1; Lin et al. 2023, T = 2
    axs[2].plot(pn, (r ** 2 - r) + (1 - r) * dist, color=plt.cm.Oranges(0.35 + 0.2 * i), lw=1.7,
                label=f"|w2* - w1*|^2 = {dist:g}")
axs[2].axhline(0, color=INK, lw=0.6)
axs[2].set_xscale("log"); axs[2].set_xlabel("parameters per training sample, p / n"); axs[2].set_ylabel("expected forgetting of task 1")
axs[2].set_title("Two tasks, noiseless\n(formula of Lin et al. 2023)"); axs[2].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "t6_linear_forgetting.png")); plt.close(fig)

# ---------------------------------------------------------------- capacity: shared linear bottleneck
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
rng = np.random.default_rng(3)
d, n = 64, 2000
X = rng.normal(size=(n, d))
for i, (K, rho) in enumerate([(8, 0.0), (8, 0.8), (32, 0.0), (32, 0.8)]):
    common = rng.normal(size=(d, 1))
    Th = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(size=(d, K))      # task directions, pairwise cosine ~ rho
    Y = X @ Th
    s = np.linalg.svd(Y, compute_uv=False) ** 2
    qs = np.arange(0, K + 1)
    exc = [s[q:].sum() / s.sum() for q in qs]                 # best rank-q shared representation (Eckart-Young)
    axs[0].plot(qs / K, exc, "o-", ms=3, lw=1.4, color=C[i], label=f"K = {K} tasks, task cosine ~ {rho:g}")
    NUM.setdefault("capacity_excess", {})[f"K{K}_rho{rho}"] = {"q=K/4": float(exc[K // 4]), "q=K/2": float(exc[K // 2])}
axs[0].set_xlabel("shared width q / number of tasks K"); axs[0].set_ylabel("unexplained share of all task targets")
axs[0].set_title("Linear multi-task model, shared bottleneck"); axs[0].legend(fontsize=7)
# toy superposition count: how many random unit vectors fit in m dims with |cos| < eps
ms = np.array([64, 256, 1024, 4096])
for i, eps in enumerate([0.1, 0.2]):
    axs[1].semilogy(ms, np.exp(ms * eps ** 2 / 4), "o-", color=C[i], lw=1.4, ms=4, label=f"|cos| < {eps:g}")
axs[1].set_xscale("log"); axs[1].set_xlabel("dimensions m"); axs[1].set_ylabel("number of random directions, order of magnitude")
axs[1].set_title("Almost-orthogonal directions grow as exp(m eps^2 / 4)\n(union bound for random unit vectors)")
axs[1].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "t7_capacity.png")); plt.close(fig)

# ---------------------------------------------------------------- reported numbers from the literature
fig, axs = plt.subplots(2, 3, figsize=(10.6, 5.6))
ax = axs[0, 0]
lab = ["hard", "hard +\n256 easy", "hard +\n1K easy", "hard +\nguide", "hard + guide\n+ 1K easy"]
ax.bar(lab, [13.55, 8.22, 2.24, 15.50, 13.98], color=[C[0], C[1], C[1], C[2], C[2]], width=0.6)
ax.axhline(0.57, color=INK, lw=0.8, ls="--"); ax.text(4.4, 1.0, "base 0.57", ha="right", fontsize=7)
ax.set_ylabel("hard-set pass@1 (%)"); ax.set_title("POPE (Qu et al. 2026), Table 2\nQwen3-4B-Instruct, GRPO"); ax.tick_params(axis="x", labelsize=6.5)
ax = axs[0, 1]
x = np.arange(3); w = 0.26
ax.bar(x - w, [56.40, 67.46, 9.07], w, color=MUTED, label="base")
ax.bar(x, [np.nan, 60.60, 24.08], w, color=C[0], label="after math RL")
ax.bar(x + w, [50.80, np.nan, np.nan], w, color=C[1], label="after code RL")
ax.set_xticks(x); ax.set_xticklabels(["MATH500", "code", "puzzle"]); ax.set_ylabel("accuracy (%)")
ax.set_title("Li et al. 2025 (arXiv:2507.17512)\nQwen2.5-7B base, GRPO, cross-domain"); ax.legend(fontsize=7)
ax = axs[0, 2]
ax.plot([1, 8, 24, 64], [0.85, 0.90, 0.94, 0.97], "o-", color=C[0], lw=1.6)
ax.set_xscale("log"); ax.set_xticks([1, 8, 24, 64]); ax.set_xticklabels(["1B", "8B", "24B", "64B"])
ax.set_ylabel("normalised score after merging 2 experts"); ax.set_title("Yadav et al. 2024 (arXiv:2410.03617)\nPaLM-2-IT, weight merging")
ax = axs[1, 0]
ax.bar(["width 32", "width 2048"], [36.9, 26.7], color=C[0], width=0.5, label="Rotated MNIST, MLP")
ax.bar(["WRN-10 1x", "WRN-10 8x"], [37.3, 29.4], color=C[2], width=0.5, label="Split CIFAR-100")
ax.set_ylabel("average forgetting (points)"); ax.set_title("Mirzadeh et al. 2022: width, from scratch"); ax.legend(fontsize=7)
ax.tick_params(axis="x", labelsize=7)
ax = axs[1, 1]
ax.bar(["from scratch", "pre-trained"], [0.907, 0.466], color=[C[1], C[0]], width=0.5)
ax.set_ylabel("class-representation overlap S_AB"); ax.set_title("Ramasesh et al. 2022: ResNet101\noverlap between classes of two tasks")
ax = axs[1, 2]
ax.bar(["Swin-T", "Swin-B", "Swin-L"], [37.42, 40.34, 41.84], color=C[0], width=0.5)
ax.set_ylim(0, 50); ax.axhline(50, color=INK, lw=0.6, ls="--")
ax.set_ylabel("task-gradient pairs with negative cosine (%)"); ax.set_title("Zhang et al. 2024 (arXiv:2411.18615)\nconflict rate vs model size, NYUv2")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "t8_reported.png")); plt.close(fig)

# ---------------------------------------------------------------- coupling functions: REINFORCE, supervised, GRPO
from math import comb
pp = np.linspace(0, 1, 401)


def f_grpo(p, G):
    """Expected size of the group update along (E[s|correct] - E[s|wrong]) with std-normalised advantages:
    sum_c Binom(G, c, p) * sqrt(c/G * (1 - c/G)). Own derivation (see 00-initial-theory.md, Section 3.6)."""
    return sum(comb(G, c) * p ** c * (1 - p) ** (G - c) * np.sqrt(c / G * (1 - c / G)) for c in range(G + 1))


fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
axs[0].plot(pp, pp * (1 - pp), color=C[0], lw=1.8, label="REINFORCE: p(1-p)")
axs[0].plot(pp, 1 - pp, color=C[1], lw=1.8, label="supervised: 1-p")
axs[0].plot(pp, f_grpo(pp, 8), color=C[2], lw=1.8, label="GRPO, 8 answers, std-normalised")
axs[0].plot(pp, f_grpo(pp, 64), color=C[2], lw=1.2, ls="--", label="GRPO, 64 answers")
axs[0].set_xlabel("success probability p of the problem"); axs[0].set_ylabel("coupling function f(p)")
axs[0].set_title("How update size depends on current success"); axs[0].legend(fontsize=7)
axs[1].plot(pp, 1 - pp ** 8 - (1 - pp) ** 8, color=C[2], lw=1.8, label="8 answers")
axs[1].plot(pp, 1 - pp ** 64 - (1 - pp) ** 64, color=C[2], lw=1.2, ls="--", label="64 answers")
axs[1].set_xlabel("success probability p of the problem"); axs[1].set_ylabel("P(group is mixed) = P(non-zero gradient)")
axs[1].set_title("Share of groups that give any gradient"); axs[1].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "t9_coupling.png")); plt.close(fig)
NUM["coupling"] = {f"p={p}": {"reinforce": p * (1 - p), "grpo8": float(f_grpo(p, 8)), "p_mixed8": 1 - p ** 8 - (1 - p) ** 8}
                   for p in (0.01, 0.05, 0.2, 0.5, 0.9, 0.99)}

json.dump(NUM, open(os.path.join(TAB, "theory_numbers.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1))
