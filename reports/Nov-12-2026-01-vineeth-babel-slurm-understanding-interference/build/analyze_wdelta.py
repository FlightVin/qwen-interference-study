"""Weight-update geometry tables and figures for 01-understanding.md.

Input: the harvested `wdelta.json.gz` of experiment qis-ui-wdelta (written by src/qis/interf/wdelta.py).
Run: python build/analyze_wdelta.py <path to wdelta.json.gz>     (from reports/<ID>/)
Output: tables/wdelta_summary.json, figures/w1_cosine.png, w2_where.png, w3_rank.png
"""
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5, "legend.frameon": False})

d = json.load(gzip.open(sys.argv[1], "rt"))
runs, T = d["runs"], d["tensors"]
R = len(runs)
ix = {r: i for i, r in enumerate(runs)}
numel = np.array([np.prod(t["shape"]) for t in T], dtype=np.float64)
gram = np.array([t["gram"] for t in T])                       # [n_tensors, R, R]
base_sq = np.array([t["base_sq"] for t in T])
G = gram.sum(0)
nrm = np.sqrt(np.diag(G))
cos = G / np.outer(nrm, nrm)
out = {"runs": runs, "d": float(numel.sum()), "rel_l2": dict(zip(runs, (nrm / np.sqrt(base_sq.sum())).tolist())),
       "norm": dict(zip(runs, nrm.tolist())), "cos": cos.tolist(), "null_cos_sd": float(numel.sum() ** -0.5)}


def bil(u, v, M=G):
    """<sum_i u_i tau_i, sum_j v_j tau_j> from the Gram matrix."""
    return float(np.array(u) @ M @ np.array(v))


def vec(**kw):
    v = np.zeros(R)
    for k, c in kw.items():
        v[ix[k.replace("_", "-")]] = c
    return v


def cosv(u, v, M=G):
    return bil(u, v, M) / np.sqrt(bil(u, u, M) * bil(v, v, M))


r128 = [r for r in runs if r.endswith("r128")]
pairs = {}
for a in r128:
    for b in r128:
        if a < b:
            arm_a, seed_a = a.split("-")[0], a.split("-")[1]
            arm_b, seed_b = b.split("-")[0], b.split("-")[1]
            kind = ("same seed, different arm" if seed_a == seed_b else
                    "same arm, different seed" if arm_a == arm_b else "different arm and seed")
            pairs.setdefault(kind, []).append((a, b, float(cos[ix[a], ix[b]])))
out["cos_by_kind"] = {k: {"pairs": v, "mean": float(np.mean([x[2] for x in v])), "min": float(min(x[2] for x in v)),
                          "max": float(max(x[2] for x in v))} for k, v in pairs.items()}
# arm-difference directions: is there a reproducible "R minus U" (or "practice minus H") direction across seeds?
diffs = {}
for x, y in (("R", "U"), ("R", "H"), ("U", "H")):
    u = vec(**{f"{x}_s101_r128": 1, f"{y}_s101_r128": -1}); v = vec(**{f"{x}_s202_r128": 1, f"{y}_s202_r128": -1})
    diffs[f"{x}-{y}"] = {"cos_across_seeds": cosv(u, v), "norm_s101": float(np.sqrt(bil(u, u))), "norm_s202": float(np.sqrt(bil(v, v)))}
# control: seed-difference directions within an arm pair
u = vec(R_s101_r128=1, R_s202_r128=-1); v = vec(U_s101_r128=1, U_s202_r128=-1); w = vec(H_s101_r128=1, H_s202_r128=-1)
diffs["seed101-seed202"] = {"cos R vs U": cosv(u, v), "cos R vs H": cosv(u, w), "cos U vs H": cosv(v, w)}
out["difference_directions"] = diffs
# continuation: first half of training vs second half
cont = {}
for run in ("R-s101", "H-s202"):
    a, b = f"{run}-r64", f"{run}-r128"
    if a in ix and b in ix:
        e, f = np.zeros(R), np.zeros(R)
        e[ix[a]] = 1; f[ix[b]] = 1; f[ix[a]] = -1
        cont[run] = {"cos(tau64, tau128-tau64)": cosv(e, f), "norm64": float(np.sqrt(bil(e, e))),
                     "norm_second_half": float(np.sqrt(bil(f, f))), "norm128": float(nrm[ix[b]])}
out["continuation"] = cont

# by module type and by layer
types = ["embed", "q", "k", "v", "o", "gate", "up", "down", "norm"]
layers = []
for t in T:
    if t["layer"] not in layers:
        layers.append(t["layer"])
i128 = [ix[r] for r in r128]


def group(key, names):
    res = {}
    for nme in names:
        sel = [i for i, t in enumerate(T) if t[key] == nme]
        Gs = gram[sel].sum(0); n = np.sqrt(np.diag(Gs)); c = Gs / np.outer(n, n)
        off = [c[a, b] for a in i128 for b in i128 if a < b]
        res[nme] = {"params": float(numel[sel].sum()), "share_of_update_sq": float(np.mean(np.diag(Gs)[i128] / np.diag(G)[i128])),
                    "rel_change": float(np.mean(n[i128]) / np.sqrt(base_sq[sel].sum())),
                    "mean_cos_r128_pairs": float(np.mean(off)), "min_cos": float(np.min(off)), "max_cos": float(np.max(off)),
                    "cos_RU_diff_across_seeds": cosv(vec(R_s101_r128=1, U_s101_r128=-1), vec(R_s202_r128=1, U_s202_r128=-1), Gs)}
    return res


out["by_type"] = group("type", types)
out["by_layer"] = group("layer", layers)


def wmean(field, idx=None):
    a = np.array([t[field] for t in T], dtype=np.float64)
    if idx is not None:
        a = a[:, idx]
    return (a * numel.reshape((-1,) + (1,) * (a.ndim - 1))).sum(0) / numel.sum()


sa = wmean("sign_agree")
out["sign_agree"] = {k: float(np.mean([sa[ix[a], ix[b]] for a, b, _ in v])) for k, v in pairs.items()}
out["frac_zero"] = dict(zip(runs, wmean("frac_zero").tolist()))
out["frac_bf16_changed"] = dict(zip(runs, wmean("frac_bf16_changed").tolist()))
bfsq = np.array([t["bf16_delta_sq"] for t in T]).sum(0)
cb = np.array([t["cos_bf16_vs_fp32"] for t in T]); fsq = np.array([np.diag(t["gram"]) for t in T])
bsq = np.array([t["bf16_delta_sq"] for t in T])
dot = (cb * np.sqrt(fsq * bsq)).sum(0)
out["bf16"] = {"norm_ratio_bf16_over_fp32": dict(zip(runs, (np.sqrt(bfsq) / nrm).tolist())),
               "cos_bf16_vs_fp32": dict(zip(runs, (dot / np.sqrt(bfsq * np.diag(G))).tolist()))}
lr = d["lr"]
am = wmean("abs_mean")
out["abs_over_lr"] = {"mean": dict(zip(runs, (am / lr).tolist())),
                      "q_levels": T[0]["q_levels"],
                      "q_weighted": dict(zip(runs, (wmean("abs_q") / lr).tolist())),
                      "max": dict(zip(runs, (np.array([t["abs_max"] for t in T]).max(0) / lr).tolist())),
                      "steps_r128": 512, "random_walk_sqrt_steps": float(np.sqrt(512))}
# rank statistics of 2-D block matrices
M = [t for t in T if "stable_rank" in t]
if M:
    sr = np.array([t["stable_rank"] for t in M]); pr = np.array([t["participation_ratio"] for t in M])
    md = np.array([t["min_dim"] for t in M], dtype=np.float64)
    e = {k: np.array([t["energy_top"][k] for t in M]) for k in ("1", "8", "64")}
    sl = np.array([t["sub_left"] for t in M]); srr = np.array([t["sub_right"] for t in M])
    offm = lambda A: float(np.mean([A[:, a, b].mean() for a in i128 for b in i128 if a < b]))
    out["rank"] = {"n_matrices": len(M), "k": d["k"],
                   "stable_rank_median": float(np.median(sr[:, i128])), "stable_rank_q10_q90": [float(np.percentile(sr[:, i128], 10)), float(np.percentile(sr[:, i128], 90))],
                   "participation_ratio_median": float(np.median(pr[:, i128])),
                   "participation_ratio_over_min_dim_median": float(np.median(pr[:, i128] / md[:, None])),
                   "energy_top1_mean": float(e["1"][:, i128].mean()), "energy_top8_mean": float(e["8"][:, i128].mean()),
                   "energy_top64_mean": float(e["64"][:, i128].mean()),
                   "null_energy_top8": float(np.mean(8 / md)), "null_energy_top64": float(np.mean(64 / md)),
                   "subspace_overlap_left_mean": offm(sl), "subspace_overlap_right_mean": offm(srr),
                   "subspace_null": float(np.mean(d["k"] / md)),
                   "by_type_overlap_left": {ty: float(np.mean([np.mean([t["sub_left"][a][b] for a in i128 for b in i128 if a < b])
                                                              for t in M if t["type"] == ty])) for ty in types if any(t["type"] == ty for t in M)},
                   "by_type_stable_rank": {ty: float(np.median([np.array(t["stable_rank"])[i128] for t in M if t["type"] == ty]))
                                           for ty in types if any(t["type"] == ty for t in M)}}
json.dump(out, open(os.path.join(TAB, "wdelta_summary.json"), "w"), indent=1)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(figsize=(5.4, 4.4))
im = ax.imshow(cos, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(R)); ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(R)); ax.set_yticklabels(runs, fontsize=7); ax.grid(False)
for i in range(R):
    for j in range(R):
        ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if cos[i, j] > 0.6 else INK)
ax.set_title("Cosine between weight updates (theta_run - theta_base)")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.savefig(os.path.join(FIG, "w1_cosine.png")); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
bt = out["by_type"]
axs[0].bar(types, [bt[t]["share_of_update_sq"] for t in types], color=C[0], width=0.6)
axs[0].set_ylabel("share of squared update norm"); axs[0].set_title("Where the update is, by module type")
axs[1].bar(types, [bt[t]["rel_change"] * 1e3 for t in types], color=C[0], width=0.6)
axs[1].set_ylabel("|update| / |base weights|  (x 1e-3)"); axs[1].set_title("Relative change, by module type")
bl = out["by_layer"]
num = [l for l in layers if l.isdigit()]
axs[2].plot([int(l) for l in num], [bl[l]["mean_cos_r128_pairs"] for l in num], "o-", color=C[0], ms=3, lw=1.4, label="mean cosine between runs")
axs[2].plot([int(l) for l in num], [bl[l]["cos_RU_diff_across_seeds"] for l in num], "s-", color=C[1], ms=3, lw=1.2, label="cos of (R - U) across seeds")
axs[2].axhline(0, color=INK, lw=0.6); axs[2].set_xlabel("transformer block"); axs[2].set_title("Update similarity by depth"); axs[2].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "w2_where.png")); plt.close(fig)

if M:
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ex = [t for t in M if t.get("sv_top")]
    for t in ex:
        if t["type"] in ("q", "down") and t["layer"] in ("0", "13", "27"):
            s = np.array(t["sv_top"])[i128[0]]
            axs[0].semilogy(np.arange(1, len(s) + 1), s / s[0], lw=1.3, label=f"{t['type']} block {t['layer']}",
                            color=C[("0", "13", "27").index(t["layer"])], ls="-" if t["type"] == "q" else "--")
    axs[0].set_xlabel("singular value index"); axs[0].set_ylabel("singular value / largest"); axs[0].legend(fontsize=6.5)
    axs[0].set_title(f"Spectrum of the update, run {r128[0]}")
    ty = list(out["rank"]["by_type_overlap_left"])
    axs[1].bar(ty, [out["rank"]["by_type_overlap_left"][t] for t in ty], color=C[0], width=0.6)
    axs[1].axhline(out["rank"]["subspace_null"], color=C[1], lw=1, ls="--", label="random subspaces")
    axs[1].set_ylabel(f"overlap of top-{d['k']} left singular subspaces"); axs[1].set_title("Do runs change the same directions?")
    axs[1].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "w3_rank.png")); plt.close(fig)
print(json.dumps({k: out[k] for k in ("rel_l2", "cos_by_kind", "difference_directions", "continuation", "sign_agree",
                                      "frac_zero", "frac_bf16_changed", "bf16", "abs_over_lr")}, indent=1)[:6000])
if M:
    print(json.dumps(out["rank"], indent=1))
