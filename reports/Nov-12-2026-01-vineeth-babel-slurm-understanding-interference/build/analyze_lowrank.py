"""Capacity demand: which part of an update carries its effect? Table and figure for Section 1.7 of 01-understanding.md.

Input: harvested folders qis-ui-lowrank-mean6 and qis-ui-lowrank-R101 (lp.npz, meta.json, shares.json), written by
src/qis/interf/lowrank_path.py (FP32 weights, FP32 compute, base-policy answers of four domains).
Run: python build/analyze_lowrank.py <harness results dir>      (from reports/<ID>/)
Output: tables/lowrank.json, figures/c1_lowrank.png

For condition c (full update, its rank-r part per weight matrix, or the remainder):
  S_D(c)  = mean over mixed groups of domain D of (1/8) sum_i A_i [lp_i(c) - lp_i(base)]   (as build/analyze_lp.py)
  S(c)    = mean of S_D over the four domains;  fall(c) = -mean_i [lp_i(c) - lp_i(base)] over all answers
Intervals: bootstrap over problems, 1000 draws, 95% percentile; the ratio to the full update is bootstrapped jointly.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOM = ["hard", "related", "unrelated", "control"]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5, "legend.frameon": False})
out = {}
rng = np.random.default_rng(0)
for exp, run in (("qis-ui-lowrank-mean6", "mean6"), ("qis-ui-lowrank-R101", "R-s101-r128")):
    D = os.path.join(sys.argv[1], exp)
    if not os.path.exists(os.path.join(D, "lp.npz")):
        continue
    z = np.load(os.path.join(D, "lp.npz")); conds = [str(c) for c in z["conds"]]; lp = z["lp"]
    meta = json.load(open(os.path.join(D, "meta.json"))); sh = json.load(open(os.path.join(D, "shares.json")))[run]
    rew = np.array([float(m["reward"]) for m in meta]); lab = np.array([m["label"] for m in meta])
    gid = np.array([m["label"] + "|" + m["prompt_id"] for m in meta])
    keys = sorted(set(gid)); ix = {g: np.where(gid == g)[0] for g in keys}
    adv = np.zeros(len(meta)); mixed = {p: [] for p in DOM}
    for g, i in ix.items():
        r = rew[i]
        if 0 < r.sum() < len(i):
            adv[i] = (r - r.mean()) / (r.std() + 1e-6); mixed[g.split("|")[0]].append(g)
    d = lp - lp[conds.index("base@0")][None, :]
    res = {"shares": sh, "n_answers": len(meta), "n_mixed": {p: len(mixed[p]) for p in DOM}, "conds": {}}
    full = conds.index(f"{run}:full")
    contrib = {p: np.stack([[(adv[ix[g]] * d[c, ix[g]]).mean() for g in mixed[p]] for c in range(len(conds))]) for p in DOM}
    boots = {p: rng.integers(0, len(mixed[p]), (1000, len(mixed[p]))) for p in DOM}
    Sb = np.mean([contrib[p][:, boots[p]].mean(2) for p in DOM], 0)          # [cond, boot], mean over domains
    for c, name in enumerate(conds):
        if c == 0:
            continue
        kind = name.split(":")[1]
        S_D = {p: float(contrib[p][c].mean()) for p in DOM}
        ci_D = {p: [float(np.percentile(contrib[p][c][boots[p]].mean(1), q)) for q in (2.5, 97.5)] for p in DOM}
        S = float(np.mean(list(S_D.values())))
        ratio = Sb[c] / Sb[full]
        share = 1.0 if kind == "full" else (sh["share_top"][kind[3:]] if kind.startswith("top") else 1 - sh["share_top"][kind[4:]])
        res["conds"][kind] = {"norm2_share": share, "S_D": S_D, "S_D_ci": ci_D, "S": S,
                              "S_ci": [float(np.percentile(Sb[c], q)) for q in (2.5, 97.5)],
                              "S_ratio_to_full": S / float(np.mean([contrib[p][full].mean() for p in DOM])),
                              "S_ratio_ci": [float(np.percentile(ratio, q)) for q in (2.5, 97.5)],
                              "fall": float(-d[c].mean()), "fall_ratio_to_full": float(d[c].mean() / d[full].mean())}
    out[run] = res
    print(run, "norm", round(sh["norm"], 3), "2-D share", round(sh["share_2d"], 4), "median stable rank", round(sh["median_stable_rank"], 1))
    for k, v in res["conds"].items():
        print(f"  {k:7s} norm2 share {v['norm2_share']:.3f}  S {v['S']:+.5f} [{v['S_ci'][0]:+.5f},{v['S_ci'][1]:+.5f}]  S/full {v['S_ratio_to_full']:.2f} "
              f"[{v['S_ratio_ci'][0]:.2f},{v['S_ratio_ci'][1]:.2f}]  fall/full {v['fall_ratio_to_full']:.3f}  "
              + " ".join(f"{p[0].upper()}={v['S_D'][p]:+.4f}" for p in DOM))
json.dump(out, open(os.path.join(HERE, "tables", "lowrank.json"), "w"), indent=1)

if out:
    fig, axs = plt.subplots(1, len(out), figsize=(5.2 * len(out), 3.3), squeeze=False)
    for ax, (run, res) in zip(axs[0], out.items()):
        kinds = [k for k in res["conds"] if k != "full"]
        tops = sorted([k for k in kinds if k.startswith("top")], key=lambda k: int(k[3:]))
        tails = sorted([k for k in kinds if k.startswith("tail")], key=lambda k: int(k[4:]))
        for ks, col, name in ((tops, C[0], "rank-r part"), (tails, C[1], "remainder after removing rank r")):
            x = [res["conds"][k]["norm2_share"] for k in ks]; y = [res["conds"][k]["S_ratio_to_full"] for k in ks]
            lo = [res["conds"][k]["S_ratio_ci"][0] for k in ks]; hi = [res["conds"][k]["S_ratio_ci"][1] for k in ks]
            ax.errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - y], fmt="o-", ms=5, lw=1.3, color=col, label=name)
            for k, xi, yi in zip(ks, x, y):
                ax.annotate("r = " + k.replace("top", "").replace("tail", ""), (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=7, color=col)
        ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="effect proportional to squared norm")
        ax.axhline(0, color=INK, lw=0.6); ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("share of the squared norm of the update"); ax.set_ylabel("share of the effect S of the full update")
        ax.set_title(f"{run}: effect of a part of the update (FP32)"); ax.legend(fontsize=7)
    fig.savefig(os.path.join(HERE, "figures", "c1_lowrank.png")); plt.close(fig)
