"""Linearity and surrogate-transfer tables for 01-understanding.md.

Input: harvested folder of qis-ui-lp (lp.npz, meta.json), written by src/qis/interf/logp_path.py.
Run: python build/analyze_lp.py <harness results dir>      (from reports/<ID>/)
Output: tables/lp_summary.json, figures/l1_transfer.png

Definitions. Attempts are answers sampled from the BASE model, 8 per problem (one group). lp_i(c) is the mean token
log-probability of answer i under condition c = "run@alpha", i.e. weights theta_base + alpha (theta_run - theta_base).
  d_i(c)       = lp_i(c) - lp_i(base)
  A_i          = (r_i - mean r) / (sd r + 1e-6) inside the group of 8 (the GRPO advantage of the training code)
  S_D(c)       = mean over mixed groups of domain D of (1/8) sum_i A_i d_i(c): the change of the GRPO surrogate of
                 domain D at the base policy caused by the weight change. S > 0: probability moved towards correct answers.
  dC, dW       = mean d_i over correct / wrong answers of mixed groups
  linearity    = S_D(run@0.5) / S_D(run@1) (first-order value 0.5) and S_D(run@2) / S_D(run@1) (first-order value 2)
Confidence intervals: bootstrap over problems, 1000 draws, 95% percentile.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.titlesize": 9.5})
D = os.path.join(sys.argv[1], "qis-ui-lp")
z = np.load(os.path.join(D, "lp.npz"))
conds = [str(c) for c in z["conds"]]
lp = z["lp"]
meta = json.load(open(os.path.join(D, "meta.json")))
DOM = ["hard", "related", "unrelated", "control"]
lab = np.array([m["label"] for m in meta]); rew = np.array([float(m["reward"]) for m in meta])
gid = np.array([m["label"] + "|" + m["prompt_id"] for m in meta])
groups = {}
for i, g in enumerate(gid):
    groups.setdefault(g, []).append(i)
adv = np.zeros(len(meta)); mixed = np.zeros(len(meta), dtype=bool)
for g, ix in groups.items():
    r = rew[ix]
    if 0 < r.sum() < len(ix):
        adv[ix] = (r - r.mean()) / (r.std() + 1e-6); mixed[ix] = True
d = lp - lp[conds.index("base@0")][None, :]
gkeys = {p: [g for g in groups if g.startswith(p + "|") and mixed[groups[g][0]]] for p in DOM}
# per-group contributions [n_cond, n_group]
contrib = {p: np.stack([[(adv[groups[g]] * d[c, groups[g]]).mean() for g in gkeys[p]] for c in range(len(conds))])
           for p in DOM}
rng = np.random.default_rng(0)
out = {"n_attempts": len(meta), "n_mixed_groups": {p: len(gkeys[p]) for p in DOM}, "conds": conds, "S": {}, "S_ci": {},
       "dC": {}, "dW": {}, "d_all": {}, "linearity": {}}
for p in DOM:
    X = contrib[p]; n = X.shape[1]
    S = X.mean(1)
    bs = np.stack([X[:, rng.integers(0, n, n)].mean(1) for _ in range(1000)])
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    sel = (lab == p) & mixed
    for c, name in enumerate(conds):
        out["S"].setdefault(name, {})[p] = float(S[c]); out["S_ci"].setdefault(name, {})[p] = [float(lo[c]), float(hi[c])]
        out["dC"].setdefault(name, {})[p] = float(d[c, sel & (rew > 0.5)].mean())
        out["dW"].setdefault(name, {})[p] = float(d[c, sel & (rew < 0.5)].mean())
        out["d_all"].setdefault(name, {})[p] = float(d[c, lab == p].mean())
runs = sorted({c.split("@")[0] for c in conds if c != "base@0"}, key=lambda r: conds.index(next(c for c in conds if c.startswith(r + "@"))))
for r in runs:
    have = {float(c.split("@")[1]): c for c in conds if c.split("@")[0] == r}
    if 1.0 not in have:
        continue
    out["linearity"][r] = {}
    for p in DOM:
        s1 = out["S"][have[1.0]][p]
        out["linearity"][r][p] = {f"S({a})/S(1)": (out["S"][have[a]][p] / s1 if s1 else float("nan")) for a in have if a != 1.0}
        dc1 = out["dC"][have[1.0]][p]
        out["linearity"][r][p].update({f"dC({a})/dC(1)": out["dC"][have[a]][p] / dc1 for a in have if a != 1.0})
json.dump(out, open(os.path.join(TAB, "lp_summary.json"), "w"), indent=1)

# figure: S_D(run@1) heat map, runs x domains
rows = [r for r in runs if f"{r}@1.0" in conds or f"{r}@1" in conds]
def cn(r):
    return f"{r}@1.0" if f"{r}@1.0" in conds else f"{r}@1"
M = np.array([[out["S"][cn(r)][p] for p in DOM] for r in rows])
fig, ax = plt.subplots(figsize=(4.6, 0.33 * len(rows) + 1.2))
v = np.abs(M).max()
im = ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(DOM); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.5)
for i in range(len(rows)):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=7, color=INK if abs(M[i, j]) < 0.6 * v else "white")
ax.set_title("Surrogate transfer S_D(run) at alpha = 1\n(base-policy answers; > 0 = towards correct answers)")
fig.colorbar(im, ax=ax, fraction=0.05)
fig.savefig(os.path.join(FIG, "l1_transfer.png")); plt.close(fig)

print("mixed groups", out["n_mixed_groups"])
for c in conds[1:]:
    print(f"{c:18s}", " ".join(f"{p[:1].upper()} S={out['S'][c][p]:+.4f} [{out['S_ci'][c][p][0]:+.4f},{out['S_ci'][c][p][1]:+.4f}] dC={out['dC'][c][p]:+.3f} dW={out['dW'][c][p]:+.3f} |" for p in DOM))
for r, v in out["linearity"].items():
    print(r, {p: {k: round(x, 2) for k, x in q.items()} for p, q in v.items()})
