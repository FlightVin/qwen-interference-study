"""Gradient-geometry tables and figures for 01-understanding.md.

Input: harvested result folders of the qis-ui-geom-* experiments (summary.json, small.npz, groups_slim.json,
sample_*.json), written by src/qis/interf/summ.py.
Run: python build/analyze_geom.py <harness results dir>      (from reports/<ID>/)
Output: tables/geom_summary.json, figures/g1_cos_hist.png, g2_pairs.png, g3_progress.png, g4_layers.png
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
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5, "legend.frameon": False})
RES = sys.argv[1]
ORDER = ["q06", "base", "base64", "q4", "R-s101-r64", "H-s202-r64", "R-s202-r128", "U-s202-r128", "H-s101-r128"]
DOM = ["hard", "related", "unrelated", "control"]
SHORT = {"hard": "H", "related": "R", "unrelated": "U", "control": "C"}
out = {}
for ck in ORDER:
    d = os.path.join(RES, f"qis-ui-geom-{ck}")
    if not os.path.exists(os.path.join(d, "summary.json")):
        continue
    r = json.load(open(os.path.join(d, "summary.json")))
    g = r["global"]
    row = {"n_mixed": g["n_groups"], "n_problems_with_mixed": g["n_problems"], "mean_norm": g["mean_norm"],
           "same_problem": g["same_problem"], "pairs": g["pairs"], "power": g["power"], "rho": g["rho"],
           "ci95": g["ci95"], "sum_cos": g["sum_cos"], "sketch_norm_ratio": r["sketch_norm_ratio"],
           "pass1": {}, "groups_total": {}, "norm_share_type": r["norm_share_type"]}
    for p in DOM:
        f = os.path.join(d, f"sample_{p}.json")
        if os.path.exists(f):
            s = json.load(open(f))
            row["pass1"][p] = s["pass1"]; row["groups_total"][p] = s["problems"] * s["n"] // 8
    if "scaled" in r:
        sc = r["scaled"]
        row["scaled"] = {k: sc[k] for k in ("same_problem", "pairs", "power", "rho", "ci95", "sum_cos")}
    row["by_type"] = {t: {"pairs": {k: v["mean_cos"] for k, v in x["pairs"].items()},
                          "same_problem": {k: v["mean_cos"] for k, v in x["same_problem"].items()}}
                      for t, x in r["by_type"].items()}
    row["by_layer"] = {t: {"pairs": {k: v["mean_cos"] for k, v in x["pairs"].items()},
                           "same_problem": {k: v["mean_cos"] for k, v in x["same_problem"].items()}}
                       for t, x in r["by_layer"].items()}
    out[ck] = row
json.dump(out, open(os.path.join(TAB, "geom_summary.json"), "w"), indent=1)
cks = list(out)


def pk(a, b):
    return f"{a}|{b}" if f"{a}|{b}" in out[cks[0]]["pairs"] or a <= b else f"{b}|{a}"


def key(a, b):
    return "|".join(sorted([a, b]))


# ---------- g1: cosine histograms at the highest-powered base checkpoint
ck0 = "base64" if "base64" in out else "base"
ckh = ck0 if os.path.exists(os.path.join(RES, f"qis-ui-geom-{ck0}", "small.npz")) else "base"   # base64 matrix exceeds the harvest cap
z = np.load(os.path.join(RES, f"qis-ui-geom-{ckh}", "small.npz"))
gs = [g for g in json.load(open(os.path.join(RES, f"qis-ui-geom-{ckh}", "groups_slim.json"))) if g["row"] >= 0]
gs.sort(key=lambda g: g["row"])
lab = np.array([g["label"] for g in gs]); pid = np.array([g["label"] + "|" + g["prompt_id"] for g in gs])
Cm = z["cos"].astype(np.float64)
iu = np.triu_indices(len(gs), 1)
same_p = (pid[:, None] == pid[None, :])[iu]; same_d = (lab[:, None] == lab[None, :])[iu]
c = Cm[iu]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.1)); fig.subplots_adjust(wspace=0.28)
bins = np.linspace(-0.2, 0.3, 101)
for sel, name, col in ((~same_d, "different domains", C[0]), (same_d & ~same_p, "same domain, different problems", C[2]),
                       (same_p, "same problem, independent samples", C[1])):
    axs[0].hist(c[sel], bins=bins, density=True, histtype="step", lw=1.5, color=col,
                label=f"{name}: mean {c[sel].mean():+.4f}, n = {sel.sum():,}")
axs[0].axvline(0, color=INK, lw=0.6); axs[0].set_xlabel("cosine between two group gradients"); axs[0].set_ylabel("density")
axs[0].set_title(f"Pairwise gradient cosines, base model ({ckh})"); axs[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
axs[0].set_xlim(-0.2, 0.3)
g = out[ck0]
names, vals, lo, hi = [], [], [], []
for a in DOM:
    names.append(f"{SHORT[a]}*"); vals.append(g["same_problem"][a]["mean_cos"]); lo.append(np.nan); hi.append(np.nan)
for i, a in enumerate(DOM):
    for b in DOM[i:]:
        k = key(a, b)
        names.append(f"{SHORT[a]}-{SHORT[b]}"); vals.append(g["pairs"][k]["mean_cos"])
        ci = g["ci95"][f"mean_cos:{k}"]; lo.append(ci[0]); hi.append(ci[1])
x = np.arange(len(names))
cols = [C[1]] * 4 + [C[2] if n.split("-")[0] == n.split("-")[1] else C[0] for n in names[4:]]
axs[1].bar(x, vals, color=cols, width=0.65)
for xi, l, h in zip(x, lo, hi):
    if not np.isnan(l):
        axs[1].plot([xi, xi], [l, h], color=INK, lw=1)
axs[1].set_xticks(x); axs[1].set_xticklabels(names, fontsize=6.5); axs[1].set_yscale("symlog", linthresh=0.005)
axs[1].axhline(0, color=INK, lw=0.6); axs[1].set_ylabel("mean cosine (symlog axis)")
axs[1].set_title(f"Mean cosine by pair type ({ck0}); X* = same problem")
fig.savefig(os.path.join(FIG, "g1_cos_hist.png")); plt.close(fig)

# ---------- g2 / g3: by checkpoint
PAIRS = [("hard", "hard"), ("hard", "related"), ("hard", "unrelated"), ("related", "unrelated"), ("hard", "control")]
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.2)); fig.subplots_adjust(wspace=0.33)
x = np.arange(len(cks))
for j, (a, b) in enumerate(PAIRS):
    k = key(a, b)
    v = [out[c_]["pairs"].get(k, {}).get("mean_cos", np.nan) for c_ in cks]
    ci = [out[c_]["ci95"].get(f"mean_cos:{k}", [np.nan, np.nan]) for c_ in cks]
    off = (j - 2) * 0.15
    axs[0].errorbar(x + off, v, yerr=[np.array(v) - np.array([q[0] for q in ci]), np.array([q[1] for q in ci]) - np.array(v)],
                    fmt="o", ms=4, color=C[j], lw=1, label=f"{SHORT[a]}-{SHORT[b]}")
axs[0].axhline(0, color=INK, lw=0.6); axs[0].set_xticks(x); axs[0].set_xticklabels(cks, rotation=35, ha="right", fontsize=7)
axs[0].set_ylabel("mean cosine, different problems"); axs[0].set_title("Cosine by domain pair (95% interval)"); axs[0].legend(fontsize=7, ncol=2)
for j, a in enumerate(DOM):
    v = [out[c_]["same_problem"].get(a, {}).get("mean_cos", np.nan) for c_ in cks]
    axs[1].plot(x, v, "o-", ms=4, lw=1.2, color=C[j], label=SHORT[a])
axs[1].axhline(0, color=INK, lw=0.6); axs[1].set_xticks(x); axs[1].set_xticklabels(cks, rotation=35, ha="right", fontsize=7)
axs[1].set_ylabel("mean cosine, same problem"); axs[1].set_title("Repeatable share of one group gradient"); axs[1].legend(fontsize=7)
for j, a in enumerate(DOM):
    v = [out[c_]["n_mixed"].get(a, 0) / max(out[c_]["groups_total"].get(a, 1), 1) for c_ in cks]
    axs[2].plot(x, v, "o-", ms=4, lw=1.2, color=C[j], label=SHORT[a])
axs[2].set_xticks(x); axs[2].set_xticklabels(cks, rotation=35, ha="right", fontsize=7); axs[2].set_ylim(0, 1)
axs[2].set_ylabel("share of groups with a gradient"); axs[2].set_title("Groups with right and wrong answers"); axs[2].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "g2_progress.png")); plt.close(fig)

# ---------- g4: by layer at ck0
bl = out[ck0]["by_layer"]
num = [l for l in bl if l.isdigit()]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
for j, (a, b) in enumerate(PAIRS):
    k = key(a, b)
    axs[0].plot([int(l) for l in num], [bl[l]["pairs"][k] for l in num], "o-", ms=3, lw=1.2, color=C[j], label=f"{SHORT[a]}-{SHORT[b]}")
axs[0].axhline(0, color=INK, lw=0.6); axs[0].set_xlabel("transformer block"); axs[0].set_ylabel("mean pairwise cosine")
axs[0].set_title(f"Cross-domain cosine by depth ({ck0})"); axs[0].legend(fontsize=7, ncol=2)
for j, a in enumerate(DOM):
    axs[1].plot([int(l) for l in num], [bl[l]["same_problem"][a] for l in num], "o-", ms=3, lw=1.2, color=C[j], label=SHORT[a])
axs[1].axhline(0, color=INK, lw=0.6); axs[1].set_xlabel("transformer block"); axs[1].set_ylabel("mean cosine, same problem")
axs[1].set_title("Same-problem cosine by depth"); axs[1].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "g4_layers.png")); plt.close(fig)

for c_ in cks:
    g = out[c_]
    print(c_, "mixed", g["n_mixed"], "pass1", {k: round(v, 3) for k, v in g["pass1"].items()})
    print("   same-problem", {SHORT[k]: round(v["mean_cos"], 4) for k, v in g["same_problem"].items()})
    print("   pairs", {k: (round(v["mean_cos"], 5), round(v["conflict_rate"], 3)) for k, v in g["pairs"].items()})
    print("   rho", {k: round(v, 2) for k, v in g["rho"].items()})
    if "scaled" in g:
        print("   scaled pairs", {k: round(v["mean_cos"], 5) for k, v in g["scaled"]["pairs"].items()})
        print("   scaled same", {SHORT[k]: round(v["mean_cos"], 4) for k, v in g["scaled"]["same_problem"].items()})
