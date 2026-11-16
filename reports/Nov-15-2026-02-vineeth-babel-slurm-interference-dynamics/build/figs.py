"""Report figures for study s02 (Nov-15-2026-02). Reads harvested metrics.jsonl only.

Usage: python figs.py --res slurm-agentic-engineering/results --out reports/<ID>/figures
Figures:
  add9_strata.png   add9 pass@1 per stratum (d4, d6, d8) vs round, one line per run (RQ2 within-family lead)
  caesar_D.png      D(t) = p(R) - p(J) per caesar stratum with 95% paired problem bootstrap (F6)
Colour: categorical slots in fixed order (dataviz default palette, light mode); every series also has its own
marker and a direct label, so identity never depends on colour alone.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def evals(res, expid):
    """{panel: {round: per_problem list}} from a harvested metrics.jsonl (last record per panel/round wins)."""
    out = {}
    p = os.path.join(res, expid, "metrics.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        r = json.loads(line)
        if r.get("kind") == "eval":
            out.setdefault(r["panel"], {})[r["round"]] = r
    return out


def pass1(rec):
    return rec["pass1"]


def paired_D(ra, rb, B=2000, seed=0):
    """D = p(a) - p(b) over problems shared by both records; 95% percentile bootstrap over problems."""
    a = {x["id"]: x["c"] / x["n"] for x in ra["per_problem"]}
    b = {x["id"]: x["c"] / x["n"] for x in rb["per_problem"]}
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_add9(res, out, runs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, facecolor=SURFACE)
    for k, (stratum, ax) in enumerate(zip(("d4", "d6", "d8"), axes)):
        style(ax)
        ends = []
        for i, (expid, label) in enumerate(runs):
            e = evals(res, expid).get(f"add9_{stratum}_sel", {})
            if not e:
                continue
            rs = sorted(e)
            ys = [pass1(e[r]) for r in rs]
            ax.plot(rs, ys, color=SLOTS[i], marker=MARKERS[i], markersize=5, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label)
            if k == 1:
                ends.append([ys[-1], rs[-1], label])
        ends.sort()
        for n in range(1, len(ends)):                      # keep direct labels >= 0.06 apart
            ends[n][0] = max(ends[n][0], ends[n - 1][0] + 0.06)
        for y, x, label in ends:
            ax.annotate(label, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
        ax.set_title(f"add9 {stratum}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training round", fontsize=8, color=MUTED)
        ax.set_xlim(-1, 44 if k == 1 else 33)
    axes[0].set_ylabel("pass@1 (sel, 64 problems x 4)", fontsize=8, color=MUTED)
    axes[0].set_ylim(-0.02, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "add9_strata.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_caesar_D(res, out, pairs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True, facecolor=SURFACE)
    for stratum, ax in zip(("s4", "s8", "s12"), axes):
        style(ax)
        ax.axhline(0, color=MUTED, linewidth=1)
        for i, (j, r, label) in enumerate(pairs):
            ej, er = evals(res, j).get(f"caesar_{stratum}_sel", {}), evals(res, r).get(f"caesar_{stratum}_sel", {})
            rs = sorted(set(ej) & set(er))
            if not rs:
                continue
            D = [paired_D(er[x], ej[x]) for x in rs]
            m = [x[0] for x in D]
            ax.fill_between(rs, [x[1] for x in D], [x[2] for x in D], color=SLOTS[i], alpha=0.15, linewidth=0)
            ax.plot(rs, m, color=SLOTS[i], marker=MARKERS[i], markersize=5, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        ax.set_title(f"caesar {stratum}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training round", fontsize=8, color=MUTED)
    axes[0].set_ylabel("D = p(alone) - p(joint)\n(+ = joint behind)", fontsize=8, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    if l:
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "caesar_D.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_add9(a.res, a.out, [("qis-s02-e1-add9-s1", "mixed strata, seed 1"),
                            ("qis-s02-e3-add9mix-s2", "mixed strata, seed 2"),
                            ("qis-s02-e3-add9d6-s1", "d6 only"),
                            ("qis-s02-j-add9-add10-s1", "mixed + add10"),
                            ("qis-s02-e3-add9d4-s1", "d4 only"),
                            ("qis-s02-e3-add9d8-s1", "d8 only")])
    fig_caesar_D(a.res, a.out, [("qis-s02-j-caesar-terse-s1", "qis-s02-e1-caesar-s1", "seed 1"),
                                ("qis-s02-j-caesar-terse-s2", "qis-s02-e1-caesar-s2", "seed 2")])


if __name__ == "__main__":
    main()
