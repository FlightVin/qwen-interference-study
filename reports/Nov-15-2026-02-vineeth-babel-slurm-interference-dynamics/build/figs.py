"""Report figures for study s02 (Nov-15-2026-02). Reads harvested metrics.jsonl only.

Usage: python figs.py --res slurm-agentic-engineering/results --out reports/<ID>/figures
Figures:
  add9_d6.png       add9 d6 pass@1 vs round: colour = training condition, line style = seed (RQ2 collapse)
  add9_strata.png   d4 / d8 for the same runs (context)
  recovery.png      E3b branches from the seed-2 collapse state (d6 conf and d4 sel)
  caesar_D.png      D(t) = p(alone) - p(joint) per caesar stratum, 95% paired problem bootstrap (F6, F11, F15)
  offtarget.png     off-target change r32 - r0 for single-task runs (diverging table)
Colour: dataviz default categorical slots in fixed order (light mode). Identity never depends on colour alone:
seeds differ by line style, conditions by marker, and every chart has a legend or direct labels.
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
STYLES = {1: "-", 2: "--", 3: ":"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
DIV_NEG, DIV_POS = "#e34948", "#2a78d6"     # diverging poles (red = harm, blue = gain), grey midpoint


def evals(res, expid):
    """{panel: {round: record}} from a harvested metrics.jsonl (last record per panel/round wins)."""
    out = {}
    p = os.path.join(res, expid, "metrics.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        r = json.loads(line)
        if r.get("kind") == "eval":
            out.setdefault(r["panel"], {})[r["round"]] = r
    return out


def paired_D(ra, rb, B=2000, seed=0):
    """mean of p(a) - p(b) over shared problems; 95% percentile bootstrap over problems."""
    a = {x["id"]: x["c"] / x["n"] for x in ra["per_problem"]}
    b = {x["id"]: x["c"] / x["n"] for x in rb["per_problem"]}
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    bs = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(axis=1)
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


def line(ax, xs, ys, i, seed=1, label=None):
    ax.plot(xs, ys, color=SLOTS[i], marker=MARKERS[i], markersize=5, linewidth=2, linestyle=STYLES.get(seed, "-"),
            markeredgecolor=SURFACE, markeredgewidth=1, label=label)


def legend_top(fig, ax, ncol=None):
    h, l = ax.get_legend_handles_labels()
    if len(l) > 1:
        fig.legend(h, l, loc="upper center", ncol=ncol or len(l), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.995))


# condition -> (slot, [(expid, seed)])
ADD9 = [("mixed d4+d6+d8", 0, [("qis-s02-e1-add9-s1", 1), ("qis-s02-e3-add9mix-s2", 2), ("qis-s02-e3-add9mix-s3r", 3)]),
        ("d4+d6 (no d8)", 4, [("qis-s02-e3-add9d4d6-s1", 1), ("qis-s02-e3-add9d4d6-s2", 2), ("qis-s02-e3-add9d4d6-s3", 3)]),
        ("d6 only", 1, [("qis-s02-e3-add9d6-s1", 1)]),
        ("d4 only", 2, [("qis-s02-e3-add9d4-s1", 1)]),
        ("d8 only", 3, [("qis-s02-e3-add9d8-s1", 1)]),
        ("mixed + add10", 5, [("qis-s02-j-add9-add10-s1", 1)]),
        ("mixed + terse", 6, [("qis-s02-j-add9-terse-s1", 1)])]


def fig_add9(res, out):
    for name, strata, size in (("add9_d6.png", ["d6"], (7.5, 4.2)), ("add9_strata.png", ["d4", "d8"], (11, 3.8))):
        fig, axes = plt.subplots(1, len(strata), figsize=size, sharey=True, facecolor=SURFACE, squeeze=False)
        for st, ax in zip(strata, axes[0]):
            style(ax)
            for cond, slot, runs in ADD9:
                for k, (expid, seed) in enumerate(runs):
                    e = evals(res, expid).get(f"add9_{st}_sel", {})
                    if e:
                        rs = sorted(e)
                        lab = f"{cond}" + (f" (seed {seed})" if len(runs) > 1 else "")
                        line(ax, rs, [e[r]["pass1"] for r in rs], slot, seed, lab)
            ax.set_title(f"add9 {st} (sel, 64 problems x 4 samples)", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("training round", fontsize=8, color=MUTED)
            ax.set_ylim(-0.02, 1.0)
        axes[0][0].set_ylabel("pass@1", fontsize=8, color=MUTED)
        legend_top(fig, axes[0][0], ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.80))
        fig.savefig(os.path.join(out, name), dpi=150, facecolor=SURFACE)
        plt.close(fig)


def fig_recovery(res, out):
    parent = evals(res, "qis-s02-e3-add9mix-s2")
    br = [("original mixture", 0, "qis-s02-e3b-r16-mixed"), ("d6 only, 8 samples", 1, "qis-s02-e3b-r16-d6"),
          ("d6 only, 32 samples", 2, "qis-s02-e3b-r16-d6n32")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, panel, title in ((axes[0], "add9_d6_conf", "d6 (conf, 64 x 8)"), (axes[1], "add9_d4_sel", "d4 (sel, 64 x 4)")):
        style(ax)
        p = parent.get(panel, {})
        if p:
            rs = [r for r in sorted(p) if r <= 16]
            ax.plot(rs, [p[r]["pass1"] for r in rs], color=MUTED, linewidth=1.5, marker="o", markersize=3,
                    label="parent run (mixed, seed 2)")
        for name, i, expid in br:
            e = evals(res, expid).get(panel, {})
            if e:
                rs = sorted(e)
                line(ax, rs, [e[r]["pass1"] for r in rs], i, 1, name)
        ax.axvline(16, color=GRID, linewidth=1)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training round (branches start at 16)", fontsize=8, color=MUTED)
    axes[0].set_ylabel("pass@1", fontsize=8, color=MUTED)
    legend_top(fig, axes[1])                     # right panel also carries the parent line
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(out, "recovery.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_caesar_D(res, out):
    pairs = [("qis-s02-j-caesar-terse-s1", "qis-s02-e1-caesar-s1", "with terse, seed 1", 0, 1),
             ("qis-s02-j-caesar-terse-s2", "qis-s02-e1-caesar-s2", "with terse, seed 2", 0, 2),
             ("qis-s02-j-caesar-terse-s3", "qis-s02-e1-caesar-s3", "with terse, seed 3", 0, 3),
             ("qis-s02-j-caesar-mbpp-s1", "qis-s02-e1-caesar-s1", "with MBPP, seed 1", 1, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, facecolor=SURFACE)
    for stratum, ax in zip(("s4", "s8", "s12"), axes):
        style(ax)
        ax.axhline(0, color=MUTED, linewidth=1)
        for j, r, label, slot, seed in pairs:
            ej, er = evals(res, j).get(f"caesar_{stratum}_sel", {}), evals(res, r).get(f"caesar_{stratum}_sel", {})
            rs = sorted(set(ej) & set(er))
            if not rs:
                continue
            D = [paired_D(er[x], ej[x]) for x in rs]
            ax.fill_between(rs, [x[1] for x in D], [x[2] for x in D], color=SLOTS[slot], alpha=0.10, linewidth=0)
            line(ax, rs, [x[0] for x in D], slot, seed, label)
        ax.set_title(f"caesar {stratum}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training round", fontsize=8, color=MUTED)
    axes[0].set_ylabel("D = p(caesar alone) - p(joint)\n(+ = joint run behind)", fontsize=8, color=MUTED)
    legend_top(fig, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(out, "caesar_D.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_offtarget(res, out):
    runs = [("caesar", "qis-s02-e1-caesar-s1"), ("caesar s2", "qis-s02-e1-caesar-s2"), ("caesar s3", "qis-s02-e1-caesar-s3"), ("MBPP", "qis-s02-e1-mbpp-s1"),
            ("add9 mixed", "qis-s02-e1-add9-s1"), ("add9 d4", "qis-s02-e3-add9d4-s1"), ("add9 d6", "qis-s02-e3-add9d6-s1"),
            ("add9 d8", "qis-s02-e3-add9d8-s1")]
    panels = ["caesar_s4_sel", "caesar_s8_sel", "caesar_s12_sel", "add9_d4_sel", "add9_d6_sel", "add9_d8_sel",
              "mbpp_all_sel", "arc_all_sel", "countdown_n3_sel", "add10_d14_sel", "terse10_k2_sel"]
    M = np.full((len(runs), len(panels)), np.nan)
    sig = np.zeros_like(M, dtype=bool)
    for i, (_, expid) in enumerate(runs):
        e = evals(res, expid)
        for j, p in enumerate(panels):
            if p in e and 0 in e[p] and 32 in e[p]:
                d, lo, hi = paired_D(e[p][32], e[p][0])
                M[i, j] = d
                sig[i, j] = lo > 0 or hi < 0
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, "#f1f0ec", DIV_POS])
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURFACE)
    ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -0.9, 0.9), aspect="auto")
    for i in range(len(runs)):
        for j in range(len(panels)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}" + ("" if sig[i, j] else "·"), ha="center", va="center",
                        fontsize=7, color=INK, fontweight="bold" if sig[i, j] else "normal")
    ax.set_xticks(range(len(panels)), [p.replace("_sel", "").replace("_all", "") for p in panels], fontsize=8,
                  color=MUTED, rotation=30, ha="right")
    ax.set_yticks(range(len(runs)), [r[0] for r in runs], fontsize=8, color=MUTED)
    ax.set_title("pass@1 change r32 - r0 by trained task (rows) and evaluated panel (columns); bold = 95% CI excludes 0, "
                 "· = not significant", fontsize=9, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "offtarget.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_add9(a.res, a.out)
    fig_recovery(a.res, a.out)
    fig_caesar_D(a.res, a.out)
    fig_offtarget(a.res, a.out)


if __name__ == "__main__":
    main()
