"""Additional report figures (requested 2026-11-17). Reads harvested metrics.jsonl only; no project code.

Usage: python figs_extra.py --res slurm-agentic-engineering/results --out reports/<ID>/figures [--momentum FILE]
Figures:
  caesar_mbpp_curves.png  MBPP alone vs with Caesar; Caesar (3 shifts) alone vs with MBPP vs alone at half lr
  step_sharing.png        Caesar-MBPP lag vs step-size sensitivity: competitor activity, r32 accuracy by condition,
                          joint-run lag vs half-lr cost over training (a comparison of magnitudes, not an equivalence)
  add9_pairs.png          base-9 addition alone (collapsed seed, non-collapsed seeds as a range) vs with add10 / terse;
                          competitor learning
  terse_spillover.png     terse accuracy and answer length in every run, trained vs untrained
  momentum_branches.png   measured: per-step gradient norm and accuracy in the three recovery branches
  momentum_cpu.png        measured on CPU (if --momentum given): weight movement under zero-gradient AdamW steps
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs import SLOTS, MARKERS, STYLES, INK, MUTED, GRID, SURFACE, evals, paired_D, style, line, legend_top  # noqa: E402


def load(res, expid):
    """(evals, rounds, steps) from a harvested metrics.jsonl; last record per key wins."""
    E, R, S = {}, {}, {}
    p = os.path.join(res, expid, "metrics.jsonl")
    for l in open(p):
        r = json.loads(l)
        if r["kind"] == "eval":
            E.setdefault(r["panel"], {})[r["round"]] = r
        elif r["kind"] == "round":
            R[r["round"]] = r
        elif r["kind"] == "step":
            S[r["opt_step"]] = r
    return E, R, S


def boot_ci(rec, B=2000, seed=0):
    v = np.array([x["c"] / x["n"] for x in rec["per_problem"]])
    bs = v[np.random.default_rng(seed).integers(0, len(v), (B, len(v)))].mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def curve(ax, E, panel, i, seed=1, label=None, band=True, color=None, lw=2):
    e = E.get(panel, {})
    rs = sorted(e)
    if not rs:
        return
    ys = [e[r]["pass1"] for r in rs]
    c = color or SLOTS[i]
    if band:
        ci = [boot_ci(e[r]) for r in rs]
        ax.fill_between(rs, [x[0] for x in ci], [x[1] for x in ci], color=c, alpha=0.10, linewidth=0)
    ax.plot(rs, ys, color=c, marker=MARKERS[i % len(MARKERS)], markersize=5, linewidth=lw,
            linestyle=STYLES.get(seed, "-"), markeredgecolor=SURFACE, markeredgewidth=1, label=label)


def save(fig, out, name, top=0.86):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(os.path.join(out, name), dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 1. Caesar and MBPP, alone and together
def fig_caesar_mbpp(res, out):
    RC, RM, J, H = (load(res, e)[0] for e in ("qis-s02-e1-caesar-s1", "qis-s02-e1-mbpp-s1",
                                               "qis-s02-j-caesar-mbpp-s1", "qis-s02-e1-caesar-lr1e6-s1"))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), facecolor=SURFACE)
    ax = axes[0]; style(ax)
    curve(ax, RM, "mbpp_all_sel", 1, label="trained alone")
    curve(ax, J, "mbpp_all_sel", 0, label="trained with Caesar")
    ax.set_title("MBPP", fontsize=10, color=INK, loc="left"); ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("accuracy (pass@1)", fontsize=8, color=MUTED)
    for ax, st in zip(axes[1:], ("s4", "s8", "s12")):
        style(ax)
        curve(ax, RC, f"caesar_{st}_sel", 1, label="trained alone")
        curve(ax, J, f"caesar_{st}_sel", 0, label="trained with MBPP")
        curve(ax, H, f"caesar_{st}_sel", 7, seed=2, label="trained alone, half learning rate", band=False, color=MUTED)
        ax.set_title(f"Caesar, shift {st[1:]}", fontsize=10, color=INK, loc="left"); ax.set_ylim(-0.02, 1.0)
    for ax in axes:
        ax.set_xlabel("training round", fontsize=8, color=MUTED)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    save(fig, out, "caesar_mbpp_curves.png")


# ---------------------------------------------------------------- 2. update sharing
def fig_step_sharing(res, out):
    RC, _, _ = load(res, "qis-s02-e1-caesar-s1")
    H, _, _ = load(res, "qis-s02-e1-caesar-lr1e6-s1")
    J, JR, _ = load(res, "qis-s02-j-caesar-mbpp-s1")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1.2, 1.3]})
    # (a) competitor activity: cumulative mixed (gradient-producing) competitor groups
    ax = axes[0]; style(ax)
    rs = sorted(JR)
    ax.plot(rs, np.cumsum([JR[r]["by_role"]["C"]["mixed"] for r in rs]), color=SLOTS[1], marker="s", markersize=3,
            linewidth=2.2, label="MBPP, trained with Caesar")
    for s in (1, 2, 3):
        _, TR, _ = load(res, f"qis-s02-j-caesar-terse-s{s}")
        rt = sorted(TR)
        ax.plot(rt, np.cumsum([TR[r]["by_role"]["C"]["mixed"] for r in rt]), color=SLOTS[0], linewidth=1.8,
                linestyle=STYLES[s], label=f"terse, trained with Caesar (seed {s})")
    ax.set_title("(a) Competitor groups that produced gradient\n(cumulative mixed groups)", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("training round", fontsize=8, color=MUTED); ax.set_ylabel("cumulative mixed groups", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    # (b) accuracy at round 32 with 95% CI, by condition and shift
    ax = axes[1]; style(ax)
    conds = [("alone, full learning rate", RC, 1), ("with MBPP", J, 0), ("alone, half learning rate", H, 7)]
    x = np.arange(3); w = 0.26
    for k, (name, E, slot) in enumerate(conds):
        ys, lo, hi = [], [], []
        for st in ("s4", "s8", "s12"):
            rec = E[f"caesar_{st}_sel"][32]; a, b = boot_ci(rec)
            ys.append(rec["pass1"]); lo.append(rec["pass1"] - a); hi.append(b - rec["pass1"])
        ax.bar(x + (k - 1) * w, ys, w * 0.92, color=SLOTS[slot] if slot != 7 else MUTED, label=name,
               yerr=[lo, hi], error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
        for xi, y in zip(x + (k - 1) * w, ys):
            ax.text(xi, y + 0.05, f"{y:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, ["shift 4", "shift 8", "shift 12"], fontsize=8, color=MUTED); ax.set_ylim(0, 1.1)
    ax.set_title("(b) Caesar accuracy at round 32 (95% CI)", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    # (c) lag over training: colour = condition (as in b), marker = shift, direct labels at round 32
    ax = axes[2]; style(ax); ax.axhline(0, color=MUTED, linewidth=1)
    ends = []
    for E, col, lab in ((J, SLOTS[0], "with MBPP"), (H, MUTED, "half learning rate")):
        for st, mk in (("s4", "o"), ("s8", "s"), ("s12", "^")):
            e1, e2 = RC[f"caesar_{st}_sel"], E[f"caesar_{st}_sel"]
            rs = sorted(set(e1) & set(e2))
            D = [paired_D(e1[r], e2[r])[0] for r in rs]
            ax.plot(rs, D, color=col, marker=mk, markersize=5, linewidth=1.8, markeredgecolor=SURFACE,
                    label=lab if st == "s4" else None)
            ends.append([D[-1], rs[-1], f"shift {st[1:]}"])
    ends.sort()
    for n in range(1, len(ends)):
        ends[n][0] = max(ends[n][0], ends[n - 1][0] + 0.025)
    for y, x0, t in ends:
        ax.annotate(t, (x0, y), xytext=(5, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.set_xlim(-1, 38)
    ax.set_title("(c) Accuracy lost relative to Caesar alone at full rate\n(markers: circle 4, square 8, triangle 12)",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("training round", fontsize=8, color=MUTED); ax.set_ylabel("accuracy lost", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, out, "step_sharing.png", top=0.97)


# ---------------------------------------------------------------- 3. base-9 addition with competitors; terse
def fig_add9_pairs(res, out):
    A1 = load(res, "qis-s02-e1-add9-s1")
    alone_ok = [load(res, f"qis-s02-e3-add9mix-{s}")[0] for s in ("s3r", "s4", "s5")]
    JA, JAR, _ = load(res, "qis-s02-j-add9-add10-s1")
    JT, JTR, _ = load(res, "qis-s02-j-add9-terse-s1")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), facecolor=SURFACE)
    for ax, st in zip(axes[:3], ("d4", "d6", "d8")):
        style(ax)
        rs = sorted(alone_ok[0][f"add9_{st}_sel"])
        M = np.array([[E[f"add9_{st}_sel"][r]["pass1"] for r in rs] for E in alone_ok])
        ax.fill_between(rs, M.min(0), M.max(0), color=MUTED, alpha=0.18, linewidth=0,
                        label="alone, seeds 3-5 (no collapse): range")
        curve(ax, A1[0], f"add9_{st}_sel", 7, label="alone, seed 1 (collapsed)", band=False, color=MUTED, lw=1.5)
        curve(ax, JA, f"add9_{st}_sel", 5, label="with decimal addition (seed 1)", band=False)
        curve(ax, JT, f"add9_{st}_sel", 6, label="with terse (seed 1)", band=False)
        ax.set_title(f"base-9 addition, {st[1:]} digits", fontsize=10, color=INK, loc="left"); ax.set_ylim(-0.02, 1.0)
        ax.set_xlabel("training round", fontsize=8, color=MUTED)
    axes[0].set_ylabel("accuracy (pass@1)", fontsize=8, color=MUTED)
    ax = axes[3]; style(ax)
    curve(ax, JA, "add10_d14_sel", 5, label="decimal addition, trained (with base-9)", band=True)
    curve(ax, A1[0], "add10_d14_sel", 7, label="decimal addition, untrained (base-9 alone)", band=False, color=MUTED, lw=1.5)
    curve(ax, JT, "terse10_k2_sel", 6, label="terse, trained (with base-9)", band=True)
    curve(ax, A1[0], "terse10_k2_sel", 3, label="terse, untrained (base-9 alone)", band=False)
    ax.set_title("the competitor tasks", fontsize=10, color=INK, loc="left"); ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("training round", fontsize=8, color=MUTED); ax.legend(frameon=False, fontsize=6.5, loc="center right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    save(fig, out, "add9_pairs.png")


def fig_terse(res, out):
    groups = [  # (label, slot, [(expid, seed)], trained?)
        ("terse trained with Caesar", 0, [("qis-s02-j-caesar-terse-s1", 1), ("qis-s02-j-caesar-terse-s2", 2), ("qis-s02-j-caesar-terse-s3", 3)]),
        ("terse trained with base-9", 6, [("qis-s02-j-add9-terse-s1", 1)]),
        ("untrained: Caesar alone", 1, [("qis-s02-e1-caesar-s1", 1), ("qis-s02-e1-caesar-s2", 2), ("qis-s02-e1-caesar-s3", 3)]),
        ("untrained: base-9 alone", 2, [("qis-s02-e1-add9-s1", 1)]),
        ("untrained: MBPP alone", 3, [("qis-s02-e1-mbpp-s1", 1)]),
        ("untrained: Caesar + MBPP", 4, [("qis-s02-j-caesar-mbpp-s1", 1)]),
        ("untrained: base-9 + decimal", 5, [("qis-s02-j-add9-add10-s1", 1)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE)
    for label, slot, runs in groups:
        for k, (expid, seed) in enumerate(runs):
            E = load(res, expid)[0]["terse10_k2_sel"]
            rs = sorted(E)
            lab = (label + (f" ({len(runs)} seeds)" if len(runs) > 1 else "")) if k == 0 else None
            for ax, key in ((axes[0], "pass1"), (axes[1], "len_mean")):
                ax.plot(rs, [E[r][key] for r in rs], color=SLOTS[slot], marker=MARKERS[slot], markersize=4,
                        linewidth=2 if "trained with" in label else 1.6, linestyle=STYLES[seed],
                        markeredgecolor=SURFACE, label=lab)
    for ax in axes:
        style(ax); ax.set_xlabel("training round", fontsize=8, color=MUTED)
    axes[0].set_title("terse task accuracy (64 problems x 4)", fontsize=10, color=INK, loc="left"); axes[0].set_ylim(-0.02, 1.0)
    axes[1].set_title("mean terse answer length (tokens); limit = 24", fontsize=10, color=INK, loc="left")
    axes[1].axhline(24, color=MUTED, linewidth=1, linestyle="--"); axes[1].set_yscale("log")
    legend_top(fig, axes[0], ncol=4)
    save(fig, out, "terse_spillover.png", top=0.84)


# ---------------------------------------------------------------- 4. momentum: measured branch data
def fig_momentum_branches(res, out):
    br = [("d6 only, 8 samples", 1, "qis-s02-e3b-r16-d6"), ("d6 only, 32 samples", 2, "qis-s02-e3b-r16-d6n32"),
          ("original mixture", 0, "qis-s02-e3b-r16-mixed")]
    fig = plt.figure(figsize=(14, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    for j, (name, slot, expid) in enumerate(br):
        E, R, S = load(res, expid)
        ax = fig.add_subplot(gs[0, j]); style(ax)
        steps = sorted(S)
        # rounds are 0-based in step records; round index r spans [r, r+1) so the eval "after N rounds" sits at x = N
        x = [S[s]["round"] + S[s]["minibatch"] / 4 for s in steps]
        g = [S[s]["grad_norm_preclip"] for s in steps]
        empty = [S[s]["responses"] == 0 for s in steps]
        ax.bar([xi for xi, e in zip(x, empty) if not e], [gi for gi, e in zip(g, empty) if not e], width=0.2,
               color=SLOTS[slot], align="edge")
        ax.plot([xi + 0.1 for xi, e, gi in zip(x, empty, g) if not e and gi == 0], [0.02] * sum(1 for e, gi in zip(empty, g) if not e and gi == 0),
                linestyle="none", marker="|", color=SLOTS[slot], markersize=7, label="zero-gradient step (AdamW still updates)")
        ax.plot([xi + 0.1 for xi, e in zip(x, empty) if e], [0.02] * sum(empty), linestyle="none", marker="x",
                color=MUTED, markersize=5, label="empty minibatch (no update)")
        ax.set_ylim(0, max(1.6, max(g) * 1.1)); ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlim(16, 24.2); ax.set_xticks(range(16, 25))
        ax.set_title(f"{name}: gradient norm per optimizer step", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("rounds completed (4 steps per round)", fontsize=8, color=MUTED)
        if j == 0:
            ax.set_ylabel("pre-clip gradient norm\n(dashed: clip at 1.0)", fontsize=8, color=MUTED)
            ax.legend(frameon=False, fontsize=7, loc="center left")
        nz = sum(1 for v in g if v > 0)
        ax.text(0.02, 0.92, f"{nz} of {len(g)} steps non-zero", transform=ax.transAxes, fontsize=8, color=INK)
    for k, (panel, title) in enumerate((("add9_d6_conf", "6-digit accuracy (confirmation set)"),
                                        ("add9_d4_sel", "4-digit accuracy (not trained in d6-only branches)"))):
        ax = fig.add_subplot(gs[1, k]); style(ax)
        for name, slot, expid in br:
            curve(ax, load(res, expid)[0], panel, slot, label=name, band=True)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left"); ax.set_xlabel("rounds completed", fontsize=8, color=MUTED)
        ax.set_xticks([16, 20, 24]); ax.set_xlim(16, 24.2)
        if k == 0:
            ax.set_ylabel("accuracy", fontsize=8, color=MUTED); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0, 0.95, "How to read this figure", fontsize=10, color=INK, weight="bold", va="top")
    ax.text(0, 0.80,
            "Top: every optimizer step of each branch (4 per round).\n"
            "Bars: non-zero gradients. Ticks at zero: the gradient was\n"
            "exactly zero, but AdamW still applied a momentum update.\n"
            "Grey x: empty minibatch, no update at all.\n\n"
            "d6 only, 8 samples: no step had any gradient, yet 4-digit\n"
            "accuracy fell from 0.55 to 0.39: the weights moved on\n"
            "momentum carried over from the checkpoint.\n\n"
            "d6 only, 32 samples: one non-zero step (round 20, clipped);\n"
            "6-digit accuracy rose over the next 4 rounds although\n"
            "every later gradient was zero.", fontsize=8, color=INK, va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "momentum_branches.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_caesar_mbpp(a.res, a.out)
    fig_step_sharing(a.res, a.out)
    fig_add9_pairs(a.res, a.out)
    fig_terse(a.res, a.out)
    fig_momentum_branches(a.res, a.out)


if __name__ == "__main__":
    main()
