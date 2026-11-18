"""Additional report figures (requested 2026-11-17). Reads harvested metrics.jsonl only; no project code.

Usage: python figs_extra.py --res slurm-agentic-engineering/results --out reports/<ID>/figures [--momentum FILE]
Figures:
  caesar_mbpp_{a-d}.png      (a) MBPP alone vs with Caesar; (b-d) Caesar shift 4/8/12 alone, with MBPP, alone at half lr
  step_sharing_{a,b,c}.png   (a) competitor activity, (b) r32 accuracy by condition, (c) joint-run lag vs half-lr cost
                             (a comparison of magnitudes, not an equivalence)
  add9_pairs_{a-d}.png       (a-c) base-9 addition 4/6/8 digits alone vs with add10 / terse; (d) competitor learning
  terse_spillover_{a,b}.png  terse accuracy (a) and answer length (b) in every run, trained vs untrained
  momentum_branches_{a-e}.png (a-c) per-step gradient norm in the three recovery branches; (d, e) accuracy
  momentum_cpu_{a,b,c}.png   CPU measurement (if --momentum given): weight movement under zero-gradient AdamW steps
Every image is one panel with its own legend and labelled axes.
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
from figs import (SLOTS, MARKERS, STYLES, INK, MUTED, GRID, SURFACE, LBL, ACC, ROUND,  # noqa: E402
                  paired_D, panel, finish)


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


# ---------------------------------------------------------------- 1. Caesar and MBPP, alone and together
def fig_caesar_mbpp(res, out):
    RC, RM, J, H = (load(res, e)[0] for e in ("qis-s02-e1-caesar-s1", "qis-s02-e1-mbpp-s1",
                                               "qis-s02-j-caesar-mbpp-s1", "qis-s02-e1-caesar-lr1e6-s1"))
    fig, ax = panel()
    curve(ax, RM, "mbpp_all_sel", 1, label="MBPP trained alone")
    curve(ax, J, "mbpp_all_sel", 0, label="MBPP trained with Caesar")
    ax.set_ylim(0.4, 0.9)
    finish(fig, ax, out, "caesar_mbpp_a.png", "(a) MBPP", ROUND, ACC)
    for tag, st in zip("bcd", ("s4", "s8", "s12")):
        fig, ax = panel()
        curve(ax, RC, f"caesar_{st}_sel", 1, label="Caesar trained alone")
        curve(ax, J, f"caesar_{st}_sel", 0, label="Caesar trained with MBPP")
        curve(ax, H, f"caesar_{st}_sel", 7, seed=2, label="Caesar trained alone, half learning rate", band=False, color=MUTED)
        ax.set_ylim(-0.02, 1.0)
        finish(fig, ax, out, f"caesar_mbpp_{tag}.png", f"({tag}) Caesar, shift {st[1:]}", ROUND, ACC)


# ---------------------------------------------------------------- 2. update sharing
def fig_step_sharing(res, out):
    RC, _, _ = load(res, "qis-s02-e1-caesar-s1")
    H, _, _ = load(res, "qis-s02-e1-caesar-lr1e6-s1")
    J, JR, _ = load(res, "qis-s02-j-caesar-mbpp-s1")
    # (a) competitor activity: cumulative mixed (gradient-producing) competitor groups
    fig, ax = panel()
    rs = sorted(JR)
    ax.plot(rs, np.cumsum([JR[r]["by_role"]["C"]["mixed"] for r in rs]), color=SLOTS[1], marker="s", markersize=3,
            linewidth=2.2, label="MBPP, trained with Caesar")
    for s in (1, 2, 3):
        _, TR, _ = load(res, f"qis-s02-j-caesar-terse-s{s}")
        rt = sorted(TR)
        ax.plot(rt, np.cumsum([TR[r]["by_role"]["C"]["mixed"] for r in rt]), color=SLOTS[0], linewidth=1.8,
                linestyle=STYLES[s], label=f"terse, trained with Caesar (seed {s})")
    finish(fig, ax, out, "step_sharing_a.png", "(a) Competitor groups that produced gradient", ROUND,
           "cumulative mixed groups (count)", ncol=1)
    # (b) accuracy at round 32 with 95% CI, by condition and shift
    fig, ax = panel()
    conds = [("Caesar alone, full learning rate", RC, 1), ("Caesar with MBPP", J, 0), ("Caesar alone, half learning rate", H, 7)]
    x = np.arange(3); w = 0.26
    for k, (name, E, slot) in enumerate(conds):
        ys, lo, hi = [], [], []
        for st in ("s4", "s8", "s12"):
            rec = E[f"caesar_{st}_sel"][32]; a, b = boot_ci(rec)
            ys.append(rec["pass1"]); lo.append(rec["pass1"] - a); hi.append(b - rec["pass1"])
        ax.bar(x + (k - 1) * w, ys, w * 0.92, color=SLOTS[slot] if slot != 7 else MUTED, label=name,
               yerr=[lo, hi], error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
        for xi, y, h in zip(x + (k - 1) * w, ys, hi):
            ax.text(xi, y + h + 0.02, f"{y:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, ["4", "8", "12"], fontsize=8, color=MUTED); ax.set_ylim(0, 1.1)
    finish(fig, ax, out, "step_sharing_b.png", "(b) Caesar accuracy at round 32 (95% interval)",
           "Caesar shift (letters)", "accuracy at round 32, pass@1")
    # (c) lag over training: colour = condition (as in b), marker = shift, direct labels at round 32
    fig, ax = panel()
    ax.axhline(0, color=MUTED, linewidth=1)
    ends = []
    for E, col, lab in ((J, SLOTS[0], "Caesar with MBPP"), (H, MUTED, "Caesar alone, half learning rate")):
        for st, mk in (("s4", "o"), ("s8", "s"), ("s12", "^")):
            e1, e2 = RC[f"caesar_{st}_sel"], E[f"caesar_{st}_sel"]
            rs = sorted(set(e1) & set(e2))
            D = [paired_D(e1[r], e2[r])[0] for r in rs]
            ax.plot(rs, D, color=col, marker=mk, markersize=5, linewidth=1.8, markeredgecolor=SURFACE,
                    label=f"{lab}, shift {st[1:]}")
            ends.append([D[-1], rs[-1], f"shift {st[1:]}"])
    ends.sort()
    for n in range(1, len(ends)):
        ends[n][0] = max(ends[n][0], ends[n - 1][0] + 0.03)
    for y, x0, t in ends:
        ax.annotate(t, (x0, y), xytext=(5, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.set_xlim(-1, 38)
    finish(fig, ax, out, "step_sharing_c.png", "(c) Accuracy lost vs Caesar alone at full rate", ROUND,
           "accuracy lost (pass@1 difference)", ncol=2)


# ---------------------------------------------------------------- 3. base-9 addition with competitors; terse
def fig_add9_pairs(res, out):
    A1 = load(res, "qis-s02-e1-add9-s1")
    alone_ok = [load(res, f"qis-s02-e3-add9mix-{s}")[0] for s in ("s3r", "s4", "s5")]
    JA, JAR, _ = load(res, "qis-s02-j-add9-add10-s1")
    JT, JTR, _ = load(res, "qis-s02-j-add9-terse-s1")
    for tag, st in zip("abc", ("d4", "d6", "d8")):
        fig, ax = panel()
        rs = sorted(alone_ok[0][f"add9_{st}_sel"])
        M = np.array([[E[f"add9_{st}_sel"][r]["pass1"] for r in rs] for E in alone_ok])
        ax.fill_between(rs, M.min(0), M.max(0), color=MUTED, alpha=0.18, linewidth=0,
                        label="base-9 alone, seeds 3-5 (no collapse): range")
        curve(ax, A1[0], f"add9_{st}_sel", 7, label="base-9 alone, seed 1 (collapsed)", band=False, color=MUTED, lw=1.5)
        curve(ax, JA, f"add9_{st}_sel", 5, label="base-9 with decimal addition (seed 1)", band=False)
        curve(ax, JT, f"add9_{st}_sel", 6, label="base-9 with terse (seed 1)", band=False)
        ax.set_ylim(-0.02, 1.0)
        finish(fig, ax, out, f"add9_pairs_{tag}.png", f"({tag}) Base-9 addition, {st[1:]} digits", ROUND, ACC)
    fig, ax = panel()
    curve(ax, JA, "add10_d14_sel", 5, label="decimal addition, trained (with base-9)", band=True)
    curve(ax, A1[0], "add10_d14_sel", 7, label="decimal addition, untrained (base-9 alone)", band=False, color=MUTED, lw=1.5)
    curve(ax, JT, "terse10_k2_sel", 6, label="terse, trained (with base-9)", band=True)
    curve(ax, A1[0], "terse10_k2_sel", 3, label="terse, untrained (base-9 alone)", band=False)
    ax.set_ylim(-0.02, 1.0)
    finish(fig, ax, out, "add9_pairs_d.png", "(d) The competitor tasks", ROUND, ACC)


def fig_terse(res, out):
    groups = [  # (label, slot, [(expid, seed)])
        ("terse trained with Caesar", 0, [("qis-s02-j-caesar-terse-s1", 1), ("qis-s02-j-caesar-terse-s2", 2), ("qis-s02-j-caesar-terse-s3", 3)]),
        ("terse trained with base-9", 6, [("qis-s02-j-add9-terse-s1", 1)]),
        ("untrained: Caesar alone", 1, [("qis-s02-e1-caesar-s1", 1), ("qis-s02-e1-caesar-s2", 2), ("qis-s02-e1-caesar-s3", 3)]),
        ("untrained: base-9 alone", 2, [("qis-s02-e1-add9-s1", 1)]),
        ("untrained: MBPP alone", 3, [("qis-s02-e1-mbpp-s1", 1)]),
        ("untrained: Caesar + MBPP", 4, [("qis-s02-j-caesar-mbpp-s1", 1)]),
        ("untrained: base-9 + decimal", 5, [("qis-s02-j-add9-add10-s1", 1)]),
    ]
    for tag, key, title, ylabel in (("a", "pass1", "(a) Terse task accuracy", ACC),
                                    ("b", "len_mean", "(b) Mean terse answer length",
                                     "mean length (tokens, log scale)")):
        fig, ax = panel()
        for label, slot, runs in groups:
            for k, (expid, seed) in enumerate(runs):
                E = load(res, expid)[0]["terse10_k2_sel"]
                rs = sorted(E)
                lab = (label + (f" ({len(runs)} seeds; line style = seed)" if len(runs) > 1 else "")) if k == 0 else None
                ax.plot(rs, [E[r][key] for r in rs], color=SLOTS[slot], marker=MARKERS[slot], markersize=4,
                        linewidth=2 if "trained with" in label else 1.6, linestyle=STYLES[seed],
                        markeredgecolor=SURFACE, label=lab)
        if key == "pass1":
            ax.set_ylim(-0.02, 1.0)
        else:
            ax.axhline(24, color=MUTED, linewidth=1, linestyle="--", label="length limit (24 tokens)")
            ax.set_yscale("log")
        finish(fig, ax, out, f"terse_spillover_{tag}.png", title, ROUND, ylabel, ncol=1)


# ---------------------------------------------------------------- 4. momentum: measured branch data
BRANCHES = [("6-digit only, 8 samples per prompt", 1, "qis-s02-e3b-r16-d6"),
            ("6-digit only, 32 samples per prompt", 2, "qis-s02-e3b-r16-d6n32"),
            ("original mixture", 0, "qis-s02-e3b-r16-mixed")]


def fig_momentum_branches(res, out):
    for tag, (name, slot, expid) in zip("abc", BRANCHES):
        E, R, S = load(res, expid)
        fig, ax = panel()
        steps = sorted(S)
        # rounds are 0-based in step records; round index r spans [r, r+1) so the eval "after N rounds" sits at x = N
        x = [S[s]["round"] + S[s]["minibatch"] / 4 for s in steps]
        g = [S[s]["grad_norm_preclip"] for s in steps]
        empty = [S[s]["responses"] == 0 for s in steps]
        ax.bar([xi for xi, e in zip(x, empty) if not e], [gi for gi, e in zip(g, empty) if not e], width=0.2,
               color=SLOTS[slot], align="edge", label="step with non-zero gradient")
        zx = [xi + 0.1 for xi, e, gi in zip(x, empty, g) if not e and gi == 0]
        ax.plot(zx, [0.02] * len(zx), linestyle="none", marker="|", color=SLOTS[slot], markersize=7,
                label="zero-gradient step (AdamW still updates)")
        ex = [xi + 0.1 for xi, e in zip(x, empty) if e]
        ax.plot(ex, [0.02] * len(ex), linestyle="none", marker="x", color=MUTED, markersize=5,
                label="empty minibatch (no update)")
        ax.set_ylim(0, max(1.6, max(g) * 1.1))
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--", label="clipping threshold (1.0)")
        ax.set_xlim(16, 24.2); ax.set_xticks(range(16, 25))
        nz = sum(1 for v in g if v > 0)
        ax.text(0.02, 0.92, f"{nz} of {len(g)} steps non-zero", transform=ax.transAxes, fontsize=8, color=INK)
        finish(fig, ax, out, f"momentum_branches_{tag}.png", f"({tag}) Branch: {name}",
               "rounds completed (4 optimizer steps per round)", "pre-clip gradient norm (L2)", ncol=1)
    for tag, (panel_id, title) in zip("de", (("add9_d6_conf", "(d) 6-digit accuracy (confirmation set)"),
                                             ("add9_d4_sel", "(e) 4-digit accuracy (not trained in 6-digit-only branches)"))):
        fig, ax = panel()
        for name, slot, expid in BRANCHES:
            curve(ax, load(res, expid)[0], panel_id, slot, label=f"branch: {name}", band=True)
        ax.set_xticks([16, 20, 24]); ax.set_xlim(16, 24.2)
        finish(fig, ax, out, f"momentum_branches_{tag}.png", title, "rounds completed", ACC)


# ---------------------------------------------------------------- 4b. momentum: controlled CPU measurement
def fig_momentum_cpu(path, out):
    recs = [json.loads(l) for l in open(path)]
    seq = [r for r in recs if "kind" in r]                      # step records, in execution order
    bounds, k = [], 0
    for name in ("warmup", "zero", "single", "zero_after_single"):
        n = sum(1 for r in seq if r["phase"] == name)
        if n:
            bounds.append((k, n, name)); k += n
    # (a) size of every weight update, in execution order
    fig, ax = panel()
    xs = np.arange(1, len(seq) + 1)
    ax.bar(xs, [r["step_norm"] for r in seq], color=[SLOTS[1] if r["kind"] == "real" else SLOTS[0] for r in seq], width=0.8)
    for start, n, name in bounds:
        ax.axvline(start + 0.5, color=GRID, linewidth=1)
    for start, n, name in bounds:                                # exact AdamW prediction for zero-gradient phases
        if name.startswith("zero"):
            # with g = 0: m <- 0.9 m, v <- 0.999 v; update = lr * mhat / sqrt(vhat), mhat = m/(1-0.9^t), vhat = v/(1-0.999^t)
            pred = [seq[start]["step_norm"]]
            for r in seq[start + 1:start + n]:
                t = r["adam_step_count"]
                ratio = (0.9 / np.sqrt(0.999)) * (1 - 0.9 ** (t - 1)) / (1 - 0.9 ** t) * np.sqrt((1 - 0.999 ** t) / (1 - 0.999 ** (t - 1)))
                pred.append(pred[-1] * ratio)
            ax.plot(np.arange(start + 1, start + 1 + n), pred, color=INK, linewidth=1.2, linestyle="--")
    ax.bar([0], [0], color=SLOTS[1], label="real gradient step")
    ax.bar([0], [0], color=SLOTS[0], label="zero-gradient step")
    ax.plot([], [], color=INK, linestyle="--", label="AdamW prediction with no new gradient")
    ax.set_xlim(0.3, len(seq) + 0.7)
    finish(fig, ax, out, "momentum_cpu_a.png", "(a) Size of each weight update, in order",
           "optimizer step (execution order)", "update size ‖Δθ‖₂ (L2, all weights)", ncol=1)
    # (b) cumulative movement during zero-gradient phases vs the real steps that preceded them
    fig, ax = panel()
    for name, slot, lab in (("zero", 0, "after 6 real steps"), ("zero_after_single", 2, "after 1 real step")):
        ph = [r for r in seq if r["phase"] == name]
        if ph:
            ax.plot([r["step"] for r in ph], [r["cum_from_phase_start"] for r in ph], color=SLOTS[slot], marker="o",
                    markersize=4, linewidth=2, label=f"zero-gradient steps {lab}")
    single = [r for r in seq if r["phase"] == "single"]
    warm = [r for r in seq if r["phase"] == "warmup"]
    if warm:
        ax.axhline(warm[-1]["cum_from_phase_start"], color=SLOTS[1], linewidth=1.2, linestyle="--",
                   label="distance moved by the 6 real steps")
    if single:
        ax.axhline(single[0]["step_norm"], color=SLOTS[1], linewidth=1.2, linestyle=":",
                   label="size of the one real step")
    ax.xaxis.get_major_locator().set_params(integer=True)
    finish(fig, ax, out, "momentum_cpu_b.png", "(b) Total distance moved with zero gradient",
           "zero-gradient steps taken (count)", "distance moved ‖θ − θ₀‖₂ (L2, all weights)", ncol=1)
    # (c) policy change on held-out correct answers
    fig, ax = panel()
    pr = [(i + 1, r) for i, r in enumerate(seq) if isinstance(r.get("probe"), dict)]
    files = sorted({f for _, r in pr for f in r["probe"]})
    first_probe = [r for r in recs if isinstance(r.get("probe"), dict)]
    base = first_probe[0]["probe"] if first_probe else {}
    for f, slot in zip(files, (2, 4)):
        lab = "4-digit" if "d4" in f else "6-digit"
        ax.plot([0] + [i for i, _ in pr], [0.0] + [r["probe"][f] - base[f] for _, r in pr], color=SLOTS[slot], marker="o",
                markersize=4, linewidth=2, label=f"held-out correct {lab} answers")
    for start, n, name in bounds:
        if name in ("warmup", "single"):
            ax.axvspan(start + 0.5, start + n + 0.5, color=SLOTS[1], alpha=0.08, linewidth=0)
    ax.fill_between([], [], color=SLOTS[1], alpha=0.15, label="real gradient steps (shaded)")
    ax.axhline(0, color=MUTED, linewidth=1)
    finish(fig, ax, out, "momentum_cpu_c.png", "(c) Change in log-probability of held-out correct answers",
           "optimizer step (execution order)", "Δ mean log-prob (nats per token)", ncol=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--momentum", default=None, help="momentum_probe output (jsonl)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_caesar_mbpp(a.res, a.out)
    fig_step_sharing(a.res, a.out)
    fig_add9_pairs(a.res, a.out)
    fig_terse(a.res, a.out)
    fig_momentum_branches(a.res, a.out)
    if a.momentum:
        fig_momentum_cpu(a.momentum, a.out)


if __name__ == "__main__":
    main()
