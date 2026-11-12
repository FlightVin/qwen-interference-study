#!/usr/bin/env python3
"""Compact figures for reports/report-condensed.md. Reads only audited tables (reports/tables/v3, v4).

Usage: python build_condensed.py --root <repo>   (needs matplotlib)
"""
import argparse
import json
import os

COLOR = {"H": "#3F5A87", "R": "#E07B12", "U": "#2BB39A"}
MARK = {101: "o", 202: "s"}
LS = {101: "-", 202: "--"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888888", "text.color": INK,
                         "axes.labelcolor": INK, "lines.linewidth": 1.8, "font.family": "DejaVu Sans"})
    return plt


def save(fig, d, name):
    fig.savefig(os.path.join(d, name + ".png"), dpi=170, bbox_inches="tight", facecolor="white",
                metadata={"Software": None})


def load(root, rel):
    return json.load(open(os.path.join(root, rel)))


def fig_endpoints(plt, root, d):
    rows = {r["cell"]: r for r in load(root, "reports/tables/v3/t01_endpoints.json")["rows"]}
    con = load(root, "reports/tables/v3/t02_contrasts_test.json")["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.85})
    base = 100 * rows["base"]["pass1"]
    a.axhline(base, color=MUTED, lw=1, ls=":")
    a.text(-0.5, base + 1.2, f"base model {base:.1f}%", color=MUTED, fontsize=9.5, ha="left")
    for j, arm in enumerate("HRU"):
        for s, dx in ((101, -0.12), (202, 0.12)):
            v = 100 * rows[f"{arm}-s{s}"]["pass1"]
            a.plot(j + dx, v, MARK[s], color=COLOR[arm], ms=7, mfc=COLOR[arm] if s == 101 else "white", mew=1.8)
            a.text(j + dx + (0.1 if s == 202 else -0.1), v, f"{v:.1f}", fontsize=8.5, va="center",
                   ha="left" if s == 202 else "right")
    a.set_xticks(range(3), ["H", "R", "U"])
    a.set_xlim(-0.6, 2.75)
    a.set_ylim(45, 88)
    a.set_ylabel("sealed-test pass@1 (%)")
    a.set_title("Scores after 128 rounds", loc="left")
    a.grid(axis="y", color=GRID, lw=0.7)
    labels = []
    for i, r in enumerate(con[:6]):
        y = 5 - i
        dv, lo, hi = 100 * r["diff"], 100 * r["ci95"][0], 100 * r["ci95"][1]
        col = INK
        b.plot([lo, hi], [y, y], color=col, lw=1.6)
        b.plot(dv, y, MARK[r["seed"]], color=col, ms=6.5, mfc=col if r["seed"] == 101 else "white", mew=1.6)
        b.text(hi + 0.4, y, f"{dv:+.1f}".replace("-", "−"), fontsize=9, va="center")
        labels.append(f"{r['contrast'].replace('-', ' − ')}, seed {r['seed']}")
    b.axvline(0, color=MUTED, lw=1)
    b.set_yticks(range(5, -1, -1), labels)
    b.set_xlabel("difference (points), 95% interval")
    b.set_xlim(-9, 13)
    b.set_title("Paired differences", loc="left")
    b.grid(axis="x", color=GRID, lw=0.7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=INK, label="seed 101"),
         Line2D([], [], marker="s", ls="", color=INK, mfc="white", label="seed 202")]
    a.legend(handles=h, loc="lower right", frameon=False, fontsize=9)
    save(fig, d, "c1_endpoints")
    plt.close(fig)


def fig_dev(plt, root, d):
    rows = [r for r in load(root, "reports/tables/v3/t03_curves.json")["rows"] if r["panel"] == "dev"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, s in zip(axes, (101, 202)):
        for arm in "HRU":
            pts = sorted((r["round"], 100 * r["pass1"]) for r in rows if r["run"] == f"{arm}-s{s}")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[arm], ls=LS[s], marker=MARK[s], ms=4.5,
                    label=arm)
        ax.axvline(32, color=MUTED, lw=1, ls=":")
        ax.text(34, 41, "cap 8,192 → 12,288", fontsize=9, color=MUTED)
        ax.set_title(f"Seed {s}", loc="left")
        ax.set_xticks([0, 32, 64, 96, 128])
        ax.set_xlabel("training round")
        ax.grid(axis="y", color=GRID, lw=0.7)
    axes[0].set_ylabel("dev pass@1 (%)")
    axes[0].legend(frameon=False, loc="upper left", ncol=3, fontsize=9.5)
    axes[0].set_ylim(38, 80)
    save(fig, d, "c2_dev")
    plt.close(fig)


def fig_saturation(plt, root, d):
    series = load(root, "reports/tables/v4/t10_signal.json")["per_round"]
    import numpy as np
    bins = [(a, a + 7) for a in range(1, 129, 8)]
    mids = [a + 3.5 for a, _ in bins]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for run, s in series.items():
        arm, seed = run[0], int(run[-3:])
        y = [100 * np.mean([x["slots8_15"]["all_correct"] / 8 for x in s if a <= x["round"] <= b]) for a, b in bins]
        ax.plot(mids, y, color=COLOR[arm], ls=LS[seed], marker=MARK[seed], ms=3.5, label=f"{arm} s{seed}")
    ax.axvline(32.5, color=MUTED, lw=1, ls=":")
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_xlabel("training round")
    ax.set_ylabel("% of groups\nall 8 correct")
    ax.set_title("Slots 8–15: how often all 8 answers were already correct (no learning signal)", loc="left",
                 fontsize=10.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower right")
    save(fig, d, "c3_saturation")
    plt.close(fig)


def fig_length(plt, root, d):
    rows = [r for r in load(root, "reports/tables/v3/t05_length.json")["rows"] if r["panel"] == "test"]
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    for i, r in enumerate(rows):
        arm, seed = r["run"][0], int(r["run"][-3:])
        a, b = 100 * r["pass1_12288"], 100 * r["pass1_16384"]
        y = len(rows) - 1 - i
        ax.plot([a, b], [y, y], color=COLOR[arm], lw=2)
        ax.plot(a, y, MARK[seed], color=COLOR[arm], mfc="white", ms=6.5, mew=1.6)
        ax.plot(b, y, MARK[seed], color=COLOR[arm], ms=6.5)
        ax.text(b + 0.3, y, f"+{b - a:.1f}", fontsize=9, va="center")
    ax.set_yticks(range(len(rows) - 1, -1, -1), [r["run"] for r in rows])
    ax.set_xlabel("sealed-test pass@1 (%): open = 12,288-token limit, filled = 16,384")
    ax.set_xlim(73, 87.5)
    ax.grid(axis="x", color=GRID, lw=0.7)
    save(fig, d, "c4_length")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = ap.parse_args().root
    d = os.path.join(root, "reports/figures/condensed")
    os.makedirs(d, exist_ok=True)
    plt = mpl()
    fig_endpoints(plt, root, d)
    fig_dev(plt, root, d)
    fig_saturation(plt, root, d)
    fig_length(plt, root, d)
    print("ok", d)


if __name__ == "__main__":
    main()
