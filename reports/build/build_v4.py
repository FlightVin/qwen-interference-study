#!/usr/bin/env python3
"""Report v4 additions (owner instruction A27): study-design diagram (F0), training signal over time (F15, t10).

Reads only audited pass-B records (research/audit-a26/pass-b/data/train_groups.json); writes reports/tables/v4/ and
reports/figures/v4/. Style constants copied from build_v3.py so arm colours and seed cues match the v3 figures.

Usage: python build_v4.py --root <repo>
"""
import argparse
import json
import os

COLOR = {"H": "#3F5A87", "R": "#E07B12", "U": "#2BB39A", "base": "#222222"}
MARKER = {101: "o", 202: "s"}
LSTYLE = {101: "-", 202: "--"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
RUNS = ["H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202"]


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "axes.edgecolor": "#888888", "axes.linewidth": 0.9, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.2, "lines.markersize": 7, "savefig.dpi": 200,
        "font.family": "DejaVu Sans"})
    return plt


def save(fig, fdir, name):
    os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, name + ".png"), dpi=200, bbox_inches="tight", facecolor="white",
                metadata={"Software": None})


# ---------------------------------------------------------------- F0: study design
def fig_design(plt, fdir):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.0, 6.4))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 64)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="#888888", fs=12, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    box(12, 57.5, 36, 5.5, "Base model: Qwen3-1.7B\nsame start for all six runs", fc="#F2F2F2", fs=13)
    xs = {"H": 1, "R": 21, "U": 41}
    desc = {"H": "8 more\nH_train", "R": "8 related\n(factoring)", "U": "8 unrelated\n(modular)"}
    for arm, x in xs.items():
        arrow(30, 56.6, x + 9, 51.2)
        box(x, 46, 18, 4.6, f"Arm {arm}", fc=COLOR[arm], ec=COLOR[arm], bold=True, tc="white", fs=13)
        box(x, 37.0, 18, 7.4, "slots 0–7\n8 shared\nH_train", fc="#E9EEF6", ec=COLOR["H"], fs=13)
        box(x, 28.0, 18, 7.4, "slots 8–15\n" + desc[arm], fc="white", ec=COLOR[arm], fs=13)
    ax.text(30, 24.3, "Same seed: slots 0–7 identical in all arms.\nR and U: matched pairs in slots 8–15.",
            ha="center", va="center", fontsize=12.5, color=MUTED)
    # comparison brackets
    ax.plot([30.4, 30.4, 50, 50], [21.0, 22.4, 22.4, 21.0], color=INK, lw=1.2)
    ax.text(43, 19.6, "R − U: same hard\nexposure (8,192 hard\nattempts each)", ha="center", va="top", fontsize=12.5)
    ax.plot([10, 10, 30, 30], [21.0, 22.0, 22.0, 21.0], color=INK, lw=1.2)
    ax.text(13, 19.6, "R − H, U − H:\nH has 2× the hard\nattempts (16,384)", ha="center", va="top", fontsize=12.5)
    arrow(30, 11.4, 30, 9.0)
    box(1, 0.2, 58, 8.2, "Same evaluations for every checkpoint:\nsealed test 128 × 32 · dev, J_H, J_R, J_U\n"
        "stress · control · cue", fc="#F2F2F2", fs=12.5)
    save(fig, fdir, "f00_design")
    plt.close(fig)


# ---------------------------------------------------------------- t10 / F15: groups without reward contrast
def group_series(groups, run):
    """Per round: (pool, all_wrong, all_correct, n) for shared slots and for slots 8-15."""
    out = []
    second = "hard_extra" if run.startswith("H") else "aux"
    for row in groups[run]["per_round"]:
        r = row["round_1based"]
        rec = {"round": r}
        for pool in ("hard_common", second):
            aw = row.get(f"{pool}|all_wrong", 0)
            ac = row.get(f"{pool}|all_correct", 0)
            mx = row.get(f"{pool}|mixed", 0)
            assert aw + ac + mx == 8, (run, r, pool)
            rec["shared" if pool == "hard_common" else "slots8_15"] = {"all_wrong": aw, "all_correct": ac, "mixed": mx}
        out.append(rec)
    assert [x["round"] for x in out] == list(range(1, 129)), run
    return out


def build_signal(root, tdir):
    groups = json.load(open(os.path.join(root, "research/audit-a26/pass-b/data/train_groups.json")))["groups"]
    series = {run: group_series(groups, run) for run in RUNS}
    quarters = [(1, 32), (33, 64), (65, 96), (97, 128)]
    table = {}
    for run, s in series.items():
        table[run] = {}
        for where in ("shared", "slots8_15"):
            table[run][where] = []
            for a, b in quarters:
                rows = [x[where] for x in s if a <= x["round"] <= b]
                table[run][where].append({"rounds": [a, b], "groups": 8 * len(rows),
                                          "all_wrong": sum(r["all_wrong"] for r in rows),
                                          "all_correct": sum(r["all_correct"] for r in rows),
                                          "mixed": sum(r["mixed"] for r in rows)})
    os.makedirs(tdir, exist_ok=True)
    json.dump({"source": "research/audit-a26/pass-b/data/train_groups.json", "quarters": table, "per_round": series},
              open(os.path.join(tdir, "t10_signal.json"), "w"), indent=1)
    pool_name = {"H": "extra H_train", "R": "related practice", "U": "unrelated practice"}
    lines = ["# t10 — groups without a within-group reward contrast, per pool and 32-round quarter", "",
             "Each cell: all-wrong / all-correct groups of 256 (8 slots × 32 rounds). A group is the 8 attempts of one "
             "slot in one round. Source: `research/audit-a26/pass-b/data/train_groups.json` (audit pass B).", "",
             "| Run | Slots | Pool | r1–32 | r33–64 | r65–96 | r97–128 | Total of 1,024 |", "|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        for where in ("shared", "slots8_15"):
            q = table[run][where]
            pool = "H_train (shared)" if where == "shared" else pool_name[run[0]]
            cells = [f"{x['all_wrong']} / {x['all_correct']}" for x in q]
            tot = f"{sum(x['all_wrong'] for x in q)} / {sum(x['all_correct'] for x in q)}"
            lines.append(f"| {run} | {'0–7' if where == 'shared' else '8–15'} | {pool} | " + " | ".join(cells)
                         + f" | {tot} |")
    open(os.path.join(tdir, "t10_signal.md"), "w").write("\n".join(lines) + "\n")
    return series


def fig_signal(plt, fdir, series):
    import numpy as np
    bins = [(a, a + 7) for a in range(1, 129, 8)]
    mids = [a + 3.5 for a, _ in bins]
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.6), sharex=True)
    spec = [("slots8_15", "all_correct", "Slots 8–15: share of groups all correct"),
            ("shared", "all_wrong", "Slots 0–7 (shared H_train): share all wrong")]
    for ax, (where, kind, title) in zip(axes, spec):
        for run in RUNS:
            arm, seed = run[0], int(run[-3:])
            y = [np.mean([x[where][kind] / 8 for x in series[run] if a <= x["round"] <= b]) for a, b in bins]
            ax.plot(mids, 100 * np.array(y), color=COLOR[arm], ls=LSTYLE[seed], marker=MARKER[seed], ms=5,
                    label=f"{arm} s{seed}")
        ax.axvline(32.5, color=MUTED, lw=1.0, ls=":")
        ax.text(33.5, 96 if kind == "all_correct" else 38.4, "cap 8,192 → 12,288", fontsize=11, color=MUTED, va="top")
        ax.set_ylim(0, 100 if kind == "all_correct" else 40)
        ax.set_ylabel("% of groups")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[1].set_xlabel("training round (8-round bins)")
    axes[1].set_xticks([0, 32, 64, 96, 128])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, fdir, "f15_signal")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    plt = setup_mpl()
    fdir = os.path.join(a.root, "reports/figures/v4")
    tdir = os.path.join(a.root, "reports/tables/v4")
    fig_design(plt, fdir)
    series = build_signal(a.root, tdir)
    fig_signal(plt, fdir, series)
    print("ok", fdir, tdir)


if __name__ == "__main__":
    main()
