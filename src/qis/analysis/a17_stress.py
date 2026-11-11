"""EXPLORATORY (A17 §5–6): stress-panel results (SuperHard and Harder tiers, stress_v1, n = 8, cap 12,288).

Inputs: PANEL = data/stress/stress_v1.json (pool "stress"; each item has id and tier) and EVAL_ROOT/<label>/records.jsonl.gz
written by qis.evaluate_ckpt (prompt_id, sample 0-7, n_tokens, finish, reward). Labels: base, {H,R,U}-s{101,202}-r128,
R-s101-r64, H-s202-r64. An absent or incomplete cell is written "missing" or "incomplete" and skipped.
Outputs in OUT_DIR (stress.json, stress.md); every delta is written next to both absolute values:
1. cells: per tier (superhard, harder, all) and checkpoint: pass@1, pass@8 (= solved in ≥ 1 of 8), truncation share t
   (finish == "length"), accuracy among finished attempts a, mean tokens, solved counts, 0/8 counts.
2. coverage against base: problems base 0/8 → checkpoint ≥ 1/8 (gained) and base ≥ 1/8 → checkpoint 0/8 (lost), with the
   expected number of chance losses at m = 8 under no change, sum over base-solved problems of (1 − p)^8, p = base c/8.
3. contrasts: R − U, R − H, U − H per seed; r128 − r64 (R-s101, H-s202); checkpoint − base. Paired problem bootstrap
   (10,000 draws; stream sha256("2027/a17-stress/<tier>/<contrast>")) for Δpass@1, Δpass@8 and the exact split
   Δpass@1 = −Δt·ā + (1 − t̄)·Δa, where a = pass@1 / (1 − t) and bars are means of the two checkpoints.
`plot(OUT_DIR, FIG_DIR)` (matplotlib venv) draws stress_panels.png: pass@1 and truncation share per tier and checkpoint.
Usage: python -m qis.analysis.a17_stress PANEL EVAL_ROOT OUT_DIR [--figures FIG_DIR]
"""
import argparse
import gzip
import hashlib
import json
import os

import numpy as np

N, DRAWS = 8, 10_000
LABELS = ["base", "H-s101-r128", "R-s101-r128", "U-s101-r128", "H-s202-r128", "R-s202-r128", "U-s202-r128",
          "R-s101-r64", "H-s202-r64"]
TIERS = ["superhard", "harder", "all"]
CONTRASTS = [("R-s101-r128", "U-s101-r128"), ("R-s202-r128", "U-s202-r128"),
             ("R-s101-r128", "H-s101-r128"), ("U-s101-r128", "H-s101-r128"),
             ("R-s202-r128", "H-s202-r128"), ("U-s202-r128", "H-s202-r128"),
             ("R-s101-r128", "R-s101-r64"), ("H-s202-r128", "H-s202-r64")] + \
            [(x, "base") for x in LABELS[1:]]


def load(d, ids):
    p = os.path.join(d, "records.jsonl.gz")
    if not os.path.exists(p):
        return "missing"
    y, tr, nt = {}, {}, {}
    for line in gzip.open(p, "rt"):
        r = json.loads(line)
        k = (r["prompt_id"], r["sample"])
        y[k], tr[k], nt[k] = r["reward"] == 1, r["finish"] == "length", r["n_tokens"]
    if any((i, s) not in y for i in ids for s in range(N)):
        return f"incomplete: {len({k[0] for k in y} & set(ids))} of {len(ids)} problems"
    arr = lambda m, t: np.array([[m[(i, s)] for s in range(N)] for i in ids], dtype=t)   # noqa: E731
    Y, T = arr(y, bool), arr(tr, bool)
    return {"y": Y, "t": T, "ntok": arr(nt, np.int64), "trunc_correct": int((Y & T).sum())}


def stats(y, t, ntok, ix=None):
    """Pass@1, pass@8, truncation, finished accuracy over the problems in ix (row indices; None = all)."""
    if ix is not None:
        y, t, ntok = y[ix], t[ix], ntok[ix]
    p1, tt = float(y.mean()), float(t.mean())
    return {"pass1": p1, "pass8": float((y.sum(1) > 0).mean()), "trunc": tt, "acc_finished": p1 / (1 - tt) if tt < 1 else None,
            "mean_tokens": float(ntok.mean()), "solved": int((y.sum(1) > 0).sum()), "zero": int((y.sum(1) == 0).sum()),
            "problems": int(y.shape[0])}


def split(pa, ta, pb, tb):
    """Exact split of pa − pb into the truncation term −Δt·ā and the accuracy term (1 − t̄)·Δa."""
    aa, ab = pa / (1 - ta), pb / (1 - tb)
    return -(ta - tb) * (aa + ab) / 2, (1 - (ta + tb) / 2) * (aa - ab)


def contrast(A, B, rows, name):
    """Paired problem bootstrap of A − B over the problems in rows."""
    yA, tA, yB, tB = A["y"][rows], A["t"][rows], B["y"][rows], B["t"][rows]
    cA, cB, kA, kB = yA.mean(1), yB.mean(1), tA.mean(1), tB.mean(1)
    sA, sB = (yA.sum(1) > 0).astype(float), (yB.sum(1) > 0).astype(float)
    seed = int.from_bytes(hashlib.sha256(f"2027/a17-stress/{name}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    n = len(rows)

    def est(ix):
        pa, pb, ta, tb = cA[ix].mean(), cB[ix].mean(), kA[ix].mean(), kB[ix].mean()
        tr, ac = split(pa, ta, pb, tb) if ta < 1 and tb < 1 else (np.nan, np.nan)
        return np.array([pa - pb, sA[ix].mean() - sB[ix].mean(), tr, ac])

    point = est(np.arange(n))
    boot = np.array([est(rng.integers(0, n, n)) for _ in range(DRAWS)])
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    keys = ["d_pass1", "d_pass8", "trunc_term", "acc_term"]
    out = {k: {"d": float(point[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, k in enumerate(keys)}
    out.update({"a_pass1": float(cA.mean()), "b_pass1": float(cB.mean()), "a_pass8": float(sA.mean()), "b_pass8": float(sB.mean()),
                "a_trunc": float(kA.mean()), "b_trunc": float(kB.mean()), "problems": n, "seed_stream": f"2027/a17-stress/{name}"})
    return out


def coverage(P, B, rows):
    cP, cB = P["y"][rows].sum(1), B["y"][rows].sum(1)
    p = cB / N
    return {"base_solved": int((cB > 0).sum()), "ckpt_solved": int((cP > 0).sum()), "gained": int(((cB == 0) & (cP > 0)).sum()),
            "lost": int(((cB > 0) & (cP == 0)).sum()), "expected_chance_lost": float(((1 - p) ** N)[cB > 0].sum()),
            "problems": len(rows)}


def ci(c, f=".3f"):
    return f"{c['d']:+{f}} [{c['lo']:+{f}}, {c['hi']:+{f}}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel"); ap.add_argument("eval_root"); ap.add_argument("out_dir"); ap.add_argument("--figures")
    a = ap.parse_args()
    panel = open(a.panel, "rb").read()
    items = json.loads(panel)["pools"]["stress"]
    ids = [it["id"] for it in items]
    tier_rows = {t: np.array([i for i, it in enumerate(items) if t == "all" or it["tier"] == t]) for t in TIERS}
    cells = {lb: load(os.path.join(a.eval_root, lb), ids) for lb in LABELS}
    have = [lb for lb in LABELS if isinstance(cells[lb], dict)]
    res = {"panel_sha256": hashlib.sha256(panel).hexdigest(), "n": N, "status": {lb: cells[lb] if isinstance(cells[lb], str) else "ok" for lb in LABELS},
           "truncated_and_correct": {lb: cells[lb]["trunc_correct"] for lb in have},
           "tier_sizes": {t: len(r) for t, r in tier_rows.items()}, "cells": {}, "coverage": {}, "contrasts": {}}
    for t, rows in tier_rows.items():
        res["cells"][t] = {lb: stats(cells[lb]["y"], cells[lb]["t"], cells[lb]["ntok"], rows) for lb in have}
        if "base" in have:
            res["coverage"][t] = {lb: coverage(cells[lb], cells["base"], rows) for lb in have if lb != "base"}
        res["contrasts"][t] = {f"{x} vs {y}": contrast(cells[x], cells[y], rows, f"{t}/{x}-{y}")
                               for x, y in CONTRASTS if x in have and y in have}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, "stress.json"), "w"), indent=1)
    md = [f"# Stress panels (EXPLORATORY; stress_v1 sha256 {res['panel_sha256'][:12]}…, n = {N}, cap 12,288)", "",
          "Cell status: " + ", ".join(f"{lb} {s}" for lb, s in res["status"].items()), ""]
    for t in TIERS:
        md += [f"## {t} ({res['tier_sizes'][t]} problems)", "",
               "| Checkpoint | pass@1 | pass@8 | Truncated | Accuracy on finished | Mean tokens | Solved ≥ 1/8 | 0/8 |",
               "|---|---|---|---|---|---|---|---|"]
        for lb, s in res["cells"][t].items():
            af = f"{s['acc_finished']:.3f}" if s["acc_finished"] is not None else "n/a"
            md.append(f"| {lb} | {s['pass1']:.3f} | {s['pass8']:.3f} | {s['trunc']:.1%} | {af} | {s['mean_tokens']:.0f} | "
                      f"{s['solved']}/{s['problems']} | {s['zero']} |")
        if res["coverage"].get(t):
            md += ["", "| Checkpoint | Base solved | Checkpoint solved | Gained (base 0/8) | Lost (base ≥ 1/8) | Expected chance losses |",
                   "|---|---|---|---|---|---|"]
            for lb, c in res["coverage"][t].items():
                md.append(f"| {lb} | {c['base_solved']} | {c['ckpt_solved']} | {c['gained']} | {c['lost']} | {c['expected_chance_lost']:.1f} |")
        md += ["", "| Contrast A vs B | pass@1 A, B | A − B pass@1 [95% CI] | pass@8 A, B | A − B pass@8 | Truncation term | Accuracy term |",
               "|---|---|---|---|---|---|---|"]
        for k, c in res["contrasts"][t].items():
            md.append(f"| {k} | {c['a_pass1']:.3f}, {c['b_pass1']:.3f} | {ci(c['d_pass1'])} | {c['a_pass8']:.3f}, {c['b_pass8']:.3f} | "
                      f"{ci(c['d_pass8'])} | {ci(c['trunc_term'])} | {ci(c['acc_term'])} |")
        md.append("")
    open(os.path.join(a.out_dir, "stress.md"), "w").write("\n".join(md))
    print("\n".join(md[:3]))
    if a.figures:
        plot(a.out_dir, a.figures)


def plot(out_dir, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(out_dir, "stress.json")))
    os.makedirs(fig_dir, exist_ok=True)
    tiers = ["superhard", "harder"]
    labels = [lb for lb in LABELS if all(lb in res["cells"][t] for t in tiers)]
    col = {"b": "#7f7f7f", "H": "#4c72b0", "R": "#dd8452", "U": "#55a868"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    w = 0.8 / max(len(labels), 1)
    for j, (key, title) in enumerate([("pass1", "pass@1 (8 attempts per problem)"), ("trunc", "Share of attempts cut at 12,288 tokens")]):
        ax = axes[j]
        for i, lb in enumerate(labels):
            xs = np.arange(len(tiers)) + (i - (len(labels) - 1) / 2) * w
            vals = [res["cells"][t][lb][key] for t in tiers]
            ax.bar(xs, vals, w, color=col[lb[0]], alpha=0.55 if "r64" in lb else (0.75 if "s202" in lb else 1.0),
                   hatch="//" if "r64" in lb else None, edgecolor="white", label=lb)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([f"{t} ({res['tier_sizes'][t]})" for t in tiers])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[1].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Stress panels (exploratory): base and trained checkpoints", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "stress_panels.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(sys.argv[2], sys.argv[3])
    else:
        main()
