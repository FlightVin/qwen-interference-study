"""Analysis of study s02 runs from harvested metrics (numpy + matplotlib only; no model code).

Reads <RES>/<expid>/metrics.jsonl (harvested by `ae harvest`). Writes to --out:
  evals.json      pass@1 per run x panel x round, with per-problem counts
  transfer.json   E1: for each single-task run X and panel Y, delta pass@1 = p_Y(X, r) - p_Y(X, 0), with a 95% bootstrap
                  interval over problems (problems are the sampling unit; responses of one problem are not independent)
  pairs.json      joint vs reference: D(t) = p_T(R, t) - p_T(J, t) per panel (paired by problem; same panels, seeds)
  groups.json     per-round, per-role A0 / mixed / A1 group counts and success
  figures/*.png   learning curves, transfer matrix, group dynamics
Usage: python analyze.py --res RES --out DIR [--runs a,b,...] [--pairs J:R,J:R]
"""
import argparse
import json
import os

import numpy as np

B = 2000
RNG = np.random.default_rng(20261116)


def load(res, expid):
    p = os.path.join(res, expid, "metrics.jsonl")
    ev, rounds = {}, []
    for line in open(p):
        r = json.loads(line)
        if r["kind"] == "eval" and "per_problem" in r and r.get("panel") != "probe_J_H":
            ev.setdefault(r["panel"], {})[r["round"]] = {x["id"]: (x["c"], x["n"]) for x in r["per_problem"]}
        elif r["kind"] == "round":
            rounds.append(r)
    return ev, rounds


def p1(d):
    return float(np.mean([c / n for c, n in d.values()]))


def boot_delta(a, b):
    """mean over shared problems of (b - a) success rates; 95% percentile bootstrap over problems."""
    ids = sorted(set(a) & set(b))
    x = np.array([b[i][0] / b[i][1] - a[i][0] / a[i][1] for i in ids])
    if len(x) == 0:
        return None
    bs = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return {"delta": float(x.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "problems": len(x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    runs = [r for r in a.runs.split(",") if r] or sorted(
        d for d in os.listdir(a.res) if d.startswith("qis-s02-e") and os.path.exists(os.path.join(a.res, d, "metrics.jsonl")))
    data = {r: load(a.res, r) for r in runs}
    evals = {r: {pn: {str(k): p1(v) for k, v in sorted(rr.items())} for pn, rr in ev.items()} for r, (ev, _) in data.items()}
    json.dump(evals, open(os.path.join(a.out, "evals.json"), "w"), indent=1)
    transfer = {}
    for r, (ev, _) in data.items():
        for pn, rr in ev.items():
            if 0 not in rr:
                continue
            transfer.setdefault(r, {})[pn] = {str(k): boot_delta(rr[0], v) for k, v in sorted(rr.items()) if k}
    json.dump(transfer, open(os.path.join(a.out, "transfer.json"), "w"), indent=1)
    groups = {r: [{"round": x["round"], **{k: v for k, v in x["by_role"].items()}, "len_mean": x["len_mean"],
                   "t_round": x["t_round"]} for x in rounds] for r, (_, rounds) in data.items()}
    json.dump(groups, open(os.path.join(a.out, "groups.json"), "w"), indent=1)
    pairs = {}
    for pr in [p for p in a.pairs.split(",") if p]:
        j, ref = pr.split(":")
        if j not in data or ref not in data:
            continue
        ej, er = data[j][0], data[ref][0]
        pairs[pr] = {pn: {str(t): boot_delta(ej[pn][t], er[pn][t]) for t in sorted(set(ej[pn]) & set(er.get(pn, {})))}
                     for pn in ej if pn in er}
    json.dump(pairs, open(os.path.join(a.out, "pairs.json"), "w"), indent=1)
    try:
        figures(a.out, evals, transfer, groups)
    except ImportError:
        pass
    for r in runs:
        last = {pn: v for pn, v in transfer.get(r, {}).items()}
        line = []
        for pn, v in sorted(last.items()):
            if v:
                k = max(v, key=int)
                d = v[k]
                line.append(f"{pn.replace('_sel', '')}:{d['delta']:+.3f}[{d['lo']:+.3f},{d['hi']:+.3f}]@{k}")
        print(r, " ".join(line))


def figures(out, evals, transfer, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = sorted(evals)
    panels = sorted({pn for r in runs for pn in evals[r]})
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.2), squeeze=False, sharey=True)
    for ax, r in zip(axes[0], runs):
        for pn in panels:
            if pn in evals[r]:
                xs = sorted(int(k) for k in evals[r][pn])
                ax.plot(xs, [evals[r][pn][str(k)] for k in xs], marker=".", label=pn.replace("_sel", ""))
        ax.set_title(r.replace("qis-s02-", ""), fontsize=8)
        ax.set_xlabel("round")
    axes[0][0].set_ylabel("pass@1 (sel panels)")
    axes[0][-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "curves.png"), dpi=130)
    plt.close(fig)
    M = np.full((len(runs), len(panels)), np.nan)
    for i, r in enumerate(runs):
        for j, pn in enumerate(panels):
            v = transfer.get(r, {}).get(pn)
            if v:
                M[i, j] = v[max(v, key=int)]["delta"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(panels), 1.0 + 0.5 * len(runs)))
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(panels)), [p.replace("_sel", "") for p in panels], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(runs)), [r.replace("qis-s02-", "") for r in runs], fontsize=7)
    for i in range(len(runs)):
        for j in range(len(panels)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Δ pass@1 vs round 0 (last eval)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "transfer.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
