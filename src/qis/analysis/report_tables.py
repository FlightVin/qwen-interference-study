"""A9 primary report table from sealed target_test summaries (run in a CPU job).

Input: a directory of `qis.evaluate_ckpt` summaries (`<name>/summary.json`, each with per_problem c/n), named
`qis-test-{H,R,U}-s{seed}-sym` and `qis-test-base`. Output (JSON + Markdown):
- one row per seed: H, R, U pass@1 (equal weight per problem), paired R−U and R−H with 95% paired bootstrap
  intervals over the 128 problems (seed 2027) — CONDITIONAL ON THESE TRAINED POLICIES (sampling and problem
  variation only, not training variability);
- the mean over seeds with the same conditional problem-resampling interval (problems resampled jointly);
- agreement across seeds: same sign, and |difference| relative to the mean;
- a two-seed t interval (df = 1) shown only with an explicit warning; no claim rests on it.
Missing runs leave empty cells marked "missing" (A9.5 partial delivery).
Usage: python -m qis.analysis.report_tables EVAL_DIR OUT_PREFIX
"""
import json
import os
import sys
from math import comb

import numpy as np

from qis import seeds
from qis.analysis.stats import T975

SEEDS, ARMS, DRAWS = (101, 202), ("H", "R", "U"), 10_000


def pass8(c, n, k=8):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def load(d):
    rates, p8 = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            p = os.path.join(d, f"qis-test-{arm}-s{s}-sym", "summary.json")
            if os.path.exists(p):
                pp = json.load(open(p))["per_problem"]
                rates.setdefault(arm, {})[s] = {x["id"]: x["c"] / x["n"] for x in pp}
                p8.setdefault(arm, {})[s] = {x["id"]: pass8(x["c"], x["n"]) for x in pp}
    b = os.path.join(d, "qis-test-base", "summary.json")
    bpp = json.load(open(b))["per_problem"] if os.path.exists(b) else None
    base = {x["id"]: x["c"] / x["n"] for x in bpp} if bpp else None
    base8 = {x["id"]: pass8(x["c"], x["n"]) for x in bpp} if bpp else None
    return rates, base, p8, base8


def boot(diff_rows, name):
    """diff_rows: seeds x problems paired differences; resample problems (same draw for every seed)."""
    D = np.asarray(diff_rows, dtype=np.float64)
    r = seeds.rng(seeds.BOOTSTRAP_SEED, f"a9/{name}")
    idx = r.integers(0, D.shape[1], (DRAWS, D.shape[1]))
    bs = D[:, idx].mean(axis=(0, 2))
    return {"est": float(D.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975))}


def main(d, out):
    rates, base, p8, base8 = load(d)
    probs = sorted(next(iter(next(iter(rates.values())).values()))) if rates else []
    rows, diffs = [], {"R-U": [], "R-H": [], "U-H": []}
    for s in SEEDS:
        row = {"seed": s}
        for arm in ARMS:
            v = rates.get(arm, {}).get(s)
            row[arm] = float(np.mean([v[p] for p in probs])) if v else "missing"
            v8 = p8.get(arm, {}).get(s)
            row[arm + "_pass8"] = float(np.mean([v8[p] for p in probs])) if v8 else "missing"
        for name, (a, b) in (("R-U", ("R", "U")), ("R-H", ("R", "H")), ("U-H", ("U", "H"))):
            va, vb = rates.get(a, {}).get(s), rates.get(b, {}).get(s)
            if va and vb:
                assert sorted(va) == probs and sorted(vb) == probs, "problem sets must match"
                dr = [va[p] - vb[p] for p in probs]
                row[name] = boot([dr], f"{name}/s{s}"); diffs[name].append(dr)
            else:
                row[name] = "missing"
        rows.append(row)
    mean = {}
    for arm in ARMS:
        vals = [r[arm] for r in rows if r[arm] != "missing"]
        mean[arm] = float(np.mean(vals)) if len(vals) == len(SEEDS) else "missing"
    agree = {}
    for name, D in diffs.items():
        if len(D) == len(SEEDS):
            mean[name] = boot(D, f"{name}/mean")
            per = [float(np.mean(x)) for x in D]
            m, sd = float(np.mean(per)), float(np.std(per, ddof=1))
            agree[name] = {"per_seed": per, "same_sign": bool(np.sign(per[0]) == np.sign(per[1]) and per[0] != 0),
                           "abs_gap": abs(per[0] - per[1]), "gap_over_mean": abs(per[0] - per[1]) / abs(m) if m else None,
                           "t_two_seed_df1": {"lo": m - T975[1] * sd / np.sqrt(2), "hi": m + T975[1] * sd / np.sqrt(2),
                                              "warning": "two seeds, df = 1: not strong evidence about training variability"}}
        else:
            mean[name] = "missing"
    res = {"rows": rows, "mean": mean, "agreement": agree, "n_problems": len(probs),
           "base_pass8": float(np.mean([base8[p] for p in probs])) if base8 and probs else "missing",
           "base_pass1": float(np.mean([base[p] for p in probs])) if base and probs else (float(np.mean(list(base.values()))) if base else "missing"),
           "interval_note": "95% paired bootstrap over the 128 target_test problems (10,000 draws, seed 2027); conditional on these trained policies"}
    json.dump(res, open(out + ".json", "w"), indent=1)
    f = lambda x: "missing" if x == "missing" else f"{x:.3f}"                              # noqa: E731
    g = lambda x: "missing" if x == "missing" else f"{x['est']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"   # noqa: E731
    md = ["| Seed | H | R | U | R − U (95% CI, conditional) | R − H (95% CI, conditional) | U − H (95% CI, conditional) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['seed']} | {f(r['H'])} | {f(r['R'])} | {f(r['U'])} | {g(r['R-U'])} | {g(r['R-H'])} | {g(r['U-H'])} |")
    md.append(f"| **Mean** | {f(mean['H'])} | {f(mean['R'])} | {f(mean['U'])} | {g(mean['R-U'])} | {g(mean['R-H'])} | {g(mean['U-H'])} |")
    md += ["", f"Base model pass@1 {f(res['base_pass1'])}, pass@8 {f(res['base_pass8'])}.", "",
           "| Seed | H pass@8 | R pass@8 | U pass@8 |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['seed']} | {f(r['H_pass8'])} | {f(r['R_pass8'])} | {f(r['U_pass8'])} |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(json.dumps(agree, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
