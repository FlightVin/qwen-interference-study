"""EXPLORATORY (A15.1, rules frozen in D122): which sealed-test problems improve, and how reliably.

Input: the sealed-test evaluation directories `<EVAL_DIR>/<name>/records.jsonl.gz` (32 attempts per problem, `sample`
0-31) for qis-test-base and qis-test-{H,R,U}-s{101,202}-sym. No new generations. Output JSON (+ per-problem table):
- per arm and seed: pass@1 (c/32) and pass@8 (1 - C(32-c,8)/C(32,8)), equal weight per problem;
- paired R-U and R-H per seed for both metrics, 95% problem bootstrap (10,000 draws, seed 2027);
- per-problem change from the base model (arm c/32 - base c/32), sorted, with counts of problems improved / unchanged /
  worse and the share of the total gain carried by the top 10% of problems;
- difficulty bands from base samples 0-15 (k of 16): B0 k=0, B1 1-5, B2 6-10, B3 11-15, B4 16; baseline accuracy in a
  band from base samples 16-31 only; per band and arm/seed: size, pass@1, change from the band's baseline;
- counts are written as c/32 ("0/32" for no success).
Usage: python -m qis.analysis.per_problem EVAL_DIR OUT_PREFIX
"""
import gzip
import json
import os
import sys
from math import comb

import numpy as np

from qis import seeds

ARMS, SEEDS, N, DRAWS = ("H", "R", "U"), (101, 202), 32, 10_000
BANDS = (("B0", 0, 0), ("B1", 1, 5), ("B2", 6, 10), ("B3", 11, 15), ("B4", 16, 16))


def pass8(c, n=N, k=8):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def load(path):
    """{prompt_id: [reward of sample 0..31]}."""
    out = {}
    for l in gzip.open(path, "rt"):
        r = json.loads(l)
        out.setdefault(r["prompt_id"], {})[r["sample"]] = r["reward"] == 1
    return {p: [v[i] for i in range(N)] for p, v in out.items() if len(v) == N}


def boot(diff, name):
    D = np.asarray(diff, dtype=np.float64)
    r = seeds.rng(seeds.BOOTSTRAP_SEED, f"a15/{name}")
    idx = r.integers(0, len(D), (DRAWS, len(D)))
    bs = D[idx].mean(axis=1)
    return {"est": float(D.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975))}


def band_of(k):
    return next(b for b, lo, hi in BANDS if lo <= k <= hi)


def main(eval_dir, out):
    base = load(os.path.join(eval_dir, "qis-test-base", "records.jsonl.gz"))
    probs = sorted(base)
    runs = {}
    for a in ARMS:
        for s in SEEDS:
            p = os.path.join(eval_dir, f"qis-test-{a}-s{s}-sym", "records.jsonl.gz")
            if os.path.exists(p):
                runs[(a, s)] = load(p)
                assert sorted(runs[(a, s)]) == probs, f"problem set differs for {a}-s{s}"
    c = {k: {p: sum(v[p]) for p in probs} for k, v in runs.items()}
    cb = {p: sum(base[p]) for p in probs}
    band = {p: band_of(sum(base[p][:16])) for p in probs}
    band_base = {b: [sum(base[p][16:]) / 16 for p in probs if band[p] == b] for b, _, _ in BANDS}
    res = {"n_problems": len(probs), "base": {"pass1": float(np.mean([cb[p] / N for p in probs])),
                                             "pass8": float(np.mean([pass8(cb[p]) for p in probs])),
                                             "zero": sum(cb[p] == 0 for p in probs)},
           "bands": {b: {"size": len(band_base[b]), "baseline_pass1_heldout16": float(np.mean(band_base[b])) if band_base[b] else None,
                         "rule": f"base samples 0-15 correct in [{lo},{hi}] of 16"} for b, lo, hi in BANDS},
           "arms": {}, "paired": {}, "per_problem": {}}
    for (a, s), cc in c.items():
        ch = [cc[p] / N - cb[p] / N for p in probs]
        pos = sorted([x for x in ch if x > 0], reverse=True)
        top = max(1, len(probs) // 10)
        res["arms"][f"{a}-s{s}"] = {
            "pass1": float(np.mean([cc[p] / N for p in probs])), "pass8": float(np.mean([pass8(cc[p]) for p in probs])),
            "zero": sum(cc[p] == 0 for p in probs), "improved": sum(x > 0 for x in ch), "unchanged": sum(x == 0 for x in ch),
            "worse": sum(x < 0 for x in ch), "net_gain": float(np.sum(ch)),
            "top10pct_share_of_positive_gain": float(sum(pos[:top]) / sum(pos)) if pos else None,
            "by_band": {b: {"size": len(band_base[b]),
                            "pass1": float(np.mean([cc[p] / N for p in probs if band[p] == b])) if band_base[b] else None,
                            "change_vs_heldout_baseline": (float(np.mean([cc[p] / N for p in probs if band[p] == b]) - np.mean(band_base[b]))
                                                           if band_base[b] else None)} for b, _, _ in BANDS}}
    for s in SEEDS:
        for name, (a, b) in (("R-U", ("R", "U")), ("R-H", ("R", "H"))):
            if (a, s) in c and (b, s) in c:
                d1 = [c[(a, s)][p] / N - c[(b, s)][p] / N for p in probs]
                d8 = [pass8(c[(a, s)][p]) - pass8(c[(b, s)][p]) for p in probs]
                res["paired"][f"{name}/s{s}"] = {"pass1": boot(d1, f"{name}/s{s}/p1"), "pass8": boot(d8, f"{name}/s{s}/p8")}
            else:
                res["paired"][f"{name}/s{s}"] = "missing"
    for p in probs:
        res["per_problem"][p] = {"band": band[p], "base": f"{cb[p]}/32", **{f"{a}-s{s}": f"{cc[p]}/32" for (a, s), cc in c.items()}}
    json.dump(res, open(out + ".json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("base", "bands", "paired")}, indent=1))
    print(json.dumps({k: {x: v[x] for x in ("pass1", "pass8", "improved", "unchanged", "worse", "zero")} for k, v in res["arms"].items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
