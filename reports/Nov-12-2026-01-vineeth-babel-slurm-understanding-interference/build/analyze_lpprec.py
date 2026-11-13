"""Precision test for Section 1.6 of 01-understanding.md.

Input: harvested folder of qis-ui-lpprec: <P>/lp.npz and <P>/meta.json for P in fp32, fp32-wround, train
(written by src/qis/interf/logp_path.py --precision P).
  fp32         FP32 weights, FP32 compute
  fp32-wround  weights rounded to BF16 and stored as FP32, FP32 compute
  train        FP32 master weights under BF16 autocast (the training forward pass)
Run: python build/analyze_lpprec.py <harness results dir>      (from reports/<ID>/)
Output: tables/lpprec.json

For each precision, run and alpha:  fall = -mean_i [lp_i(run@alpha) - lp_i(base)] over all base-policy answers, and
S = mean over mixed groups of (1/8) sum_i A_i [lp_i(run@alpha) - lp_i(base)] (as in analyze_lp.py).
Ratios are taken against alpha = 1 of the same precision. Intervals: bootstrap over problems, 1000 draws, 95%.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(sys.argv[1], "qis-ui-lpprec")
out = {}
rng = np.random.default_rng(0)
for P in ("fp32", "fp32-wround", "train"):
    z = np.load(os.path.join(D, P, "lp.npz")); conds = [str(c) for c in z["conds"]]; lp = z["lp"]
    meta = json.load(open(os.path.join(D, P, "meta.json")))
    rew = np.array([float(m["reward"]) for m in meta])
    gid = np.array([m["label"] + "|" + m["prompt_id"] for m in meta])
    keys = sorted(set(gid)); ix = [np.where(gid == g)[0] for g in keys]
    adv = np.zeros(len(meta)); mixed = []
    for j, i in enumerate(ix):
        r = rew[i]
        if 0 < r.sum() < len(i):
            adv[i] = (r - r.mean()) / (r.std() + 1e-6); mixed.append(j)
    d = lp - lp[conds.index("base@0")][None, :]
    fall_g = np.stack([[-d[c, i].mean() for i in ix] for c in range(len(conds))])            # [cond, group]
    S_g = np.stack([[(adv[ix[j]] * d[c, ix[j]]).mean() for j in mixed] for c in range(len(conds))])
    res = {"n_answers": len(meta), "n_groups": len(keys), "n_mixed": len(mixed), "base_mean_lp": float(lp[0].mean()), "runs": {}}
    for run in ("R-s101-r128", "mean6"):
        c1 = conds.index(f"{run}@1"); rr = {}
        bi = rng.integers(0, len(keys), (1000, len(keys))); bm = rng.integers(0, len(mixed), (1000, len(mixed)))
        for a in ("0.25", "0.5", "1", "2"):
            c = conds.index(f"{run}@{a}")
            fr = fall_g[c].mean() / fall_g[c1].mean()
            b = fall_g[c][bi].mean(1) / fall_g[c1][bi].mean(1)
            rr[a] = {"fall": float(fall_g[c].mean()), "fall_ratio": float(fr), "fall_ratio_ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                     "S": float(S_g[c].mean()), "S_ci": [float(np.percentile(S_g[c][bm].mean(1), 2.5)), float(np.percentile(S_g[c][bm].mean(1), 97.5))],
                     "S_ratio": float(S_g[c].mean() / S_g[c1].mean())}
        res["runs"][run] = rr
    out[P] = res
    print(P, "answers", len(meta), "groups", len(keys), "mixed", len(mixed), "base lp", round(res["base_mean_lp"], 5))
    for run, rr in res["runs"].items():
        print("  ", run, " | ".join(f"a={a} fall={v['fall']:.5f} ratio={v['fall_ratio']:.3f} [{v['fall_ratio_ci'][0]:.3f},{v['fall_ratio_ci'][1]:.3f}] S={v['S']:+.5f} [{v['S_ci'][0]:+.5f},{v['S_ci'][1]:+.5f}] Sr={v['S_ratio']:.2f}" for a, v in rr.items()))
json.dump(out, open(os.path.join(HERE, "tables", "lpprec.json"), "w"), indent=1)
