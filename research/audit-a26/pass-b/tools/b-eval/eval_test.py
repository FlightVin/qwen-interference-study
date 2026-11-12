"""A26 pass B tasks 1-3 and 7d: sealed test from per-attempt records only."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P, boot_diff, passk, correct, coverage_split, decompose, dump, integrity, load_records, metrics,
                    panel_ids)

N = 32
ORDER = panel_ids(P + "/data/main-v1/target_test_sealed.json", "test")
TE = P + "/test-eval"
CELLS = {
    "base": TE + "/qis-test-base",
    "H-s101": TE + "/qis-test-H-s101-sym", "R-s101": TE + "/qis-test-R-s101-sym", "U-s101": TE + "/qis-test-U-s101-sym",
    "H-s202": TE + "/qis-test-H-s202-sym", "R-s202": TE + "/qis-test-R-s202-sym", "U-s202": TE + "/qis-test-U-s202-sym",
    "H-s202-fb-r116": TE + "/qis-test-H-s202-sym-fallback",
    "cgpu-R-s101": P + "/test-eval-cgpu/qis-cgpu-R-s101", "cgpu-R-s202": P + "/test-eval-cgpu/qis-cgpu-R-s202",
    "cgpu-U-s101": P + "/test-eval-cgpu/qis-cgpu-U-s101", "cgpu-U-s202": P + "/test-eval-cgpu/qis-cgpu-U-s202",
}

out = {"panel_problems": len(ORDER), "n": N, "cells": {}, "integrity": {}}
R, PP = {}, {}
for name, d in CELLS.items():
    recs, dups = load_records(d + "/records.jsonl.gz", keep_hash=True)
    R[name] = recs
    out["integrity"][name] = integrity(recs, dups, ORDER, N)
    c, t = np.zeros(len(ORDER)), np.zeros(len(ORDER))
    for i, p in enumerate(ORDER):
        for j in range(N):
            r = recs[(p, j)]
            c[i] += correct(r); t[i] += r["finish"] == "length"
    PP[name] = (c, t)
    m = metrics(c, t, N)
    m["max_n_tokens"] = max(r["n_tokens"] for r in recs.values())
    m["capped_not_12288"] = sum(1 for r in recs.values() if r["finish"] == "length" and r["n_tokens"] != 12288)
    out["cells"][name] = m

# seed consistency: same seed per problem across all cells (same panel key, label, n)
seed_mismatch = 0
for p in ORDER:
    s = {R[nm][(p, 0)]["seed"] for nm in CELLS}
    seed_mismatch += len(s) > 1
out["problems_with_seed_differing_across_cells"] = seed_mismatch

# shards: merged == union of shard records, per attempt
def shard_check(root, merged, k):
    res = {"shards": {}, "attempts_compared": 0, "identical_hash_and_fields": 0}
    union = {}
    for s in range(k):
        sd = f"{root}/k{s}"
        recs, dups = load_records(sd + "/records.jsonl.gz", keep_hash=True)
        pan = panel_ids(f"{sd}/panel_{s}.json", "test")
        res["shards"][f"k{s}"] = {"problems_in_panel_file": len(pan), "problems_in_records": len({p for p, _ in recs}),
                                  "attempts": len(recs), "duplicates": len(dups),
                                  "panel_equals_interleave": pan == ORDER[s::k]}
        for kk, v in recs.items():
            assert kk not in union, f"{kk} in two shards"
            union[kk] = v
    res["union_attempts"] = len(union)
    res["union_keys_equal_merged"] = set(union) == set(merged)
    for kk, v in merged.items():
        u = union.get(kk)
        if u is None:
            continue
        res["attempts_compared"] += 1
        res["identical_hash_and_fields"] += all(u[f] == v[f] for f in ("h", "reward", "finish", "n_tokens", "seed", "status"))
    return res, union

out["shards_H-s202"], un202 = shard_check(TE + "/qis-test-H-s202-sym-shards", R["H-s202"], 6)
out["shards_H-s202-fb-r116"], _ = shard_check(TE + "/qis-test-H-s202-sym-fallback-shards", R["H-s202-fb-r116"], 6)

# 7d trace: problem ORDER[3] -> shard k3
pid = ORDER[3]
recs3, _ = load_records(TE + "/qis-test-H-s202-sym-shards/k3/records.jsonl.gz")
c_sh = sum(correct(recs3[(pid, j)]) for j in range(N))
c_mg = sum(correct(R["H-s202"][(pid, j)]) for j in range(N))
out["trace_d"] = {"problem": pid, "panel_index": 3, "shard": "k3 (index 3 mod 6)",
                  "shard_attempts": sum(1 for (p, _) in recs3 if p == pid), "shard_correct": c_sh,
                  "shard_truncated": sum(recs3[(pid, j)]["finish"] == "length" for j in range(N)),
                  "merged_correct": c_mg, "per_problem_pass1": c_mg / N,
                  "contribution_to_cell_pass1": c_mg / N / len(ORDER),
                  "cell_pass1": out["cells"]["H-s202"]["pass1"]}

# 7a trace: first correct attempt of base in panel order
for p in ORDER:
    hit = [j for j in range(N) if R["base"][(p, j)]["reward"] == 1]
    if hit:
        r = R["base"][(p, hit[0])]
        out["trace_a"] = {"cell": "base", "problem": p, "sample": hit[0], "record": {k: r[k] for k in r if k != "h"},
                          "problem_correct_of_32": int(PP["base"][0][ORDER.index(p)])}
        break

# contrasts per seed
con = {}
for s in ("s101", "s202"):
    for a, b in (("R", "U"), ("R", "H"), ("U", "H")):
        A, B = f"{a}-{s}", f"{b}-{s}"
        ca, ta = PP[A]; cb, tb = PP[B]
        con[f"{a}-{b} {s}"] = {
            "A": A, "B": B, "pass1_A": out["cells"][A]["pass1"], "pass1_B": out["cells"][B]["pass1"],
            "pass1": boot_diff(ca / N, cb / N),
            "pass8_unbiased": boot_diff([passk(N, int(x), 8) for x in ca],
                                        [passk(N, int(x), 8) for x in cb]),
            "coverage_at_32": boot_diff(ca > 0, cb > 0),
            "truncation_share": boot_diff(ta / N, tb / N),
            "decomposition": decompose(ca, ta, cb, tb, N),
            "coverage_vs_reliability": coverage_split(ca, cb, N)}
    for a in ("H", "R", "U"):
        A = f"{a}-{s}"
        ca, _ = PP[A]; cb, _ = PP["base"]
        con[f"{a}-base {s}"] = {"pass1": boot_diff(ca / N, cb / N)}
# sensitivity: fallback r116 in place of H-s202 r128
for a in ("R", "U"):
    ca, ta = PP[f"{a}-s202"]; cb, tb = PP["H-s202-fb-r116"]
    con[f"{a}-Hfb116 s202"] = {"pass1": boot_diff(ca / N, cb / N), "decomposition": decompose(ca, ta, cb, tb, N)}
# common-GPU contrast: R-U with both reruns on L40S
for s in ("s101", "s202"):
    ca, ta = PP[f"cgpu-R-{s}"]; cb, tb = PP[f"cgpu-U-{s}"]
    con[f"cgpuR-cgpuU {s}"] = {"pass1": boot_diff(ca / N, cb / N), "decomposition": decompose(ca, ta, cb, tb, N)}
out["contrasts"] = con
out["two_seed_mean"] = {k: (con[f"{k} s101"]["pass1"]["diff"] + con[f"{k} s202"]["pass1"]["diff"]) / 2
                        for k in ("R-U", "R-H", "U-H")}
out["two_seed_halfrange"] = {k: abs(con[f"{k} s101"]["pass1"]["diff"] - con[f"{k} s202"]["pass1"]["diff"]) / 2
                             for k in ("R-U", "R-H", "U-H")}

# reruns vs originals, per attempt
rr = {}
for a in ("R", "U"):
    for s in ("s101", "s202"):
        o, c = R[f"{a}-{s}"], R[f"cgpu-{a}-{s}"]
        keys = sorted(o)
        same_h = sum(o[k]["h"] == c[k]["h"] for k in keys)
        same_r = sum(correct(o[k]) == correct(c[k]) for k in keys)
        both_c = sum(correct(o[k]) == 1 and correct(c[k]) == 1 for k in keys)
        flip_up = sum(correct(o[k]) == 0 and correct(c[k]) == 1 for k in keys)
        flip_dn = sum(correct(o[k]) == 1 and correct(c[k]) == 0 for k in keys)
        same_seed = sum(o[k]["seed"] == c[k]["seed"] for k in keys)
        co, _ = PP[f"{a}-{s}"]; cc, _ = PP[f"cgpu-{a}-{s}"]
        rr[f"{a}-{s}"] = {"attempts": len(keys), "identical_responses": same_h, "identical_reward": same_r,
                          "both_correct": both_c, "orig0_rerun1": flip_up, "orig1_rerun0": flip_dn,
                          "same_seed": same_seed, "pass1_orig": out["cells"][f"{a}-{s}"]["pass1"],
                          "pass1_rerun": out["cells"][f"cgpu-{a}-{s}"]["pass1"],
                          "rerun_minus_orig": boot_diff(cc / N, co / N),
                          "problems_identical_count": int(np.sum(co == cc))}
out["reruns"] = rr
dump("eval_test.json", out)
print(json.dumps({k: {kk: (vv if not isinstance(vv, float) else round(vv, 4)) for kk, vv in v.items()}
                  for k, v in out["cells"].items()}, indent=0)[:4000])
print(json.dumps(out["integrity"], indent=0)[:3000])
