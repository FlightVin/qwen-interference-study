"""A26 pass B task 4 (stress panels) + check of the response-audit key against stress records (task 6)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import P, boot_diff, correct, coverage_split, decompose, dump, integrity, load_records, metrics

N = 8
PANEL = P + "/data/stress/stress_v1.json"
ITEMS = json.load(open(PANEL))["pools"]["stress"]
ORDER = [it["id"] for it in ITEMS]
TIER = {it["id"]: it["tier"] for it in ITEMS}
SE = P + "/stress-eval"
CELLS = ["base", "H-s101-r128", "R-s101-r128", "U-s101-r128", "H-s202-r128", "R-s202-r128", "U-s202-r128",
         "R-s101-r64", "H-s202-r64"]

out = {"panel_file_items": len(ORDER), "tiers_in_file": {t: sum(1 for x in TIER.values() if x == t) for t in set(TIER.values())},
       "n": N, "cells": {}, "integrity": {}}
R, PP = {}, {}
for name in CELLS:
    recs, dups = load_records(f"{SE}/{name}/records.jsonl.gz", keep_hash=True)
    R[name] = recs
    pids = sorted({p for p, _ in recs})
    integ = integrity(recs, dups, ORDER, N)
    integ["problems_in_records"] = len(pids)
    integ["problems_not_in_file"] = len(set(pids) - set(ORDER))
    out["integrity"][name] = integ
    order = [p for p in ORDER if any((p, j) in recs for j in range(N))]
    c, t = np.zeros(len(order)), np.zeros(len(order))
    for i, p in enumerate(order):
        for j in range(N):
            r = recs.get((p, j))
            if r is None:
                continue
            c[i] += correct(r); t[i] += r["finish"] == "length"
    PP[name] = dict(zip(order, zip(c, t)))
    cell = {"all": metrics(c, t, N)}
    for tier in ("harder", "superhard"):
        m = np.array([TIER[p] == tier for p in order])
        cell[tier] = metrics(c[m], t[m], N)
    cell["capped_not_12288"] = sum(1 for r in recs.values() if r["finish"] == "length" and r["n_tokens"] != 12288)
    out["cells"][name] = cell

common_pids = [p for p in ORDER if all(p in PP[nm] for nm in CELLS)]
out["problems_common_to_all_cells"] = len(common_pids)

seed_mis = sum(len({R[nm][(p, 0)]["seed"] for nm in CELLS}) > 1 for p in common_pids)
out["problems_with_seed_differing_across_cells"] = seed_mis

# shards of H-s202 r128
union, sh = {}, {}
for s in range(3):
    recs, dups = load_records(f"{SE}/H-s202-r128-shards/k{s}/records.jsonl.gz", keep_hash=True)
    pan = [it["id"] for it in json.load(open(f"{SE}/H-s202-r128-shards/k{s}/panel_{s}.json"))["pools"]["stress"]]
    sh[f"k{s}"] = {"panel_items": len(pan), "panel_equals_interleave": pan == ORDER[s::3], "attempts": len(recs),
                   "duplicates": len(dups)}
    union.update(recs)
m = R["H-s202-r128"]
sh["union_equals_merged_keys"] = set(union) == set(m)
sh["identical_attempts"] = sum(all(union[k][f] == m[k][f] for f in ("h", "reward", "finish", "n_tokens", "seed"))
                               for k in m if k in union)
out["shards_H-s202-r128"] = sh


def arr(name, pids):
    c = np.array([PP[name][p][0] for p in pids]); t = np.array([PP[name][p][1] for p in pids])
    return c, t


con = {}
for s in ("s101", "s202"):
    for a, b in (("R", "U"), ("R", "H"), ("U", "H")):
        A, B = f"{a}-{s}-r128", f"{b}-{s}-r128"
        for tier in ("all", "harder", "superhard"):
            pids = [p for p in common_pids if tier == "all" or TIER[p] == tier]
            ca, ta = arr(A, pids); cb, tb = arr(B, pids)
            con[f"{a}-{b} {s} {tier}"] = {"N": len(pids), "pass1": boot_diff(ca / N, cb / N),
                                          "coverage_at_8": boot_diff(ca > 0, cb > 0),
                                          "truncation_share": boot_diff(ta / N, tb / N),
                                          "decomposition": decompose(ca, ta, cb, tb, N),
                                          "coverage_vs_reliability": coverage_split(ca, cb, N)}
for A, B in (("R-s101-r128", "R-s101-r64"), ("H-s202-r128", "H-s202-r64"), ("R-s101-r64", "base"), ("H-s202-r64", "base")):
    ca, ta = arr(A, common_pids); cb, tb = arr(B, common_pids)
    con[f"{A} - {B} all"] = {"N": len(common_pids), "pass1": boot_diff(ca / N, cb / N),
                             "decomposition": decompose(ca, ta, cb, tb, N)}
out["contrasts"] = con

# response-audit key vs stress records (sample 0)
key = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_key.json")))["key"]
mism = []
for code, k in key.items():
    r = R[k["policy"]][(k["item"], 0)]
    if (r["reward"], r["finish"], r["n_tokens"]) != (k["reward"], k["finish"], k["n_tokens"]) or TIER[k["item"]] != k["tier"]:
        mism.append(code)
out["audit_key_check"] = {"entries": len(key), "mismatches_vs_records": len(mism), "mismatch_codes": mism}
dump("eval_stress.json", out)
for nm in CELLS:
    print(nm, json.dumps(out["integrity"][nm]))
