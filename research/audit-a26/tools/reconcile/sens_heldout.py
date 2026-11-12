"""A26 reconcile: sensitivity checks (a) close variants of training problems in held-out sets, (c) verifier-unknown,
and a tranche-1 bootstrap seed-variability check. Records only; stdlib + numpy; no project code.

Run on a compute node:  python sens_heldout.py OUT_JSON
Reads /data/user_data/vineethb/qis (read-only)."""
import gzip
import json
import math
import sys

import numpy as np

P = "/data/user_data/vineethb/qis"
OUT = sys.argv[1]
SEED = 20261112          # reconcile's own bootstrap seed
B = 10000
TEST_DROP = "intermediate_algebra/test/00383"   # close variant of H_train intermediate_algebra/train/00301 (J3m 0.725)
TEST_DROP2 = "intermediate_algebra/test/00891"  # J3m 0.500 to H_train intermediate_algebra/train/00048 (shared phrasing)
DEV_DROP = "algebra/train/01291"                # numeric variant of H_train algebra/train/01269 (J3m 1.0)


def rows(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def counts(path):
    """{pid: [correct, attempts, unknown]} from a records file (unknown counts as incorrect)."""
    c = {}
    for r in rows(path):
        x = c.setdefault(r["prompt_id"], [0, 0, 0])
        x[0] += 1 if r["reward"] == 1 else 0
        x[1] += 1
        x[2] += 1 if r["status"] == "unknown" else 0
    return c


def pass1(c, ids):
    return float(np.mean([c[p][0] / c[p][1] for p in ids]))


def pass8(c, ids):
    v = []
    for p in ids:
        k, n = c[p][0], c[p][1]
        v.append(1.0 - (math.comb(n - k, 8) / math.comb(n, 8) if n - k >= 8 else 0.0))
    return float(np.mean(v))


def boot(ca, cb, ids, seed=SEED):
    d = np.array([ca[p][0] / ca[p][1] - cb[p][0] / cb[p][1] for p in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "N": len(d)}


out = {"seed": SEED, "draws": B}

# ---------- (a) sealed test without the close variant(s) ----------
TEST_IDS = [it["id"] for it in json.load(open(P + "/data/main-v1/target_test_sealed.json"))["pools"]["test"]]
assert len(TEST_IDS) == 128 and TEST_DROP in TEST_IDS and TEST_DROP2 in TEST_IDS
CELLS = {"base": "test-eval/qis-test-base", "H-s101": "test-eval/qis-test-H-s101-sym", "R-s101": "test-eval/qis-test-R-s101-sym",
         "U-s101": "test-eval/qis-test-U-s101-sym", "H-s202": "test-eval/qis-test-H-s202-sym",
         "R-s202": "test-eval/qis-test-R-s202-sym", "U-s202": "test-eval/qis-test-U-s202-sym",
         "H-s202-fb116": "test-eval/qis-test-H-s202-sym-fallback", "cgpu-R-s101": "test-eval-cgpu/qis-cgpu-R-s101",
         "cgpu-R-s202": "test-eval-cgpu/qis-cgpu-R-s202", "cgpu-U-s101": "test-eval-cgpu/qis-cgpu-U-s101",
         "cgpu-U-s202": "test-eval-cgpu/qis-cgpu-U-s202"}
C = {k: counts(f"{P}/{v}/records.jsonl.gz") for k, v in CELLS.items()}
for k, c in C.items():
    assert set(c) == set(TEST_IDS) and all(x[1] == 32 for x in c.values()), k
SETS = {"full128": TEST_IDS, "drop00383_127": [p for p in TEST_IDS if p != TEST_DROP],
        "drop00383_00891_126": [p for p in TEST_IDS if p not in (TEST_DROP, TEST_DROP2)]}
t = {"cells": {}, "per_problem_dropped": {}, "contrasts": {}}
for k, c in C.items():
    t["cells"][k] = {s: {"pass1": pass1(c, ids), "pass8": pass8(c, ids)} for s, ids in SETS.items()}
    t["per_problem_dropped"][k] = {TEST_DROP: c[TEST_DROP][:2], TEST_DROP2: c[TEST_DROP2][:2]}
PAIRS = [("R-s101", "U-s101"), ("R-s101", "H-s101"), ("U-s101", "H-s101"), ("R-s202", "U-s202"), ("R-s202", "H-s202"),
         ("U-s202", "H-s202"), ("cgpu-R-s101", "cgpu-U-s101"), ("cgpu-R-s202", "cgpu-U-s202"), ("R-s202", "H-s202-fb116"),
         ("U-s202", "H-s202-fb116")] + [(a, "base") for a in ("H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202")]
for a, b in PAIRS:
    t["contrasts"][f"{a} - {b}"] = {s: boot(C[a], C[b], ids) for s, ids in SETS.items()}
# two-seed means (simple average of per-seed estimates)
for a, b in (("R", "U"), ("R", "H"), ("U", "H")):
    t["contrasts"][f"mean {a}-{b}"] = {s: (t["contrasts"][f"{a}-s101 - {b}-s101"][s]["diff"] +
                                          t["contrasts"][f"{a}-s202 - {b}-s202"][s]["diff"]) / 2 for s in SETS}
out["test"] = t

# ---------- (a) dev curves without the dev variant ----------
RUNS = ["H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202"]
bundle = json.load(open(P + "/data/main-v1/bundle_main_v1.json"))
pools = bundle.get("pools", bundle)
jh = [it["id"] if isinstance(it, dict) else it for it in pools["probe_J_H"]]
dev_order = [it["id"] if isinstance(it, dict) else it for it in pools["dev"]]
d = {"dev_drop_in_J_H": DEV_DROP in jh, "dev_drop_index_0based": dev_order.index(DEV_DROP), "curves": {}, "R_minus_U": {},
     "per_problem_dropped": {}}
DC = {}
for run in RUNS:
    rd = f"{P}/runs/qis-main-{run}-sym/records"
    for rnd in (0, 32, 64, 96, 128):
        DC[(run, rnd)] = counts(f"{rd}/eval_dev_r{rnd:04d}.jsonl.gz")
DC[("H-s202-a2orig", 64)] = counts(f"{P}/runs/qis-main-H-s202-sym/records_a2orig_r60-64/eval_dev_r0064.jsonl.gz")
for (run, rnd), c in DC.items():
    assert len(c) == 53 and DEV_DROP in c and all(x[1] == 8 for x in c.values()), (run, rnd)
    ids = sorted(c)
    red = [p for p in ids if p != DEV_DROP]
    d["curves"][f"{run}|r{rnd}"] = {"dev53": pass1(c, ids), "dev52": pass1(c, red),
                                     "change": pass1(c, red) - pass1(c, ids)}
    d["per_problem_dropped"][f"{run}|r{rnd}"] = c[DEV_DROP][:2]
for s in ("s101", "s202"):
    for rnd in (0, 32, 64, 96, 128):
        ca, cb = DC[(f"R-{s}", rnd)], DC[(f"U-{s}", rnd)]
        ids = sorted(ca)
        d["R_minus_U"][f"{s}|r{rnd}"] = {"dev53": boot(ca, cb, ids), "dev52": boot(ca, cb, [p for p in ids if p != DEV_DROP])}
out["dev"] = d

# ---------- (c) verifier-unknown ----------
u = {}
for k, c in C.items():
    unk = {p: x[2] for p, x in c.items() if x[2]}
    if unk:
        lo = pass1(c, TEST_IDS)
        hi_c = {p: [x[0] + x[2], x[1], 0] for p, x in c.items()}
        u[k] = {"unknown_by_problem": unk, "c_of_problem": {p: c[p][:2] for p in unk}, "pass1_as_scored": lo,
                "pass1_if_all_unknown_correct": pass1(hi_c, TEST_IDS), "max_change": pass1(hi_c, TEST_IDS) - lo,
                "pass8_as_scored": pass8(c, TEST_IDS), "pass8_if_all_unknown_correct": pass8(hi_c, TEST_IDS),
                "solved_any_as_scored": sum(1 for x in c.values() if x[0] > 0),
                "solved_any_if_correct": sum(1 for x in hi_c.values() if x[0] > 0)}
out["unknown_test"] = u

# ---------- tranche-1 bootstrap: seed variability of the s202 R-U interval at 12,288 ----------
def t1counts(ck):
    c = {}
    for r in rows(f"{P}/length-diag/{ck}-r128/t1/rows.jsonl.gz"):
        x = c.setdefault(r["prompt_id"], [0, 0, 0])
        x[0] += 1 if r["reward_1"] == 1 else 0
        x[1] += 1
    return c


ra, ua = t1counts("R-s202"), t1counts("U-s202")
ids = sorted(ra)
out["t1_s202_RU_12288"] = {"problems": len(ids), "attempts_per_problem": sorted({x[1] for x in ra.values()}),
                           "by_seed": {str(sd): boot(ra, ua, ids, seed=sd) for sd in (2027, 20261111, 20261112, 1, 2, 3, 4, 5)}}
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
