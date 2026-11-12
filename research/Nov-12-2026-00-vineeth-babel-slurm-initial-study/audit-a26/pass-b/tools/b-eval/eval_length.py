"""A26 pass B task 5: length extension 12,288 -> 16,384 from rows + original records."""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import P, boot_diff, dump, load_records, panel_ids

FROM, TO = 12288, 16384
POLS = ["H-s101", "R-s101", "U-s101", "R-s202", "U-s202"]
TEST_ORDER = panel_ids(P + "/data/main-v1/target_test_sealed.json", "test")
ST_ITEMS = json.load(open(P + "/data/stress/stress_v1.json"))["pools"]["stress"]
ST_ORDER = [it["id"] for it in ST_ITEMS]
TIER = {it["id"]: it["tier"] for it in ST_ITEMS}


def load_rows(paths):
    rows, dups, per_file = {}, 0, {}
    for p in paths:
        cnt = 0
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                r.pop("new_ids", None)
                k = (r["prompt_id"], int(r["sample"]))
                dups += k in rows
                rows[k] = r; cnt += 1
        per_file[os.path.relpath(p, P)] = cnt
    return rows, dups, per_file


def analyse(orig, rows, order, n):
    chk = {"original_attempts": len(orig), "rows": len(rows), "rows_missing": len(set(orig) - set(rows)),
           "rows_extra": len(set(rows) - set(orig))}
    capped = [k for k, r in orig.items() if r["finish"] == "length"]
    chk["original_capped"] = len(capped)
    chk["original_capped_not_12288"] = sum(orig[k]["n_tokens"] != FROM for k in capped)
    chk["original_12288_but_not_capped"] = sum(1 for r in orig.values() if r["n_tokens"] == FROM and r["finish"] != "length")
    chk["extended_rows"] = sum(1 for r in rows.values() if r["extended"])
    chk["eligible_not_extended"] = sum(1 for k in capped if not rows.get(k, {}).get("extended"))
    chk["extended_but_not_eligible"] = sum(1 for k, r in rows.items() if r["extended"] and orig[k]["finish"] != "length")
    chk["extended_rows_n_tokens_1_not_12288"] = sum(1 for r in rows.values() if r["extended"] and r["n_tokens_1"] != FROM)
    chk["row_fields_1_differ_from_original"] = sum(
        1 for k, r in rows.items() if (r["n_tokens_1"], r["finish_1"], r["reward_1"]) != (orig[k]["n_tokens"], orig[k]["finish"], orig[k]["reward"]))
    chk["unextended_reward_changed"] = sum(1 for r in rows.values() if not r["extended"] and r["reward_2"] != r["reward_1"])
    chk["extended_n_tokens_2_ne_12288_plus_new"] = sum(1 for r in rows.values() if r["extended"] and r["n_tokens_2"] != FROM + r["new_tokens"])
    chk["n_tokens_2_over_16384"] = sum(1 for r in rows.values() if r["n_tokens_2"] > TO)
    chk["still_capped_not_16384"] = sum(1 for r in rows.values() if r["extended"] and r["finish_2"] == "length" and r["n_tokens_2"] != TO)
    chk["still_capped_with_reward"] = sum(1 for r in rows.values() if r["extended"] and r["finish_2"] == "length" and r["reward_2"] == 1)
    chk["unknown_status_2"] = sum(1 for r in rows.values() if r.get("status_2") == "unknown")
    ext = [r for r in rows.values() if r["extended"]]
    cnt = {"attempts": len(rows), "capped_at_12288": len(capped), "extended": len(ext),
           "recovered": sum(r["reward_2"] == 1 for r in ext),
           "finished_wrong": sum(r["finish_2"] != "length" and r["reward_2"] != 1 for r in ext),
           "still_capped": sum(r["finish_2"] == "length" for r in ext),
           "correct_12288": sum(r["reward_1"] == 1 for r in rows.values()),
           "correct_16384": sum(r["reward_2"] == 1 for r in rows.values())}
    c1 = np.array([sum(rows[(p, j)]["reward_1"] == 1 for j in range(n)) for p in order], float)
    c2 = np.array([sum(rows[(p, j)]["reward_2"] == 1 for j in range(n)) for p in order], float)
    cnt["pass1_12288"] = float(np.mean(c1 / n)); cnt["pass1_16384"] = float(np.mean(c2 / n))
    cnt["truncation_12288"] = len(capped) / len(rows)
    cnt["truncation_16384"] = cnt["still_capped"] / len(rows)
    return chk, cnt, c1 / n, c2 / n


out = {"test": {}, "stress": {}, "contrasts": {}}
X = {"test": {}, "stress": {}}
for pol in POLS:
    orig, _ = load_records(f"{P}/test-eval/qis-test-{pol}-sym/records.jsonl.gz")
    rows, dups, pf = load_rows([f"{P}/length-test/{pol}-r128/all/rows.jsonl.gz"])
    chk, cnt, x1, x2 = analyse(orig, rows, TEST_ORDER, 32)
    chk["duplicate_rows"] = dups; chk["files"] = pf
    out["test"][pol] = {"checks": chk, "counts": cnt}
    X["test"][pol] = (x1, x2)

    orig, _ = load_records(f"{P}/stress-eval/{pol}-r128/records.jsonl.gz")
    rows, dups, pf = load_rows([f"{P}/length-diag/{pol}-r128/t1/rows.jsonl.gz", f"{P}/length-diag/{pol}-r128/rest/rows.jsonl.gz"])
    chk, cnt, x1, x2 = analyse(orig, rows, ST_ORDER, 8)
    chk["duplicate_rows_across_t1_rest"] = dups; chk["files"] = pf
    tiers = {}
    for tier in ("harder", "superhard"):
        m = np.array([TIER[p] == tier for p in ST_ORDER])
        tiers[tier] = {"problems": int(m.sum()), "pass1_12288": float(x1[m].mean()), "pass1_16384": float(x2[m].mean())}
    cnt["tiers"] = tiers
    out["stress"][pol] = {"checks": chk, "counts": cnt}
    X["stress"][pol] = (x1, x2)

for panel in ("test", "stress"):
    for A, B in (("R-s101", "U-s101"), ("R-s202", "U-s202"), ("R-s101", "H-s101"), ("U-s101", "H-s101")):
        a1, a2 = X[panel][A]; b1, b2 = X[panel][B]
        out["contrasts"][f"{panel} {A} - {B}"] = {
            "at_12288": boot_diff(a1, b1), "at_16384": boot_diff(a2, b2),
            "change": boot_diff(a2 - a1, b2 - b1)}
dump("eval_length.json", out)
print(json.dumps(out, indent=0)[:6000])
