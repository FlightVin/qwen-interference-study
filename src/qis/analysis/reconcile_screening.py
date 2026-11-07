"""Records reconciliation counts (amendment A8.5; run in a CPU job; reads committed files only).

1. Native H candidates (checklist §3.12/§3.16): the 4,096-cap stage-2 set (hard + hard_reserve, 32 attempts
   each), how many were excluded by their 4,096 score (c > 12/32: success at a larger cap is >= success at
   the smaller cap, D45), how many received 32 fresh attempts at 8,192, and how many with 4k c <= 12 were
   not re-screened (annotation/audit ineligibility).
2. Calibration panels: per split, items screened at 8,192 (16 samples), items in the low (1-5/16) and
   high (6-12/16) bands, and the frozen panel sizes.
Usage: python -m qis.analysis.reconcile_screening [CALIB_BUNDLE.json]
"""
import json
import sys


def counts(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def main():
    out = {}
    s4 = {**counts("data/screening/hard_s2_counts.jsonl"), **counts("data/screening/hard_reserve_s2_counts.jsonl")}
    k8 = counts("data/screening/h_s2k8_counts.jsonl")
    native_k8 = {i for i in k8 if i in s4}
    excl4 = {i for i, r in s4.items() if r["c"] > 12}
    le12 = {i for i, r in s4.items() if r["c"] <= 12}
    out["native"] = {"stage2_4k_items": len(s4), "n32_each": sorted({r["n"] for r in s4.values()}),
                     "excluded_by_4k_score_c_gt_12": len(excl4), "4k_c_le_12": len(le12),
                     "rescreened_8k_32_fresh": len(native_k8),
                     "rescreened_8k_n": sorted({k8[i]["n"] for i in native_k8}),
                     "le12_not_rescreened_(annotation_ineligible)": len(le12 - native_k8),
                     "rescreened_with_4k_c_gt_12_(should_be_0)": len(native_k8 & excl4),
                     "omni_rescreened_8k": len(set(k8) - set(s4))}
    sel = json.load(open("data/screening/hsel_k8.json"))["selected"]
    out["native"]["selected_H_train_native"] = sum(1 for h in sel if h["id"] in s4)
    for sp in ("calib_train", "calib_val"):
        c = counts(f"data/screening/{sp}_calib16k8_counts.jsonl")
        out[sp] = {"screened_8k": len(c), "low_1_5": sum(1 <= r["c"] <= 5 for r in c.values()),
                   "high_6_12": sum(6 <= r["c"] <= 12 for r in c.values()),
                   "zero": sum(r["c"] == 0 for r in c.values()), "above_12": sum(r["c"] > 12 for r in c.values()),
                   "unknown_attempts": sum(r.get("unknown", 0) for r in c.values())}
    panels = json.load(open("data/panels/panels.json"))
    out["frozen_panels_in_panels_json"] = {k: len(v["ids"]) for k, v in panels.items() if k.startswith("calib")}
    if len(sys.argv) > 1:
        b = json.load(open(sys.argv[1]))["pools"]
        out["frozen_calib_bundle"] = {k: len(v) for k, v in b.items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
