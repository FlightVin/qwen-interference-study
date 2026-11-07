"""Stage-1 shortlist (project.md Section 5): in fixed candidate order, keep H candidates with
0-5/16 successes and auxiliary candidates with 4-14/16, up to twice the target size per pool.
Items with any verifier-unknown attempt are held out and listed (never silently dropped).

Usage: python -m qis.data.shortlist --cand DIR --s1 DIR --out DIR
"""
import argparse
import json
import os

BANDS = {"hard": (0, 5), "hard_reserve": (0, 5), "related": (4, 14), "unrelated": (4, 14)}
TARGET = {"hard": 128, "hard_reserve": 128, "related": 128, "unrelated": 128}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for pool, (lo, hi) in BANDS.items():
        order = [json.loads(line) for line in open(os.path.join(a.cand, f"{pool}.jsonl"))]
        counts = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(a.s1, pool, "s1_counts.jsonl"))}
        keep, held, missing, hist = [], [], [], {}
        for it in order:
            c = counts.get(it["id"])
            if c is None:
                missing.append(it["id"]); continue
            hist[c["c"]] = hist.get(c["c"], 0) + 1
            if c["unknown"]:
                held.append(it["id"]); continue
            if lo <= c["c"] <= hi and len(keep) < 2 * TARGET[pool]:
                keep.append(it)
        with open(os.path.join(a.out, f"{pool}.jsonl"), "w") as f:
            for it in keep:
                f.write(json.dumps(it) + "\n")
        summary[pool] = {"candidates": len(order), "screened": len(order) - len(missing), "shortlisted": len(keep),
                         "held_unknown": held, "missing_or_over_cap": missing,
                         "s1_success_hist": dict(sorted(hist.items()))}
    json.dump(summary, open(os.path.join(a.out, "shortlist_summary.json"), "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("candidates", "shortlisted")} for k, v in summary.items()}))


if __name__ == "__main__":
    main()
