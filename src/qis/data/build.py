"""Build the split manifest and the annotation candidate exports (CPU job).

Usage: python -m qis.data.build --raw DIR --out DIR --export DIR
  --raw     pinned MATH parquet root (downloaded here if absent)
  --out     PROJ_DATA directory for the full manifest (large, stays on compute)
  --export  small files for the laptop (harvested): summary, candidate pools for annotation
"""
import argparse
import collections
import gzip
import json
import os
import re

from qis.data import math_source, splits
from qis.verify import Verifier

# Keyword prefilters choose which items are ANNOTATED; the agent rubric decides membership.
FACTOR_RE = re.compile(r"factor|\)\s*\(|difference of (two )?squares", re.I)
MOD_RE = re.compile(r"\\pmod|\\mod|\bmod\b|modulo|remainder|congruen|units digit|last digit", re.I)
ALG = ("algebra", "intermediate_algebra")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    math_source.download(a.raw)
    items = list(math_source.load_math(a.raw))
    rec, dup_pairs = splits.assign_splits(items)

    v = Verifier(a.workers)
    try:
        withgold = [it for it in items if it["gold"] is not None]
        res = v.score([{"gold": it["gold"], "response": "\\boxed{" + it["gold"] + "}", "truncated": False}
                       for it in withgold])
    finally:
        v.close()
    for it, r in zip(withgold, res):
        rec[it["id"]].update(gold_eligible=r["status"] == "correct", gold_status=r["status"])
    for it in items:
        if it["gold"] is None:
            rec[it["id"]].update(gold_eligible=False, gold_status="no_boxed")

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.export, exist_ok=True)
    with open(os.path.join(a.out, "manifest.jsonl.tmp"), "w") as f:
        for it in items:
            f.write(json.dumps({**it, **rec[it["id"]]}) + "\n")
    os.replace(os.path.join(a.out, "manifest.jsonl.tmp"), os.path.join(a.out, "manifest.jsonl"))
    with open(os.path.join(a.out, "dup_pairs.json"), "w") as f:
        json.dump(dup_pairs, f)

    # annotation exports
    pools = collections.defaultdict(list)
    for it in items:
        r = rec[it["id"]]
        if r["split"] == "excluded" or not r.get("gold_eligible"):
            continue
        row = {k: it[k] for k in ("id", "subject", "level", "problem", "solution", "gold")}
        row["split"] = r["split"]
        if it["subject"] in ALG and FACTOR_RE.search(it["solution"]):
            pools["factor"].append(row)
        if it["subject"] == "number_theory" and it["orig_split"] == "train" \
                and MOD_RE.search(it["problem"] + " " + it["solution"]):
            pools["modular"].append(row)
    for name, rows in pools.items():
        with gzip.open(os.path.join(a.export, f"cand_{name}.jsonl.gz"), "wt") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    count = collections.Counter((it["subject"], it["orig_split"], rec[it["id"]]["split"],
                                 bool(rec[it["id"]].get("gold_eligible"))) for it in items)
    summary = {
        "math_revision": math_source.MATH_REVISION,
        "n_items": len(items),
        "n_dup_pairs": len(dup_pairs),
        "split_counts": collections.Counter(rec[i]["split"] for i in rec),
        "by_subject_split_eligible": {"|".join(map(str, k)): n for k, n in sorted(count.items())},
        "gold_status": collections.Counter(rec[i].get("gold_status") for i in rec),
        "pool_sizes": {k: collections.Counter(r["split"] for r in v_) for k, v_ in pools.items()},
        "sample_dup_pairs": dup_pairs[:40],
        "manifest_sha256": math_source.sha(open(os.path.join(a.out, "manifest.jsonl")).read()),
    }
    with open(os.path.join(a.export, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("n_items", "n_dup_pairs", "split_counts", "pool_sizes")}))


if __name__ == "__main__":
    main()
