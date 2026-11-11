"""Merge sharded evaluate_ckpt outputs into one output directory (A17, D149).

A panel split into shards keeps the per-problem seeds, because the seed key is
eval/<panel_key>/<id>/<seed_label>/n<n> and does not depend on the other items. Shards differ from a
single job only in vLLM batch composition (D51: cross-job runs are not bitwise reproducible anyway).
The merged summary is recomputed from the records with the same formulas as evaluate_ckpt.
Usage: python -m qis.eval_merge PANEL.json PANEL_KEY OUT SHARD_DIR...
       python -m qis.eval_merge --split K PANEL.json PANEL_KEY OUT_PREFIX   (write K interleaved shard panels)
"""
import gzip
import json
import os
import sys

import numpy as np

from qis.train import grpo
from qis.verify import UNKNOWN

SAME = ("ckpt", "ckpt_model_sha256", "panel", "n", "max_tokens", "seed_label")


def split(k, panel, key, prefix):
    p = json.load(open(panel))
    items = p["pools"][key]
    for s in range(k):
        json.dump({**p, "pools": {key: items[s::k]}}, open(f"{prefix}{s}.json", "w"))
        print(f"{prefix}{s}.json", len(items[s::k]))


def merge(panel, key, out, shards):
    order = [it["id"] for it in json.load(open(panel))["pools"][key]]
    sums = [json.load(open(os.path.join(d, "summary.json"))) for d in shards]
    for f in SAME:
        vals = {json.dumps(s.get(f)) for s in sums}
        assert len(vals) == 1, f"shards disagree on {f}: {vals}"
    recs = {}
    for d in shards:
        with gzip.open(os.path.join(d, "records.jsonl.gz"), "rt") as f:
            for line in f:
                r = json.loads(line)
                recs.setdefault(r["prompt_id"], []).append(r)
    assert sorted(recs) == sorted(order), "shard ids do not cover the panel exactly"
    n = sums[0]["n"]
    shard_per = {p["id"]: p for s in sums for p in s["per_problem"]}
    os.makedirs(out, exist_ok=True)
    per, flat = [], []
    with gzip.open(os.path.join(out, "records.jsonl.gz"), "wt") as f:
        for pid in order:
            g = sorted(recs[pid], key=lambda r: r["sample"])
            assert [r["sample"] for r in g] == list(range(n)), f"{pid}: samples incomplete"
            for r in g:
                f.write(json.dumps(r) + "\n")
            c = sum(r["reward"] == 1 for r in g)
            c8 = sum(r["reward"] == 1 and r["n_tokens"] <= 8192 for r in g)
            assert (c, c8) == (shard_per[pid]["c"], shard_per[pid]["c_within_8192"]), f"{pid}: records disagree with shard summary"
            per.append({"id": pid, "c": c, "c_within_8192": c8, "n": n})
            flat += g
    tot = n * len(order)
    secs = sum(s["seconds"] for s in sums)
    toks = sum(r["n_tokens"] for r in flat)
    summ = {**{f: sums[0][f] for f in SAME}, "weight_check": [s.get("weight_check") for s in sums], "problems": len(order),
            "pass1": float(np.mean([p["c"] / n for p in per])),
            "pass8": float(np.mean([grpo.pass_at_k(n, p["c"], 8) for p in per])),
            "pass1_within_8192": float(np.mean([p["c_within_8192"] / n for p in per])),
            "over_8192": sum(r["n_tokens"] > 8192 for r in flat) / tot,
            "over_8192_then_correct": sum(r["reward"] == 1 and r["n_tokens"] > 8192 for r in flat) / tot,
            "tokens_per_second": toks / max(secs, 1e-9), "completions": tot,
            "truncated": sum(r["finish"] == "length" for r in flat) / tot,
            "unknown": sum(r["status"] == UNKNOWN for r in flat), "tokens": toks, "seconds": secs,
            "shard_seconds": [s["seconds"] for s in sums], "shards": [os.path.abspath(d) for d in shards],
            "per_problem": per}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_problem"}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--split":
        split(int(a[1]), a[2], a[3], a[4])
    else:
        merge(a[0], a[1], a[2], a[3:])
