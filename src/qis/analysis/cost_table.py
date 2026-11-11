"""Compute ledger: GPU-hours and CPU-hours per category from a Slurm accounting dump (run in a CPU job).

Input: `sacct -D -X -P -n -o JobID,JobName,State,Partition,ElapsedRaw,AllocTRES,Start,End -j <every ledger job id>`
(-D keeps every requeued allocation). Every allocation is charged, including failed, cancelled and superseded ones.
Main runs are listed per run and split into the canonical a2 job chain and charged-but-not-canonical allocations
(names given by the caller). Output JSON + Markdown table. Running jobs count their elapsed time so far.
Usage: python -m qis.analysis.cost_table SACCT.txt OUT_PREFIX [CANONICAL_JOBIDS_COMMA_SEPARATED]
"""
import collections
import json
import re
import sys


def tres(t, key):
    m = re.search(rf"{key}=(\d+)", t)
    return int(m.group(1)) if m else 0


def category(name, gpus):
    n = name.replace("ae-qis-", "")
    if n.startswith("main-"):
        return "main " + n[5:].replace("-sym", "")
    if n.startswith("test-"):
        return "sealed test evaluation"
    if n.startswith("diag-local") or n.startswith("diag-ref"):
        return "local-update diagnostic"
    if n.startswith("cgpu-"):
        return "common-GPU rerun (A17)"
    if n.startswith("stress-"):
        return "stress evaluation (A17)"
    if n.startswith("len-") or n.startswith("lentest-"):
        return "length diagnostic (A19)"
    if n.startswith("cue-"):
        return "strategy-cue panel (A15.4)"
    return "other GPU jobs (screening, calibration, smoke, A10/A11 studies)" if gpus else "CPU jobs"


def main(path, out, canonical=""):
    canon = set(canonical.split(",")) if canonical else set()
    cat = collections.defaultdict(lambda: {"jobs": 0, "allocations": 0, "gpu_h": 0.0, "cpu_h": 0.0,
                                           "gpu_h_canonical": 0.0, "states": collections.Counter()})
    seen = collections.defaultdict(set)
    for line in open(path):
        if not line.strip():
            continue
        jid, name, state, part, el, t, *_ = line.rstrip("\n").split("|")
        g, c, s = tres(t, "gres/gpu"), tres(t, "cpu"), int(el or 0)
        k = category(name, g)
        x = cat[k]
        seen[k].add(jid); x["allocations"] += 1
        x["gpu_h"] += g * s / 3600; x["cpu_h"] += c * s / 3600
        x["states"][state.split()[0]] += 1
        if jid in canon:
            x["gpu_h_canonical"] += g * s / 3600
    for k in cat:
        cat[k]["jobs"] = len(seen[k]); cat[k]["states"] = dict(cat[k]["states"])
    tot = {f: sum(v[f] for v in cat.values()) for f in ("jobs", "allocations", "gpu_h", "cpu_h")}
    json.dump({"categories": cat, "total": tot}, open(out + ".json", "w"), indent=1)
    md = ["| Category | Jobs | Allocations | GPU-h | of which canonical | CPU-h | Final states |", "|---|---|---|---|---|---|---|"]
    for k in sorted(cat):
        v = cat[k]
        md.append(f"| {k} | {v['jobs']} | {v['allocations']} | {v['gpu_h']:.1f} | "
                  f"{v['gpu_h_canonical']:.1f} | {v['cpu_h']:.1f} | "
                  + ", ".join(f"{s} {n}" for s, n in sorted(v["states"].items())) + " |")
    md.append(f"| **Total** | {tot['jobs']} | {tot['allocations']} | {tot['gpu_h']:.1f} | | {tot['cpu_h']:.1f} | |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:4])
