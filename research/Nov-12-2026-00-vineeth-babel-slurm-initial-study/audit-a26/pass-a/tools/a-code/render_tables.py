#!/usr/bin/env python3
"""A26 pass A (a-code): print the markdown tables of jobs.md from data/jobs.json (counts by script, not by hand).
Usage: python3 render_tables.py research/audit-a26/pass-a/data/jobs.json
"""
import json, sys, collections

J = json.load(open(sys.argv[1]))
ORDER = ["training", "sealed test", "sealed test (fallback r116)", "cgpu reruns", "stress", "length", "cue", "diag",
         "calibration", "data build/screening", "tests", "in-run monitoring/inspection", "ops/storage/recovery", "report",
         "selftest/env", "audits (A26)"]


def st(s):
    return s.split(" by ")[0]


print("| Family | Rows | Expids | Commits | Final state (sacct) | GPU types used (instances) | expsha ≠ local recipe | Job script on Babel |")
print("|---|---|---|---|---|---|---|---|")
tot = collections.Counter()
for f in ORDER:
    L = [o for o in J if o["family"] == f]
    if not L:
        continue
    s = collections.Counter(st(o["final_state"]) for o in L)
    g = collections.Counter(i["gpu"] for o in L for i in o["sacct_instances"] if i["gpu"] != "none" and i["node"] != "None assigned")
    mm = sum(o["expsha_match"] is False for o in L)
    js = sum(o["jobscript"] is not None for o in L)
    print(f"| {f} | {len(L)} | {len({o['expid'] for o in L})} | {len({o['sha7'] for o in L})} | "
          + ", ".join(f"{k} {v}" for k, v in sorted(s.items())) + " | "
          + (", ".join(f"{k} {v}" for k, v in sorted(g.items())) or "CPU only") + f" | {mm} | {js}/{len(L)} |")
    tot["rows"] += len(L); tot["mm"] += mm; tot["js"] += js
    for k, v in s.items():
        tot[k] += v
print(f"| **all** | {tot['rows']} | {len({o['expid'] for o in J})} | {len({o['sha7'] for o in J})} | "
      + ", ".join(f"{k} {tot[k]}" for k in ("COMPLETED", "CANCELLED", "FAILED", "RUNNING")) + f" | | {tot['mm']} | {tot['js']}/{tot['rows']} |")
print()
print("Science-bearing jobs (training and evaluation families): job, commit, final state, node GPU per instance.")
print()
print("| Family | Expid | Job | Commit | Final state | GPU (instances) | Seed label / cap |")
print("|---|---|---|---|---|---|---|")
for f in ORDER[:8]:
    for o in J:
        if o["family"] != f:
            continue
        g = "→".join(i["gpu"] for i in o["sacct_instances"] if i["node"] != "None assigned") or "never started"
        r = o["recipe"]
        lab = ",".join(r["seed_labels"]) + ("/" + ",".join(r["max_tokens_args"]) if r["max_tokens_args"] else "")
        print(f"| {f} | `{o['expid']}` | {o['jobid']} | {o['sha7']} | {st(o['final_state'])} | {g} | {lab or '—'} |")
