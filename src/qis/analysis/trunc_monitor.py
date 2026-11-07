"""Truncation monitor for the main study (amendment A8.4, D66; run in a CPU job; read-only).

Reads per-response training records `runs/<run>/records/train_r*.jsonl.gz` (role, finish) and the scheduled
evaluation records in metrics.jsonl. Reports per run: capped / total for hard prompts (roles starting with
"hard") and auxiliary prompts (role "aux"), per round and cumulative; per evaluation: pass@1 with truncation.
For each seed it compares R and U on their shared hard schedule (role "hard_common", same rounds). A fraction
above 5% is flagged for inspection and reporting only; the training cap never changes.
Usage: python -m qis.analysis.trunc_monitor RUNS_DIR OUT.json
"""
import glob
import gzip
import json
import os
import re
import sys

FLAG = 0.05


def run_counts(d):
    rounds = {}
    for p in sorted(glob.glob(os.path.join(d, "records", "train_r*.jsonl.gz"))):
        rnd = int(re.search(r"train_r(\d+)", p).group(1))
        c = {}
        try:
            for line in gzip.open(p, "rt"):
                r = json.loads(line)
                k = "hard" if r["role"].startswith("hard") else "aux"
                k2 = r["role"]
                for key in (k, k2):
                    t = c.setdefault(key, [0, 0])
                    t[0] += r["finish"] == "length"; t[1] += 1
        except (EOFError, OSError):      # file being written right now
            continue
        rounds[rnd] = c
    return rounds


def evals(d):
    out = []
    p = os.path.join(d, "metrics.jsonl")
    if os.path.exists(p):
        for line in open(p):
            if '"kind": "eval"' in line:
                r = json.loads(line)
                out.append({k: r.get(k) for k in ("panel", "round", "pass1", "truncated", "unknown")})
    return out


def main(runs_dir, out_path):
    res, flags = {}, []
    for d in sorted(glob.glob(os.path.join(runs_dir, "qis-main-*")) + glob.glob(os.path.join(runs_dir, "qis-sens-*"))):
        run = os.path.basename(d)
        rc = run_counts(d)
        cum = {}
        for c in rc.values():
            for k, (a, b) in c.items():
                t = cum.setdefault(k, [0, 0]); t[0] += a; t[1] += b
        last = max(rc) if rc else None
        res[run] = {"rounds_logged": len(rc), "last_round": last,
                    "cumulative": {k: {"capped": a, "total": b, "frac": a / b} for k, (a, b) in cum.items()},
                    "last_round_counts": rc.get(last, {}), "evals": evals(d)}
        for k, (a, b) in cum.items():
            if b and a / b > FLAG:
                flags.append(f"{run} cumulative {k} {a}/{b}")
        for e in res[run]["evals"]:
            if e["truncated"] is not None and e["truncated"] > FLAG:
                flags.append(f"{run} eval {e['panel']} r{e['round']} truncated {e['truncated']:.3f}")
    pairs = {}
    for run in res:
        m = re.match(r"qis-(main|sens)-([RU])-s(\d+)-(\w+)", run)
        if m:
            pairs.setdefault((m.group(1), m.group(3), m.group(4)), {})[m.group(2)] = run
    shared = {}
    for (ph, seed, clip), v in pairs.items():
        if "R" in v and "U" in v:
            rr, uu = run_counts(os.path.join(runs_dir, v["R"])), run_counts(os.path.join(runs_dir, v["U"]))
            common = sorted(set(rr) & set(uu))
            agg = {}
            for arm, rc in (("R", rr), ("U", uu)):
                a = sum(rc[r].get("hard_common", [0, 0])[0] for r in common)
                b = sum(rc[r].get("hard_common", [0, 0])[1] for r in common)
                agg[arm] = {"capped": a, "total": b, "frac": a / b if b else None}
            shared[f"{ph}-s{seed}-{clip}"] = {"common_rounds": len(common), **agg}
    json.dump({"runs": res, "R_vs_U_shared_hard": shared, "flags_over_5pct": flags}, open(out_path, "w"), indent=1)
    print(json.dumps({"R_vs_U_shared_hard": shared, "flags_over_5pct": flags,
                      "cumulative": {r: v["cumulative"] for r, v in res.items()}}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
