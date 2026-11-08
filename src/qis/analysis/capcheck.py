"""A11 cap-transition check from run logs (read-only; run in a CPU job).

For each a2 main run, around the transition (rounds 31-35, 1-based): the round metric `max_tokens` (training/rollout
cap), the per-response training records (max n_tokens, count > 8,192, count finished by length and their n_tokens,
which must equal the cap, and their grading: status truncated, reward 0), and dev/probe eval records with their
`max_tokens` (grading cap). Expected: rounds <= 32 use 8,192; round 33 onward uses 12,288; capped responses end exactly at the round's cap.
Usage: python -m qis.analysis.capcheck RUNS_DIR OUT.json
"""
import glob
import gzip
import json
import os
import sys


def check(d):
    M = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    att = M[0].get("attempt_id")
    rounds = {r["round"]: r for r in M if r.get("kind") == "round"}
    out = {"attempt_id": att, "last_round": max(rounds) if rounds else None, "rounds": {}, "evals": []}
    for rnd in range(31, 36):
        if rnd not in rounds:
            continue
        x = {"metric_max_tokens": rounds[rnd].get("max_tokens")}
        p = os.path.join(d, "records", f"train_r{rnd - 1:04d}.jsonl.gz")
        if os.path.exists(p):
            recs = [json.loads(l) for l in gzip.open(p, "rt")]
            capped = [r["n_tokens"] for r in recs if r["finish"] == "length"]
            x.update(records=len(recs), max_n_tokens=max(r["n_tokens"] for r in recs),
                     over_8192=sum(r["n_tokens"] > 8192 for r in recs), capped=len(capped),
                     capped_lengths=sorted(set(capped)),           # grading: capped -> status truncated, reward 0
                     capped_graded_truncated=sum(r["finish"] == "length" and r["status"] == "truncated" and not r["reward"]
                                                 for r in recs))
        out["rounds"][rnd] = x
    for r in M:
        if r.get("kind") == "eval" and r.get("round", 0) >= 24:
            out["evals"].append({k: r.get(k) for k in ("round", "panel", "max_tokens", "pass1", "truncated")})
    return out


def main(runs, outp):
    res = {os.path.basename(d): check(d) for d in sorted(glob.glob(os.path.join(runs, "qis-main-*")))
           if os.path.isfile(os.path.join(d, "metrics.jsonl")) and os.path.getsize(os.path.join(d, "metrics.jsonl"))}
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
