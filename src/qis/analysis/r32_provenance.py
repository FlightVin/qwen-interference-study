"""Round-32 dev provenance: superseded attempt a1 vs retained attempt a2 (read-only; run in a CPU job).

For each s101 run and attempt: run/attempt id, start events (job, code sha, bundle sha), config cap and eval panels,
every logged dev eval row at round 32 (repeats are counted), and a recomputation of pass@1 from the raw records
`records/eval_dev_r0032.jsonl.gz` (correct = reward 1; per-problem c/n, macro mean; also micro mean), with the
records-file sha256, sampling seeds and response caps. Across attempts: same (prompt, sample) seeds, identical token
ids, and the first training round whose rollout token ids differ (replay divergence). Checkpoints present per tree.
Usage: python -m qis.analysis.r32_provenance PROJ_DATA OUT.json
"""
import collections
import gzip
import hashlib
import json
import os
import sys

RUNS = ("qis-main-R-s101-sym", "qis-main-U-s101-sym", "qis-main-H-s101-sym")
RND = 32


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def recs(p):
    return [json.loads(l) for l in gzip.open(p, "rt")] if os.path.exists(p) else None


def attempt(run_dir, ckpt_dir):
    M = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))]
    cfg = next((r["config"] for r in M if r.get("kind") == "config"), {})
    starts = [{k: r.get(k) for k in ("utc", "job", "restart", "code_sha", "bundle_sha", "resume_round")}
              for r in M if r.get("kind") == "event" and r.get("what") == "start"]
    rows = [{k: r.get(k) for k in ("utc", "attempt_id", "round", "ckpt_round", "n", "problems", "pass1", "truncated",
                                   "max_tokens", "unknown")}
            for r in M if r.get("kind") == "eval" and r.get("panel") == "dev" and r.get("round") == RND]
    logged_pp = next((r["per_problem"] for r in M if r.get("kind") == "eval" and r.get("panel") == "dev"
                      and r.get("round") == RND), None)
    p = os.path.join(run_dir, "records", f"eval_dev_r{RND:04d}.jsonl.gz")
    R = recs(p)
    out = {"run_dir": run_dir, "attempt_id": M[0].get("attempt_id"), "starts": starts,
           "config": {k: cfg.get(k) for k in ("attempt_id", "max_tokens", "cap_schedule", "eval_rounds", "eval_panels",
                                              "seed", "bundle")},
           "dev_r32_logged_rows": rows, "records_file": p, "records_sha256": sha(p),
           "checkpoints": sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else None}
    if R:
        by = collections.defaultdict(lambda: [0, 0])
        for r in R:
            by[r["prompt_id"]][0] += r["reward"] == 1
            by[r["prompt_id"]][1] += 1
        out["recomputed"] = {
            "records": len(R), "problems": len(by), "correct": sum(c for c, _ in by.values()),
            "denominator": sum(n for _, n in by.values()),
            "pass1_macro": sum(c / n for c, n in by.values()) / len(by),
            "pass1_micro": sum(c for c, _ in by.values()) / sum(n for _, n in by.values()),
            "n_per_problem": sorted({n for _, n in by.values()}),
            "max_n_tokens": max(r["n_tokens"] for r in R), "capped": sum(r["finish"] == "length" for r in R),
            "capped_lengths": sorted({r["n_tokens"] for r in R if r["finish"] == "length"}),
            "status_counts": dict(collections.Counter(r["status"] for r in R)),
            "matches_logged_per_problem": logged_pp is not None and
            all(by[x["id"]] == [x["c"], x["n"]] for x in logged_pp) and len(logged_pp) == len(by)}
    return out, R


def first_divergence(a_dir, b_dir, upto=RND):
    for rnd in range(upto):
        A = recs(os.path.join(a_dir, "records", f"train_r{rnd:04d}.jsonl.gz"))
        B = recs(os.path.join(b_dir, "records", f"train_r{rnd:04d}.jsonl.gz"))
        if A is None or B is None:
            return {"round": rnd, "reason": "missing records", "a": A is not None, "b": B is not None}
        ka = {(r["slot"], r["sample"]): r for r in A}
        kb = {(r["slot"], r["sample"]): r for r in B}
        same_prompts = all(ka[k]["prompt_id"] == kb.get(k, {}).get("prompt_id") for k in ka)
        same_seeds = all(ka[k]["seed"] == kb.get(k, {}).get("seed") for k in ka)
        same_ids = sum(ka[k]["ids"] == kb.get(k, {}).get("ids") for k in ka)
        if not (same_prompts and same_seeds and same_ids == len(ka) == len(kb)):
            return {"round_0based": rnd, "same_prompts": same_prompts, "same_seeds": same_seeds,
                    "identical_responses": same_ids, "responses": len(ka)}
    return {"round_0based": None, "note": f"rounds 0..{upto - 1} identical"}


def main(proj, outp):
    res = {}
    for run in RUNS:
        a1, R1 = attempt(os.path.join(proj, "runs", "superseded", f"{run}-a1"),
                         os.path.join(proj, "ckpt", "superseded", f"{run}-a1"))
        a2, R2 = attempt(os.path.join(proj, "runs", run), os.path.join(proj, "ckpt", run))
        cmp = {}
        if R1 and R2:
            k1 = {(r["prompt_id"], r["sample"]): r for r in R1}
            k2 = {(r["prompt_id"], r["sample"]): r for r in R2}
            common = set(k1) & set(k2)
            cmp = {"same_keys": set(k1) == set(k2), "same_seeds": all(k1[k]["seed"] == k2[k]["seed"] for k in common),
                   "identical_responses": sum(k1[k]["ids"] == k2[k]["ids"] for k in common), "responses": len(common),
                   "reward_agree": sum(k1[k]["reward"] == k2[k]["reward"] for k in common)}
        cmp["train_first_divergence"] = first_divergence(a1["run_dir"], a2["run_dir"])
        res[run] = {"a1": a1, "a2": a2, "a1_vs_a2": cmp}
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
