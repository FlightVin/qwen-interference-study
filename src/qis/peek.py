"""Compact progress summary of every run under PROJ_DATA/runs (read-only; for monitoring).

Usage: python -m qis.peek RUNS_DIR OUT_JSON
"""
import json
import os
import sys


def main(runs, out):
    res = {}
    for name in sorted(os.listdir(runs)):
        p = os.path.join(runs, name, "metrics.jsonl")
        if not os.path.exists(p):
            continue
        rounds, evals, events = [], [], []
        for line in open(p):
            r = json.loads(line)
            if r["kind"] == "round":
                rounds.append({k: r.get(k) for k in ("round", "success", "groups_mixed", "groups_all0", "groups_all1",
                                                     "truncated", "len_mean", "clipfrac_pos", "clipfrac_neg", "t_gen",
                                                     "t_oldlp", "t_train", "t_round", "gpu_used_gib_gen",
                                                     "gpu_used_gib_train", "ckpt_busy", "opt_step", "gen_tokens")}
                              | {"mism_q99": r["logprob_mismatch_absq"].get("q0.99"),
                                 "mism_q999": r["logprob_mismatch_absq"].get("q0.999"),
                                 "ratio_q001": r["ratio_q"].get("q0.001"), "ratio_q999": r["ratio_q"].get("q0.999")})
            elif r["kind"] == "eval":
                evals.append({k: r.get(k) for k in ("panel", "round", "pass1", "pass8", "truncated", "unknown", "tokens", "seconds")})
            elif r["kind"] == "event":
                events.append({k: r.get(k) for k in ("what", "round", "utc", "ok", "error", "write_s", "job", "restart")})
        res[name] = {"rounds": rounds[-6:], "n_rounds": len(rounds), "evals": evals, "events": events[-12:],
                     "complete": os.path.exists(os.path.join(runs, name, "RUN_COMPLETE"))}
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
