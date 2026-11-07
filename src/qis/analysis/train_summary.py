"""Per-run training summary in blocks of 8 rounds (read-only; run in a CPU job).

From metrics.jsonl: success by role, mixed/all-0/all-1 groups, capped responses, mean length, clip fractions,
probability-ratio tails, logprob mismatch, round time; from step records: grad norm (pre-clip), share clipped,
policy loss. Also lists scheduled evaluations. Usage: python -m qis.analysis.train_summary RUNS_DIR OUT.json
"""
import glob
import json
import os
import sys

import numpy as np


def main(runs, out):
    res = {}
    for d in sorted(glob.glob(os.path.join(runs, "qis-main-*"))):
        M = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
        R = {r["round"]: r for r in M if r.get("kind") == "round"}
        S = [r for r in M if r.get("kind") == "step"]
        blocks = []
        for b in range(0, max(R) if R else 0, 8):
            rs = [R[i] for i in range(b + 1, b + 9) if i in R]
            if not rs:
                continue
            st = [s for s in S if b <= s["round"] < b + 8]
            roles = sorted({k for r in rs for k in r["success"]})
            blocks.append({"rounds": f"{b + 1}-{b + len(rs)}",
                           "success": {k: round(float(np.mean([r["success"][k] for r in rs if k in r["success"]])), 3) for k in roles},
                           "mixed": round(float(np.mean([r["groups_mixed"] for r in rs])), 2),
                           "all0": round(float(np.mean([r["groups_all0"] for r in rs])), 2),
                           "all1": round(float(np.mean([r["groups_all1"] for r in rs])), 2),
                           "capped_per_round": round(float(np.mean([r["truncated"] for r in rs])), 2),
                           "len": round(float(np.mean([r["len_mean"] for r in rs]))),
                           "clipfrac_pos": float(np.mean([r["clipfrac_pos"] for r in rs])),
                           "clipfrac_neg": float(np.mean([r["clipfrac_neg"] for r in rs])),
                           "ratio_q001": round(float(np.mean([r["ratio_q"]["q0.001"] if "ratio_q" in r else r.get("ratio_q001", np.nan) for r in rs])), 3),
                           "ratio_q999": round(float(np.mean([r["ratio_q"]["q0.999"] if "ratio_q" in r else r.get("ratio_q999", np.nan) for r in rs])), 3),
                           "mism_q99": round(float(np.mean([r["logprob_mismatch_absq"]["q0.99"] if "logprob_mismatch_absq" in r else np.nan for r in rs])), 3),
                           "entropy": (round(float(np.nanmean([r.get("entropy_sample_mean", np.nan) for r in rs])), 3) if any("entropy_sample_mean" in r for r in rs) else None),
                           "t_round": round(float(np.mean([r["t_round"] for r in rs]))),
                           "grad_norm": round(float(np.mean([s["grad_norm_preclip"] for s in st])), 3) if st else None,
                           "grad_clipped_share": round(float(np.mean([s["grad_clipped"] for s in st])), 3) if st else None,
                           "policy_loss": round(float(np.mean([s["policy_loss"] for s in st])), 4) if st else None})
        ev = [{k: r.get(k) for k in ("panel", "round", "pass1", "pass8", "truncated", "tokens", "seconds")} for r in M if r.get("kind") == "eval"]
        res[os.path.basename(d)] = {"rounds_done": max(R) if R else 0, "blocks": blocks, "evals": ev}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v["rounds_done"] for k, v in res.items()}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
