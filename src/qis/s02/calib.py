"""Base-model calibration of candidate task pools (study Nov-15-2026-02). Sampling only; no weight changes.

Loads the model once in vLLM, then for each pool evaluates n samples per problem with the decoding of training
(temperature 1, top_p 1), and writes per pool: records (gz JSONL, under --out) and a summary (pass@1, per-stratum
counts of all-wrong / all-right problems, length quantiles, status counts, extracted-answer examples of wrong and
unparsed answers for the grader audit). The summary of all pools goes to --summary.

Usage: python -m qis.s02.calib --model M --bundle B --pools p1,p2 --n 16 --limit 64 --max-tokens 4096
         --out DIR --summary FILE.json [--ckpt DIR]
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--label", default="calib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pools = json.load(open(a.bundle))["pools"]
    from transformers import AutoTokenizer
    from qis.train.rollout import Rollout
    from qis.train.run import evaluate
    from qis.verify import Verifier
    import gzip
    tok = AutoTokenizer.from_pretrained(a.model)
    roll = Rollout(a.model, a.gpu_mem_util, max_tokens=a.max_tokens, max_num_seqs=a.max_num_seqs)
    ver = Verifier(4)
    res = {}
    for name in a.pools.split(","):
        items = pools[name][:a.limit]
        t0 = time.time()
        r = evaluate(roll, ver, tok, f"{a.label}_{name}", items, a.n, 0, a.out)
        recs = [json.loads(x) for x in gzip.open(os.path.join(a.out, f"eval_{a.label}_{name}_r0000.jsonl.gz"), "rt")]
        lens = [x["n_tokens"] for x in recs]
        st = {}
        for x in recs:
            st[x["status"]] = st.get(x["status"], 0) + 1
        gold = {it["id"]: it["gold"] for it in items}
        wrong_ex = [{"gold": gold[x["prompt_id"]], "extracted": x["extracted"], "status": x["status"]}
                    for x in recs if x["status"] in ("wrong", "no_answer")][:12]
        res[name] = {k: v for k, v in r.items() if k != "per_problem"}
        res[name].update(status_counts=st, len_q={q: float(np.quantile(lens, q)) for q in (0.1, 0.5, 0.9, 0.99, 1.0)},
                         wrong_examples=wrong_ex,
                         texts=[{"status": x["status"], "n_tokens": x["n_tokens"],
                                 "text": tok.decode(x["ids"], skip_special_tokens=True)[-1200:]}
                                for x in (recs[:1] + [y for y in recs if y["status"] != "correct"][:1])], per_problem_c=[p["c"] for p in r["per_problem"]],
                         wall_s=time.time() - t0)
        print(name, json.dumps({k: res[name][k] for k in ("pass1", "len_mean", "truncated", "wall_s")}), flush=True)
        json.dump(res, open(a.summary, "w"), indent=1)
    ver.close()


if __name__ == "__main__":
    main()
