"""Evaluate a checkpoint (or the base model) on a panel: n samples per problem, frozen decoding.

Used for the sealed final test evaluation (baseline and endpoint, 32 samples/problem) and any
re-evaluation. Evaluation seeds come from the EVAL_SEED stream keyed by panel, problem and a label
(`--seed-label`, e.g. "test-final"), independent of training and selection streams.
Writes per-attempt records (gz JSONL) and a summary with per-problem counts and the checkpoint hash.
Usage: python -m qis.evaluate_ckpt --model BASE --ckpt DIR|base --panel FILE.json --panel-key test
       --n 32 --out DIR --seed-label test-final [--max-tokens 8192]
"""
import argparse
import gzip
import hashlib
import json
import os
import time

import numpy as np
import torch

from qis import prompts, seeds
from qis.train import grpo
from qis.train.run import EVAL_SEED
from qis.verify import UNKNOWN, Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--panel-key", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-label", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.panel))["pools"][a.panel_key]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    from qis.train.rollout import Rollout
    # vLLM gets most of the GPU; checkpoint weights are read on the CPU and pushed into vLLM (no learner on GPU).
    roll = Rollout(a.model, a.max_tokens, a.gpu_mem_util)
    ck_hash = "base"
    if a.ckpt != "base":
        mpath = os.path.join(a.ckpt, "model.pt")
        ck_hash = json.load(open(os.path.join(a.ckpt, "state.json"))).get("file_sha256", {}).get("model.pt") \
            or hashlib.sha256(open(mpath, "rb").read()).hexdigest()
        sd = torch.load(mpath, map_location="cpu", weights_only=True)
        params = [(n, t.to(torch.bfloat16)) for n, t in sd.items()]
        roll.llm.collective_rpc(lambda w: w.model_runner.model.load_weights(iter(params)) and True)
        del sd, params
    ver = Verifier(8)
    ids = [tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"] for it in items]
    req = [seeds.int_seed(EVAL_SEED, f"eval/{a.panel_key}/{it['id']}/{a.seed_label}/n{a.n}") for it in items]
    t0 = time.time()
    outs = roll.generate(ids, a.n, req, logprobs=False)
    sc = ver.score([{"gold": it["gold"], "response": tok.decode(o["ids"], skip_special_tokens=True),
                     "truncated": o["finish"] == "length"} for it, g in zip(items, outs) for o in g])
    per, k, unknown, toks = [], 0, 0, 0
    c8 = []          # A10: success with completion required within 8,192 tokens (same traces)
    with gzip.open(os.path.join(a.out, "records.jsonl.gz"), "wt") as f:
        for it, g, s in zip(items, outs, req):
            c, c_8k = 0, 0
            for j, o in enumerate(g):
                r = sc[k]; k += 1
                unknown += r["status"] == UNKNOWN; c += r["reward"] == 1; toks += len(o["ids"])
                c_8k += r["reward"] == 1 and len(o["ids"]) <= 8192
                f.write(json.dumps({"prompt_id": it["id"], "seed": s, "sample": j, "n_tokens": len(o["ids"]),
                                    "finish": o["finish"], "extracted": r["extracted"], "status": r["status"],
                                    "reward": r["reward"], "ids": o["ids"]}) + "\n")
            per.append({"id": it["id"], "c": c, "c_within_8192": c_8k, "n": a.n})
    summ = {"ckpt": a.ckpt, "ckpt_model_sha256": ck_hash, "panel": a.panel_key, "n": a.n, "problems": len(items),
            "pass1": float(np.mean([p["c"] / a.n for p in per])),
            "pass8": float(np.mean([grpo.pass_at_k(a.n, p["c"], 8) for p in per])),
            "pass1_within_8192": float(np.mean([p["c_within_8192"] / a.n for p in per])),
            "over_8192": sum(len(o["ids"]) > 8192 for g in outs for o in g) / (a.n * len(items)), "max_tokens": a.max_tokens,
            "over_8192_then_correct": sum(r["reward"] == 1 and len(o["ids"]) > 8192
                                          for r, o in zip(sc, [o for g in outs for o in g])) / (a.n * len(items)),
            "tokens_per_second": toks / max(time.time() - t0, 1e-9), "completions": a.n * len(items),
            "truncated": sum(o["finish"] == "length" for g in outs for o in g) / (a.n * len(items)),
            "unknown": unknown, "tokens": toks, "seconds": time.time() - t0, "seed_label": a.seed_label,
            "per_problem": per}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_problem"}, indent=1))
    ver.close()


if __name__ == "__main__":
    main()
