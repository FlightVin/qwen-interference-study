"""Difficulty screening and length probe (project.md Section 5 "Difficulty screening").

Generates n samples per item with the frozen model/prompt/decoding/cap, scores them with the
verifier, and writes per-item counts plus per-attempt records. Streams are named by stage, so
stage 1 (16), stage 2 (32) and stage 3 (16, reporting only) use independent samples.

Usage: python -m qis.screen --items ITEMS.jsonl --stage NAME --n N --out DIR --model M
       [--max-tokens 4096] [--chunk 64]
Resumable: items already present in DIR/<stage>_counts.jsonl are skipped.
"""
import argparse
import gzip
import json
import os
import time

import numpy as np

from qis import prompts, seeds
from qis.verify import UNKNOWN, Verifier

SCREEN_SEED = 1701


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="jsonl: id, problem, gold")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    counts_path = os.path.join(a.out, f"{a.stage}_counts.jsonl")
    done = set()
    if os.path.exists(counts_path):
        done = {json.loads(line)["id"] for line in open(counts_path)}
    items = [json.loads(line) for line in open(a.items)]
    todo = [it for it in items if it["id"] not in done]
    print(f"stage={a.stage} items={len(items)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    from qis.train.rollout import Rollout
    roll = Rollout(a.model, a.gpu_mem_util, max_tokens=a.max_tokens, max_num_seqs=512)
    ver = Verifier(8)
    over_cap = []
    for c in range(0, len(todo), a.chunk):
        chunk = todo[c:c + a.chunk]
        ids, keep = [], []
        for it in chunk:
            p = tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"]
            if len(p) > prompts.PROMPT_CAP:
                over_cap.append(it["id"])
                continue
            ids.append(p); keep.append(it)
        req = [seeds.int_seed(SCREEN_SEED, f"screen/{a.stage}/{it['id']}") for it in keep]
        t0 = time.time()
        outs = roll.generate(ids, a.n, req, logprobs=False)
        gen_s = time.time() - t0
        flat = [(it, o) for it, g in zip(keep, outs) for o in g]
        sc = ver.score([{"gold": it["gold"], "response": tok.decode(o["ids"], skip_special_tokens=True),
                         "truncated": o["finish"] == "length"} for it, o in flat])
        k = 0
        with open(counts_path, "a") as fc, gzip.open(os.path.join(a.out, f"{a.stage}_records_{c:05d}.jsonl.gz"), "wt") as fr:
            for it, p, s, g in zip(keep, ids, req, outs):
                rs = sc[k:k + a.n]; k += a.n
                lens = [len(o["ids"]) for o in g]
                fc.write(json.dumps({
                    "id": it["id"], "stage": a.stage, "n": a.n, "seed": s,
                    "c": sum(r["reward"] == 1 for r in rs),
                    "unknown": sum(r["status"] == UNKNOWN for r in rs),
                    "truncated": sum(o["finish"] == "length" for o in g),
                    "truncated_would_match": sum(bool(r.get("would_match")) for r in rs),
                    "no_answer": sum(r["status"] == "no_answer" for r in rs),
                    "prompt_tokens": len(p), "resp_len_mean": float(np.mean(lens)),
                    "resp_len_max": int(max(lens))}) + "\n")
                for j, (o, r) in enumerate(zip(g, rs)):
                    fr.write(json.dumps({"id": it["id"], "sample": j, "seed": s, "n_tokens": len(o["ids"]),
                                         "finish": o["finish"], "status": r["status"], "reward": r["reward"],
                                         "extracted": r["extracted"], "ids": o["ids"]}) + "\n")
        ntok = sum(len(o["ids"]) for _, o in flat)
        print(f"chunk {c}: {len(keep)} items, {ntok} tokens, {gen_s:.0f}s gen, {ntok / gen_s:.0f} tok/s", flush=True)
    with open(os.path.join(a.out, f"{a.stage}_over_cap.json"), "w") as f:
        json.dump(over_cap, f)
    ver.close()


if __name__ == "__main__":
    main()
