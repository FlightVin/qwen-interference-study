"""Length diagnostic (A19 §2, D154; EXPLORATORY): continue capped attempts of a frozen evaluation to a longer cap.

Each selected attempt keeps its original record. An attempt that hit the first cap (finish "length", exactly FROM_CAP
response tokens) is continued from its saved tokens: prompt ids + original response ids go back to vLLM as the prompt,
with TO_CAP - FROM_CAP new tokens allowed, so the total response budget is TO_CAP and the context is PROMPT_CAP + TO_CAP.
Finished attempts are not regenerated. Scores: at FROM_CAP = the original reward (re-graded here and asserted equal); at
TO_CAP = the same grading rule on the full trajectory (a response still capped gets reward 0). Decoding as evaluate_ckpt
(T 1.0, top_p 1.0); one request per attempt, seed from EVAL_SEED keyed by panel, problem, sample and seed label, so all
checkpoints share one attempt schedule.
Selection (`select`): per tier, sources by largest remainder, items by sha256(key/id) order; independent of outcomes.
Usage: python -m qis.eval_extend --model M --ckpt C --panel P --panel-key stress --records R.jsonl.gz --out DIR
       [--select tranche1 | --select rest --exclude DIR] [--from-cap 12288 --to-cap 16384]
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import time

KEY = "a19-length-v1"


def select(items, per_tier=32, attempts=4, key=KEY):
    """Fixed stratified sample: [(prompt_id, sample)], per tier `per_tier` items over sources, samples 0..attempts-1."""
    out = []
    for tier in sorted({it["tier"] for it in items}):
        t = [it for it in items if it["tier"] == tier]
        src = collections.Counter(it["source_collection"] for it in t)
        quota = {s: per_tier * n // len(t) for s, n in src.items()}
        rem = sorted(src, key=lambda s: (-(per_tier * src[s] % len(t)), s))
        for s in rem[:per_tier - sum(quota.values())]:
            quota[s] += 1
        for s in sorted(src):
            pool = sorted((it["id"] for it in t if it["source_collection"] == s),
                          key=lambda i: hashlib.sha256(f"{key}/{i}".encode()).hexdigest())
            out += [(i, j) for i in pool[:quota[s]] for j in range(attempts)]
    return out


def plan(recs, sel, from_cap):
    """Check the selected original records and split them into kept and to-extend. recs: {(pid, sample): rec}."""
    keep, ext = [], []
    for k in sel:
        r = recs[k]
        if r["finish"] == "length":
            assert r["n_tokens"] == from_cap and r["reward"] == 0, f"{k}: capped record has {r['n_tokens']} tokens"
            ext.append(k)
        else:
            assert r["n_tokens"] <= from_cap, f"{k}: finished record longer than the cap"
            keep.append(k)
    return keep, ext


def combine(rec, new, score, to_cap):
    """Row for one attempt. new: continuation dict (ids, finish) or None; score: verifier result at TO_CAP or None."""
    row = {"prompt_id": rec["prompt_id"], "sample": rec["sample"], "n_tokens_1": rec["n_tokens"], "finish_1": rec["finish"],
           "reward_1": rec["reward"], "status_1": rec["status"], "extended": new is not None}
    if new is None:
        return {**row, "n_tokens_2": rec["n_tokens"], "finish_2": rec["finish"], "reward_2": rec["reward"],
                "status_2": rec["status"], "new_tokens": 0}
    n = rec["n_tokens"] + len(new["ids"])
    assert n <= to_cap, f"{rec['prompt_id']}/{rec['sample']}: {n} tokens > {to_cap}"
    return {**row, "n_tokens_2": n, "finish_2": new["finish"], "reward_2": score["reward"], "status_2": score["status"],
            "extracted_2": score["extracted"], "new_tokens": len(new["ids"]), "new_ids": new["ids"]}


def main():
    ap = argparse.ArgumentParser()
    for f in ("--model", "--ckpt", "--panel", "--panel-key", "--records", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--select", default="tranche1", choices=("tranche1", "rest"))
    ap.add_argument("--exclude", default=None, help="output dir of an earlier tranche; its attempts are skipped")
    ap.add_argument("--seed-label", default="extend-v1")
    ap.add_argument("--from-cap", type=int, default=12288)
    ap.add_argument("--to-cap", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer

    from qis import prompts, seeds
    from qis.train.rollout import Rollout
    from qis.train.run import EVAL_SEED
    from qis.verify import Verifier
    os.makedirs(a.out, exist_ok=True)
    items = {it["id"]: it for it in json.load(open(a.panel))["pools"][a.panel_key]}
    recs = {}
    with gzip.open(a.records, "rt") as f:
        for line in f:
            r = json.loads(line)
            recs[(r["prompt_id"], r["sample"])] = r
    sel = select(list(items.values()))
    if a.select == "rest":
        done = set()
        with gzip.open(os.path.join(a.exclude, "rows.jsonl.gz"), "rt") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["prompt_id"], r["sample"]))
        assert done == set(sel), "--exclude is not the tranche-1 output"
        sel = sorted(k for k in recs if k not in done)
    keep, ext = plan(recs, sel, a.from_cap)
    tok = AutoTokenizer.from_pretrained(a.model)
    ver = Verifier(8)
    # grading rule preserved: re-grading the original tokens must reproduce every stored reward
    rg = ver.score([{"gold": items[k[0]]["gold"], "response": tok.decode(recs[k]["ids"], skip_special_tokens=True),
                     "truncated": recs[k]["finish"] == "length"} for k in sel])
    bad = [k for k, s in zip(sel, rg) if s["reward"] != recs[k]["reward"]]
    assert not bad, f"re-grade differs from stored reward: {bad[:5]}"
    p_ids = {}
    for pid, _ in ext:
        if pid not in p_ids:
            p_ids[pid] = tok(prompts.render(tok, items[pid]["problem"]), add_special_tokens=False)["input_ids"]
            assert len(p_ids[pid]) <= prompts.PROMPT_CAP
    roll = Rollout(a.model, a.to_cap, a.gpu_mem_util)
    ck_hash = "base"
    if a.ckpt != "base":
        mpath = os.path.join(a.ckpt, "model.pt")
        ck_hash = json.load(open(os.path.join(a.ckpt, "state.json")))["file_sha256"]["model.pt"]
        sd = torch.load(mpath, map_location="cpu", weights_only=True)
        params = [(n, t.to(torch.bfloat16)) for n, t in sd.items()]
        roll.llm.collective_rpc(lambda w: w.model_runner.model.load_weights(iter(params)) and True)
        names = [n for n in ("model.embed_tokens.weight", "model.norm.weight", "model.layers.0.input_layernorm.weight",
                             "model.layers.27.mlp.down_proj.weight") if n in sd]
        want = {n: float(sd[n].to(torch.bfloat16).float().sum()) for n in names}

        def _readback(w):
            ps = dict(w.model_runner.model.named_parameters())
            return {n: float(ps[n].detach().float().sum()) for n in names if n in ps}
        got = roll.llm.collective_rpc(_readback)[0]
        assert got and all(abs(want[n] - got[n]) <= 1e-3 * max(1.0, abs(want[n])) for n in got), "weight load mismatch"
        print("weight check ok", flush=True)
        del sd, params
    t0 = time.time()
    req = [seeds.int_seed(EVAL_SEED, f"extend/{a.panel_key}/{pid}/{a.seed_label}/s{j}") for pid, j in ext]
    outs = roll.generate([p_ids[pid] + recs[(pid, j)]["ids"] for pid, j in ext], 1, req, logprobs=False,
                         max_tokens=a.to_cap - a.from_cap) if ext else []
    new = {k: o[0] for k, o in zip(ext, outs)}
    sc = ver.score([{"gold": items[k[0]]["gold"],
                     "response": tok.decode(recs[k]["ids"] + new[k]["ids"], skip_special_tokens=True),
                     "truncated": new[k]["finish"] == "length"} for k in ext])
    score = dict(zip(ext, sc))
    rows = [combine(recs[k], new.get(k), score.get(k), a.to_cap) for k in sel]
    with gzip.open(os.path.join(a.out, "rows.jsonl.gz"), "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    summ = {"ckpt": a.ckpt, "ckpt_model_sha256": ck_hash, "panel": a.panel_key, "select": a.select, "key": KEY,
            "seed_label": a.seed_label, "from_cap": a.from_cap, "to_cap": a.to_cap, "attempts": n,
            "problems": len({r["prompt_id"] for r in rows}), "capped_1": len(ext),
            "correct_1": sum(r["reward_1"] == 1 for r in rows), "correct_2": sum(r["reward_2"] == 1 for r in rows),
            "recovered": sum(r["extended"] and r["reward_2"] == 1 for r in rows),
            "finished_wrong": sum(r["extended"] and r["finish_2"] != "length" and r["reward_2"] != 1 for r in rows),
            "still_capped": sum(r["extended"] and r["finish_2"] == "length" for r in rows),
            "new_tokens": sum(r["new_tokens"] for r in rows), "seconds": time.time() - t0,
            "gpu": torch.cuda.get_device_name(0), "sampling": {"temperature": 1.0, "top_p": 1.0, "n": 1}}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    ver.close()


if __name__ == "__main__":
    main()
