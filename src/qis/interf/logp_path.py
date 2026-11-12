"""Log-probability of fixed base-policy answers along the line theta_base + alpha * (theta_run - theta_base)
(study Nov-12-2026-01).

Purpose 1, linearity: if training stayed in the first-order regime, the change of the mean token log-probability of a
fixed answer is proportional to alpha. Purpose 2, surrogate transfer: with base-policy samples and their advantages
A_i, S_D(run, alpha) = mean over mixed groups of domain D of (1/8) sum_i A_i * [lp_i(alpha) - lp_i(0)] is the change
of the GRPO objective of domain D caused by the run's weight change. It is positive when the run moved probability
towards correct answers of D, also for domains the run never trained on.

lp_i is the mean over response tokens of log pi(token), teacher forced, BF16 autocast with an FP32 LM head (the
training code path, `Learner.token_logprobs(train=False)`). Forward passes only.

Output: --out/lp.npz with `lp` [n_cond, n_attempts] float64 (mean token log-prob), `conds` (names "run@alpha", first
"base@0"), and meta.json with one row per attempt (label, prompt_id, sample, reward, n_tokens). Resumable per condition.

Usage: python -m qis.interf.logp_path --model BASE --out DIR --set hard=RECORDS:PANEL:key [...]
         --ckpt name=DIR [...] --alphas 0.5,1,2 [--samples 8] [--max-prompts N]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from qis import prompts
from qis.interf.grad_geom import load_set
from qis.train.learner import Learner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-prompts", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    alphas = [float(x) for x in a.alphas.split(",")]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    learner = Learner(a.model, lr=0.0)
    model = learner.model
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in model.named_parameters()]
    params = dict(model.named_parameters())
    base = {n: params[n].detach().to("cpu", copy=True) for n in names}

    attempts, meta = [], []
    for spec in a.set:
        sets = load_set(spec)
        if a.max_prompts:
            sets = sets[:a.max_prompts]
        for label, pid, problem, recs in sets:
            p_ids = tok(prompts.render(tok, problem), add_special_tokens=False)["input_ids"]
            for r in recs[:a.samples]:
                attempts.append((p_ids, r["ids"]))
                meta.append({"label": label, "prompt_id": pid, "sample": r["sample"], "reward": r["reward"],
                             "n_tokens": r["n_tokens"]})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    print("attempts", len(attempts), flush=True)

    npz = os.path.join(a.out, "lp.npz")
    conds, rows = [], []
    if os.path.exists(npz):
        z = np.load(npz)
        conds, rows = list(z["conds"]), list(z["lp"])
        assert rows[0].shape[0] == len(attempts)

    def run_cond(name):
        if name in conds:
            return
        t0 = time.time()
        out = np.zeros(len(attempts))
        with torch.no_grad():
            for i, (p_ids, ids) in enumerate(attempts):
                out[i] = float(learner.token_logprobs(p_ids, ids, train=False).double().mean())
        conds.append(name); rows.append(out)
        np.savez(npz + ".tmp.npz", conds=np.array(conds), lp=np.stack(rows))
        os.replace(npz + ".tmp.npz", npz)
        print(f"cond {name} mean lp {out.mean():.5f} {time.time() - t0:.0f}s", flush=True)

    run_cond("base@0")
    for c in a.ckpt:
        name, path = c.split("=", 1)
        if all(f"{name}@{al:g}" in conds for al in alphas):
            continue
        sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
        for al in alphas:
            cname = f"{name}@{al:g}"
            if cname in conds:
                continue
            with torch.no_grad():
                for n in names:
                    params[n].copy_((base[n] + al * (sd[n].float() - base[n])).to(params[n].device))
            run_cond(cname)
        del sd
    open(os.path.join(a.out, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
