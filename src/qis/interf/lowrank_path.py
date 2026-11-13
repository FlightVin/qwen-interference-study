"""How many directions of a weight update carry its effect? (study Nov-12-2026-01, capacity question). No training.

For an update tau = theta_ckpt - theta_base, each 2-D weight matrix W has an update D_W. Its rank-r part top_r(D_W)
is the best rank-r approximation (truncated SVD). The tool measures the log-probability of fixed base-policy answers
(as `logp_path.py`, FP32 weights and FP32 compute) at
  name:full     theta_base + tau
  name:top<r>   theta_base + [top_r(D_W) for every 2-D matrix; all other parameters at their base value]
  name:tail<r>  theta_base + tau - (the top<r> update)
If top<r> reproduces the effect of the full update and tail<r> has none, the learning used at most r directions per
matrix. The squared-norm share of each part is written to shares.json.

The rank-r part is computed exactly by projection on the top eigenvectors of the Gram matrix of the smaller side
(D^T D or D D^T, float64), which equals the truncated SVD.

Output: --out/lp.npz (`lp` [n_cond, n_attempts], `conds`), meta.json (one row per attempt), shares.json.
Usage: python -m qis.interf.lowrank_path --model BASE --out DIR --set label=RECORDS:PANEL:key [...]
         --ckpt name=DIR [...] --ranks 1,8,64 [--samples 8]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from qis import prompts
from qis.interf.grad_geom import load_set
from qis.interf.logp_path import fp32_logprob
from qis.train.learner import Learner


def top_basis(d, rmax):
    """Orthonormal basis [n_small, rmax] of the top singular subspace on the smaller side of d, and all eigenvalues."""
    x = d.double()
    right = x.shape[1] <= x.shape[0]
    g = x.T @ x if right else x @ x.T
    ev, vec = torch.linalg.eigh(g)
    return right, vec[:, -rmax:].flip(1).float().contiguous(), ev.flip(0).clamp(min=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--ranks", default="1,8,64")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-prompts", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ranks = sorted(int(x) for x in a.ranks.split(","))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    learner = Learner(a.model, lr=0.0)
    model = learner.model
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in model.named_parameters()]
    params = dict(model.named_parameters())
    dev = next(model.parameters()).device
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

    npz, shp = os.path.join(a.out, "lp.npz"), os.path.join(a.out, "shares.json")
    conds, rows = [], []
    if os.path.exists(npz):
        z = np.load(npz)
        conds, rows = [str(c) for c in z["conds"]], list(z["lp"])
        assert rows[0].shape[0] == len(attempts)
    shares = json.load(open(shp)) if os.path.exists(shp) else {}

    def run_cond(name):
        t0 = time.time()
        out = np.zeros(len(attempts))
        with torch.no_grad():
            for i, (p_ids, ids) in enumerate(attempts):
                out[i] = fp32_logprob(model, p_ids, ids)
        conds.append(name); rows.append(out)
        np.savez(npz + ".tmp.npz", conds=np.array(conds), lp=np.stack(rows))
        os.replace(npz + ".tmp.npz", npz)
        print(f"cond {name} mean lp {out.mean():.5f} {time.time() - t0:.0f}s", flush=True)

    if "base@0" not in conds:
        run_cond("base@0")
    for c in a.ckpt:
        name, path = c.split("=", 1)
        want = [f"{name}:full"] + [f"{name}:{kind}{r}" for r in ranks for kind in ("top", "tail")]
        if all(w in conds for w in want):
            continue
        sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
        t0 = time.time()
        basis, tot2, mat2, top2, srank = {}, 0.0, 0.0, {r: 0.0 for r in ranks}, []
        for n in names:
            d = (sd[n].float() - base[n]).to(dev)
            n2 = float(d.double().pow(2).sum())
            tot2 += n2
            if d.dim() == 2 and min(d.shape) > max(ranks):
                right, v, ev = top_basis(d, max(ranks))
                basis[n] = (right, v.cpu())
                mat2 += n2
                for r in ranks:
                    top2[r] += float(ev[:r].sum())
                srank.append(float(ev.sum() / ev[0]))
            del d
        shares[name] = {"norm": tot2 ** 0.5, "share_2d": mat2 / tot2, "n_matrices": len(basis),
                        "median_stable_rank": float(np.median(srank)),
                        "share_top": {str(r): top2[r] / tot2 for r in ranks}}
        json.dump(shares, open(shp, "w"), indent=1)
        print(f"svd {name} {time.time() - t0:.0f}s {json.dumps(shares[name])}", flush=True)
        for w in want:
            if w in conds:
                continue
            kind = w.split(":")[1]
            with torch.no_grad():
                for n in names:
                    d = (sd[n].float() - base[n]).to(dev)
                    if kind != "full":
                        r = int(kind.replace("top", "").replace("tail", ""))
                        if n in basis:
                            right, v = basis[n]
                            v = v[:, :r].to(dev)
                            top = (d @ v) @ v.T if right else v @ (v.T @ d)
                        else:
                            top = torch.zeros_like(d)
                        d = top if kind.startswith("top") else d - top
                    params[n].copy_(base[n].to(dev) + d)
            run_cond(w)
        del sd, basis
    open(os.path.join(a.out, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
