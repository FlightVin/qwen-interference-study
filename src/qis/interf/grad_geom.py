"""Per-group GRPO gradients at one checkpoint, stored as count sketches (study Nov-12-2026-01).

Input: on-policy samples of the checkpoint written by `qis.evaluate_ckpt` (16 answers per problem). Each problem gives
two groups of 8 answers ("halves": samples 0-7 and 8-15). A group is *mixed* when it has 1-7 correct answers; only
mixed groups have a GRPO gradient. For a mixed group the gradient is

    g = (1/8) * sum_i A_i * grad_theta mean_t log pi_theta(y_it | x, y_i<t),   A_i = (r_i - mean r) / (std r + 1e-6),

which is the gradient of the training objective (`qis.train.grpo`) at ratio 1, that is, for samples drawn from the
current policy. The sign is that of ascent on reward. No optimizer step is taken and no weight changes.

Output in --out: `groups.jsonl` (one row per group, mixed or not), `sk_global.f32` (memmap [n_mixed, 2**log2_global]),
`sk_scaled.f32` (same, for the Adam-preconditioned gradient, only with --optim), `sk_type.f32`, `sk_layer.f32`, and
`sq.f64` (exact squared norms per module type and per layer). Row k of every array belongs to the k-th mixed group of
`groups.jsonl` (field `row`). The job is resumable: finished groups are skipped.

Adam preconditioning (--optim CKPT_DIR/optim.pt): with v the stored second-moment estimate, v_hat = v / (1 - beta2^t)
and P = 1 / (sqrt(v_hat) + eps), the scaled sketch holds g * sqrt(P). Its inner products are g_a^T P g_b, the
first-order effect of an Adam-preconditioned step along g_a on the objective of group b (momentum ignored).

Usage: python -m qis.interf.grad_geom --model BASE --ckpt DIR|base --out DIR
         --set hard=RECORDS.jsonl.gz:PANEL.json:hard [--set ...] [--optim FILE] [--max-prompts N]
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import torch

from qis import prompts
from qis.interf.sketch import Sketcher
from qis.train import grpo
from qis.train.learner import Learner

N_HALF = 8


def load_set(spec):
    label, rest = spec.split("=", 1)
    rec_path, panel_path, key = rest.rsplit(":", 2)
    items = {it["id"]: it for it in json.load(open(panel_path))["pools"][key]}
    by = {}
    with gzip.open(rec_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            by.setdefault(r["prompt_id"], []).append(r)
    out = []
    for pid, recs in by.items():
        recs.sort(key=lambda r: r["sample"])
        out.append((label, pid, items[pid]["problem"], recs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--optim", default=None)
    ap.add_argument("--max-prompts", type=int, default=0, help="per set; 0 = all (canary: small number)")
    ap.add_argument("--log2-global", type=int, default=21)
    ap.add_argument("--log2-cat", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    learner = Learner(a.model, lr=0.0)
    model = learner.model
    if a.ckpt != "base":
        sd = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        del sd
    params = list(model.parameters())
    names = [n for n, _ in model.named_parameters()]
    sk = Sketcher(names, a.log2_global, a.log2_cat)

    scale = None
    if a.optim:
        o = torch.load(a.optim, map_location="cpu", weights_only=False)["optim"]
        st, pg = o["state"], o["param_groups"][0]
        b2, eps = pg["betas"][1], pg["eps"]
        assert len(st) == len(params), (len(st), len(params))
        scale = []
        for i, p in enumerate(params):
            v = st[i]["exp_avg_sq"].to("cuda", torch.float32)
            assert v.shape == p.shape
            vhat = v / (1.0 - b2 ** float(st[i]["step"]))
            scale.append((vhat.sqrt() + eps).rsqrt())            # sqrt(P)
        del o, st

    # ---- group list
    groups = []
    for spec in a.set:
        sets = load_set(spec)
        if a.max_prompts:
            sets = sets[:a.max_prompts]
        for label, pid, problem, recs in sets:
            for h in range(len(recs) // N_HALF):
                groups.append((label, pid, problem, h, recs[h * N_HALF:(h + 1) * N_HALF]))
    rewards = [torch.tensor([[r["reward"] for r in g[4]]], dtype=torch.float32) for g in groups]
    mixed = [bool(0 < float(r.sum()) < N_HALF) for r in rewards]
    n_mixed = sum(mixed)
    print(f"groups {len(groups)} mixed {n_mixed}", flush=True)

    def mm(name, shape, dtype):
        path = os.path.join(a.out, name)
        mode = "r+" if os.path.exists(path) else "w+"
        return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape) if mode == "w+" \
            else np.load(path, mmap_mode="r+")
    A_g = mm("sk_global.npy", (n_mixed, sk.mg), np.float32)
    A_t = mm("sk_type.npy", (n_mixed, len(sk.types), sk.mc), np.float32)
    A_l = mm("sk_layer.npy", (n_mixed, len(sk.layers), sk.mc), np.float32)
    A_q = mm("sq.npy", (n_mixed, len(sk.types) + len(sk.layers) + 1), np.float64)
    A_s = mm("sk_scaled.npy", (n_mixed, sk.mg), np.float32) if scale is not None else None

    gpath = os.path.join(a.out, "groups.jsonl")
    done = {}
    if os.path.exists(gpath):
        for line in open(gpath):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(r["label"], r["prompt_id"], r["half"])] = r
    json.dump({"ckpt": a.ckpt, "optim": a.optim, "types": sk.types, "layers": sk.layers, "names": names,
               "log2_global": a.log2_global, "log2_cat": a.log2_cat, "seed": sk.seed, "n_mixed": n_mixed,
               "n_groups": len(groups), "sets": a.set}, open(os.path.join(a.out, "meta.json"), "w"))

    row, t0 = -1, time.time()
    with open(gpath, "a") as gf:
        for gi, (label, pid, problem, h, recs) in enumerate(groups):
            if mixed[gi]:
                row += 1
            key = (label, pid, h)
            if key in done:
                assert done[key].get("row", -1) == (row if mixed[gi] else -1), "group order changed between attempts"
                continue
            info = {"label": label, "prompt_id": pid, "half": h, "n_correct": int(rewards[gi].sum()),
                    "n_tokens": [r["n_tokens"] for r in recs], "row": row if mixed[gi] else -1}
            if mixed[gi]:
                adv = grpo.group_advantages(rewards[gi])[0]
                p_ids = tok(prompts.render(tok, problem), add_special_tokens=False)["input_ids"]
                model.zero_grad(set_to_none=True)
                obj = 0.0
                for j, r in enumerate(recs):
                    lp = learner.token_logprobs(p_ids, r["ids"], train=True)
                    term = float(adv[j]) * lp.mean() / N_HALF
                    term.backward()                               # ascent direction on reward
                    obj += float(term)
                grads = [p.grad for p in params]
                s = sk.sketch(grads, scale=scale)
                A_g[row] = s["global"].cpu().numpy()
                A_t[row] = s["type"].cpu().numpy()
                A_l[row] = s["layer"].cpu().numpy()
                sq = torch.cat([s["sq_type"], s["sq_layer"],
                                (s["sq_scaled"] if scale is not None else torch.zeros((), dtype=torch.float64,
                                                                                        device="cuda")).reshape(1)])
                A_q[row] = sq.cpu().numpy()
                if A_s is not None:
                    A_s[row] = s["global_scaled"].cpu().numpy()
                info["grad_norm"] = float(s["sq_type"].sum().sqrt())
                info["finite"] = bool(torch.isfinite(s["global"]).all())
                if (row + 1) % 10 == 0:
                    for arr in (A_g, A_t, A_l, A_q, A_s):
                        if arr is not None:
                            arr.flush()
                    print(f"mixed group {row + 1}/{n_mixed} label {label} |g| {info['grad_norm']:.4g} "
                          f"elapsed {time.time() - t0:.0f}s", flush=True)
                del grads, s
            if mixed[gi]:
                for arr in (A_g, A_t, A_l, A_q, A_s):
                    if arr is not None and (row + 1) % 10 == 0:
                        arr.flush()
            gf.write(json.dumps(info) + "\n")
            if mixed[gi] and (row + 1) % 10 == 0:
                gf.flush()
    for arr in (A_g, A_t, A_l, A_q, A_s):
        if arr is not None:
            arr.flush()
    open(os.path.join(a.out, "DONE"), "w").write(f"{n_mixed}\n")
    print("done", n_mixed, "mixed groups", flush=True)


if __name__ == "__main__":
    main()
