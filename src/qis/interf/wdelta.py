"""Weight-update geometry of stored checkpoints (study Nov-12-2026-01).

For each run checkpoint the update is tau = theta - theta_base (FP32, tied lm_head dropped). The script reports, per
parameter tensor and for all pairs of runs: inner products of updates (a Gram matrix), base norms, sign agreement,
coordinate statistics, the share of coordinates whose BF16 value changed, and for every 2-D block matrix the singular
values of the update and the overlap of the top-k singular subspaces between runs. It also writes a count sketch of
every update (same hash as `qis.interf.grad_geom`) so that <gradient, update> can be estimated later.

Nothing is trained and no checkpoint is modified.

Usage: python -m qis.interf.wdelta --model BASE --out DIR --ckpt name=DIR [--ckpt ...] [--lr 2e-6] [--k 8]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from qis.interf.sketch import Sketcher, categories

QS = (0.5, 0.9, 0.99, 0.999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-svd", action="store_true")
    ap.add_argument("--log2-global", type=int, default=21)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    names = [n for n, _ in base_model.named_parameters()]          # tied lm_head appears once (as the embedding)
    base = {n: p.detach().clone() for n, p in base_model.named_parameters()}
    del base_model
    runs = [c.split("=", 1) for c in a.ckpt]
    R = len(runs)
    delta = []
    for name, path in runs:
        sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
        assert all(n in sd for n in names), "parameter names differ from the base model"
        delta.append({n: (sd[n].float() - base[n]) for n in names})
        del sd
        print("loaded", name, flush=True)

    sk = Sketcher(names, a.log2_global, 15, device=dev)
    sketches = np.zeros((R, sk.mg), np.float32)
    for r in range(R):
        s = sk.sketch([delta[r][n] for n in names], cats=False)
        sketches[r] = s["global"].cpu().numpy()
    np.save(os.path.join(a.out, "sk_delta.npy"), sketches)

    per = []
    t0 = time.time()
    for ti, n in enumerate(names):
        numel = base[n].numel()
        gram = torch.zeros(R, R, dtype=torch.float64, device=dev)
        both = torch.zeros(R, R, dtype=torch.float64, device=dev)
        ss = torch.zeros(R, R, dtype=torch.float64, device=dev)
        n_zero = torch.zeros(R, dtype=torch.float64, device=dev)
        abs_sum = torch.zeros(R, dtype=torch.float64, device=dev)
        abs_max = torch.zeros(R, device=dev)
        bf_changed = torch.zeros(R, dtype=torch.float64, device=dev)
        bf_dot = torch.zeros(R, dtype=torch.float64, device=dev)
        bf_sq = torch.zeros(R, dtype=torch.float64, device=dev)
        base_sq, samp = 0.0, []
        bflat = base[n].reshape(-1)
        CH = 1 << 24                                               # coordinates per chunk (memory bound: R x CH floats)
        for s0 in range(0, numel, CH):
            bb = bflat[s0:s0 + CH].to(dev)
            D = torch.stack([delta[r][n].reshape(-1)[s0:s0 + CH].to(dev) for r in range(R)])
            Dd = D.double()
            gram += Dd @ Dd.t()
            sgn = torch.sign(D).double()
            nz = (sgn != 0).double()
            both += nz @ nz.t(); ss += sgn @ sgn.t()
            n_zero += (1 - nz).sum(1)
            absD = D.abs()
            abs_sum += absD.double().sum(1); abs_max = torch.maximum(abs_max, absD.max(1).values)
            k_s = max(1, int(2_000_000 * D.shape[1] / numel))
            samp.append(absD[:, torch.randint(0, D.shape[1], (k_s,), device=dev)])
            base_sq += float((bb.double() ** 2).sum())
            bf_base = bb.to(torch.bfloat16)
            newbf = (bb[None, :] + D).to(torch.bfloat16)
            bf_changed += (newbf != bf_base[None, :]).double().sum(1)
            dbf = (newbf.float() - bf_base.float()[None, :]).double()   # the update as BF16 generation sees it
            bf_dot += (dbf * Dd).sum(1); bf_sq += (dbf ** 2).sum(1)
            del bb, D, Dd, sgn, nz, absD, newbf, dbf
        q = torch.quantile(torch.cat(samp, 1), torch.tensor(QS, device=dev), dim=1).t().cpu().numpy()
        agree = ((ss + both) / 2 / both.clamp(min=1)).cpu().numpy()   # sign agreement where both runs moved
        dn = torch.sqrt(torch.diagonal(gram))
        row = {"name": n, "shape": list(base[n].shape), "type": categories(n)[0], "layer": categories(n)[1],
               "base_sq": base_sq, "gram": gram.cpu().numpy().tolist(), "sign_agree": agree.tolist(),
               "frac_zero": (n_zero / numel).cpu().numpy().tolist(),
               "abs_mean": (abs_sum / numel).cpu().numpy().tolist(), "abs_max": abs_max.cpu().numpy().tolist(),
               "abs_q": q.tolist(), "q_levels": list(QS),
               "frac_bf16_changed": (bf_changed / numel).cpu().numpy().tolist(),
               "bf16_delta_sq": bf_sq.cpu().numpy().tolist(),
               "cos_bf16_vs_fp32": (bf_dot / (bf_sq.sqrt() * dn).clamp(min=1e-30)).cpu().numpy().tolist()}
        if not a.no_svd and base[n].dim() == 2 and row["type"] != "embed":
            U, S, V = [], [], []
            for r in range(R):
                u, s, vh = torch.linalg.svd(delta[r][n].to(dev), full_matrices=False)
                U.append(u[:, :a.k]); V.append(vh[:a.k].t()); S.append(s)
            S = torch.stack(S)
            s2 = S.double() ** 2
            row["sv_top"] = S[:, :32].cpu().numpy().tolist()
            row["stable_rank"] = (s2.sum(1) / s2[:, 0]).cpu().numpy().tolist()
            row["participation_ratio"] = (s2.sum(1) ** 2 / (s2 ** 2).sum(1)).cpu().numpy().tolist()
            row["energy_top"] = {str(k): (s2[:, :k].sum(1) / s2.sum(1)).cpu().numpy().tolist() for k in (1, 8, 64)}
            row["sub_left"] = [[float((U[i].t() @ U[j]).pow(2).sum() / a.k) for j in range(R)] for i in range(R)]
            row["sub_right"] = [[float((V[i].t() @ V[j]).pow(2).sum() / a.k) for j in range(R)] for i in range(R)]
            row["min_dim"] = int(min(base[n].shape))
        per.append(row)
        if ti % 20 == 0:
            print(f"tensor {ti}/{len(names)} {n} elapsed {time.time() - t0:.0f}s", flush=True)
    out = {"runs": [r[0] for r in runs], "paths": [r[1] for r in runs], "lr": a.lr, "k": a.k, "tensors": per,
           "log2_global": a.log2_global, "sketch_seed": sk.seed}
    json.dump(out, open(os.path.join(a.out, "wdelta.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
