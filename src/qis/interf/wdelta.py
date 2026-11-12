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
        b = base[n].to(dev)
        D = torch.stack([delta[r][n].to(dev).reshape(-1) for r in range(R)])        # [R, numel]
        Dd = D.double()
        gram = (Dd @ Dd.t()).cpu().numpy()
        sgn = torch.sign(D)
        nz = (sgn != 0).float()
        # sign agreement over coordinates that moved in both runs
        both = nz @ nz.t()
        agree = ((sgn @ sgn.t()) + both) / 2 / both.clamp(min=1)
        absD = D.abs()
        idx = torch.randint(0, D.shape[1], (min(D.shape[1], 2_000_000),), device=dev)
        samp = absD[:, idx]
        q = torch.quantile(samp, torch.tensor(QS, device=dev), dim=1).t().cpu().numpy()
        bb = b.reshape(-1)
        bf_base = bb.to(torch.bfloat16)
        row = {"name": n, "shape": list(base[n].shape), "type": categories(n)[0], "layer": categories(n)[1],
               "base_sq": float((bb.double() ** 2).sum()), "gram": gram.tolist(),
               "sign_agree": agree.cpu().numpy().tolist(),
               "frac_zero": (1 - nz.mean(1)).cpu().numpy().tolist(),
               "abs_mean": absD.mean(1).cpu().numpy().tolist(), "abs_max": absD.max(1).values.cpu().numpy().tolist(),
               "abs_q": q.tolist(), "q_levels": list(QS),
               "frac_bf16_changed": [float(((bb + D[r]).to(torch.bfloat16) != bf_base).float().mean())
                                     for r in range(R)]}
        # BF16 view of the update (what vLLM generation sees): cosine with the FP32 update
        cos_bf = []
        for r in range(R):
            dbf = ((bb + D[r]).to(torch.bfloat16).float() - bf_base.float()).double()
            den = dbf.norm() * Dd[r].norm()
            cos_bf.append(float((dbf @ Dd[r]) / den) if float(den) > 0 else 0.0)
        row["cos_bf16_vs_fp32"] = cos_bf
        if not a.no_svd and base[n].dim() == 2 and row["type"] != "embed":
            U, S, V = [], [], []
            for r in range(R):
                u, s, vh = torch.linalg.svd(D[r].reshape(base[n].shape), full_matrices=False)
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
        del b, D, Dd, sgn, nz, absD
        if ti % 20 == 0:
            print(f"tensor {ti}/{len(names)} {n} elapsed {time.time() - t0:.0f}s", flush=True)
    out = {"runs": [r[0] for r in runs], "paths": [r[1] for r in runs], "lr": a.lr, "k": a.k, "tensors": per,
           "log2_global": a.log2_global, "sketch_seed": sk.seed}
    json.dump(out, open(os.path.join(a.out, "wdelta.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
