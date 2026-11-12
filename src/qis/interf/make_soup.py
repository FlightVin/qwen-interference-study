"""Build linear combinations of stored weight updates as checkpoints (study Nov-12-2026-01). No training.

Each run update is tau_r = theta_r - theta_base. The tool writes theta_base + sum_r c_r tau_r for five combinations:
  mean6        mean of the six round-128 updates (run-specific parts average down, the common part stays)
  mean6x       mean6 scaled to the norm of the R-s101 update
  loo5         mean of the five updates other than R-s101
  resid-R101   tau_R101 - loo5 (R-s101 with the common part removed)
  scaled-R101  tau_R101 scaled to the norm of mean6
Output: OUT/<name>/model.pt (same keys as a training checkpoint), OUT/<name>/state.json, OUT/soups.json (norms, cosines).
Usage: python -m qis.interf.make_soup --model BASE --out DIR --ckpt name=DIR (six times; one must be R-s101-r128)
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    bm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    pnames = [n for n, _ in bm.named_parameters()]
    base = {k: v.detach().clone() for k, v in bm.state_dict().items()}
    del bm
    runs = [c.split("=", 1) for c in a.ckpt]
    names = [r[0] for r in runs]
    assert "R-s101-r128" in names and len(names) == 6
    tau = {}
    for name, path in runs:
        sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
        assert set(sd) == set(base), "checkpoint keys differ from the base state dict"
        tau[name] = {k: sd[k].float() - base[k] for k in base}
        del sd
    others = [n for n in names if n != "R-s101-r128"]

    def comb(coef):
        return {k: sum(c * tau[n][k] for n, c in coef.items()) for k in base}

    def dot(x, y):
        return float(sum((x[k].double() * y[k].double()).sum() for k in pnames))
    mean6 = comb({n: 1 / 6 for n in names})
    loo5 = comb({n: 1 / 5 for n in others})
    tR = tau["R-s101-r128"]
    nR, n6 = dot(tR, tR) ** 0.5, dot(mean6, mean6) ** 0.5
    soups = {"mean6": mean6, "mean6x": {k: v * (nR / n6) for k, v in mean6.items()}, "loo5": loo5,
             "resid-R101": {k: tR[k] - loo5[k] for k in base}, "scaled-R101": {k: v * (n6 / nR) for k, v in tR.items()}}
    info = {"norm_R101": nR, "norm_mean6": n6, "scale_mean6x": nR / n6, "scale_R101": n6 / nR, "soups": {}}
    for name, d in soups.items():
        od = os.path.join(a.out, name)
        os.makedirs(od, exist_ok=True)
        torch.save({k: (base[k] + d[k]).contiguous() for k in base}, os.path.join(od, "model.pt"))
        json.dump({"soup": name, "sources": dict(runs)}, open(os.path.join(od, "state.json"), "w"))
        info["soups"][name] = {"norm": dot(d, d) ** 0.5, "cos_with_R101": dot(d, tR) / (dot(d, d) ** 0.5 * nR),
                               "cos_with_mean6": dot(d, mean6) / (dot(d, d) ** 0.5 * n6),
                               "cos_with_loo5": dot(d, loo5) / (dot(d, d) * dot(loo5, loo5)) ** 0.5}
        print(name, info["soups"][name], flush=True)
    json.dump(info, open(os.path.join(a.out, "soups.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
