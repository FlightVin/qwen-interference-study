"""Coupling between two rewards on the same base-policy samples (study Nov-12-2026-01, domain study).

Input: two `grad_geom` output folders computed at the same checkpoint with the same sketch seed.
  --a  gradients of the correctness reward (labels hard, related, ...)
  --b  gradients of an alternative reward on the same samples (labels hard~short, ...)
For every pair (label L of a, label M of b) and for the pairs inside b, the tool reports the cosine between group
gradients in three classes:
  diff_problem   the two groups belong to different problems (the cross-problem coupling)
  same_problem   same problem, different groups of 8 answers (independent samples; the per-problem coupling)
  same_group     same problem and the same 8 answers (shares sampling noise; reported for reference only)
and the corrected cosine rho = mean inner product over diff_problem pairs / sqrt(P_L P_M), with P the mean inner
product over different-problem pairs inside one label. Intervals: bootstrap over problems, 95% percentile.

Output: --out/summary.json. Usage: python -m qis.interf.summ_reward --a DIR --b DIR --out DIR
"""
import argparse
import json
import os

import numpy as np


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "groups.jsonl")) if l.strip()]
    mixed = sorted([r for r in rows if r["row"] >= 0], key=lambda r: r["row"])
    n = len(mixed)
    X = np.asarray(np.load(os.path.join(d, "sk_global.npy"), mmap_mode="r")[:n], dtype=np.float64)
    return X, mixed, len(rows)


def block(Xa, ma, Xb, mb, same_set, n_boot, rng):
    """Statistics for groups ma (rows of Xa) against mb (rows of Xb)."""
    na, nb = np.linalg.norm(Xa, axis=1), np.linalg.norm(Xb, axis=1)
    G = Xa @ Xb.T
    C = G / np.outer(na, nb)
    pa = np.array([m["prompt_id"] for m in ma]); pb = np.array([m["prompt_id"] for m in mb])
    ha = np.array([m["half"] for m in ma]); hb = np.array([m["half"] for m in mb])
    sp = pa[:, None] == pb[None, :]
    sg = sp & (ha[:, None] == hb[None, :])
    valid = np.ones_like(sp)
    if same_set:
        valid = ~np.eye(len(ma), dtype=bool)
    out = {"n_a": len(ma), "n_b": len(mb)}
    for name, sel in (("diff_problem", ~sp & valid), ("same_problem", sp & ~sg & valid), ("same_group", sg & valid)):
        if sel.sum() == 0:
            continue
        c = C[sel]
        out[name] = {"n_pairs": int(sel.sum()) // (2 if same_set else 1), "mean_cos": float(c.mean()),
                     "conflict_rate": float((c < 0).mean()), "mean_ip": float(G[sel].mean())}
    # bootstrap over problems for the mean cosine of diff_problem and same_problem pairs
    ua, ia = np.unique(pa, return_inverse=True); ub, ib = np.unique(pb, return_inverse=True)
    if same_set:
        ub, ib = ua, ia
    Ia = np.zeros((len(ua), len(ma))); Ia[ia, np.arange(len(ma))] = 1
    Ib = np.zeros((len(ub), len(mb))); Ib[ib, np.arange(len(mb))] = 1
    un = np.unique(np.concatenate([ua, ub])); pos = {p: k for k, p in enumerate(un)}
    pos_a = np.array([pos[p] for p in ua]); pos_b = np.array([pos[p] for p in ub])
    res = {}
    for name, sel in (("diff_problem", ~sp & valid), ("same_problem", sp & ~sg & valid)):
        if sel.sum() == 0:
            continue
        S = Ia @ (C * sel) @ Ib.T; N = Ia @ sel.astype(float) @ Ib.T
        bs = []
        for _ in range(n_boot):
            if same_set:
                w = np.bincount(rng.integers(0, len(ua), len(ua)), minlength=len(ua)).astype(float)
                num, den = w @ S @ w, w @ N @ w
            else:                                   # resample the union of problems once, so shared problems move together
                w = np.bincount(rng.integers(0, len(un), len(un)), minlength=len(un)).astype(float)
                wa, wb = w[pos_a], w[pos_b]
                num, den = wa @ S @ wb, wa @ N @ wb
            if den > 0:
                bs.append(num / den)
        res[name] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["ci95_mean_cos"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    Xa, ma, ta = load(a.a); Xb, mb, tb = load(a.b)
    la = sorted({m["label"] for m in ma}); lb = sorted({m["label"] for m in mb})
    ra = {L: np.array([i for i, m in enumerate(ma) if m["label"] == L]) for L in la}
    rb = {L: np.array([i for i, m in enumerate(mb) if m["label"] == L]) for L in lb}
    res = {"n_groups_a": ta, "n_mixed_a": len(ma), "n_groups_b": tb, "n_mixed_b": len(mb), "within_a": {}, "within_b": {},
           "cross": {}, "rho": {}}
    for L in la:
        res["within_a"][L] = block(Xa[ra[L]], [ma[i] for i in ra[L]], Xa[ra[L]], [ma[i] for i in ra[L]], True, a.boot, rng)
    for i, L in enumerate(lb):
        res["within_b"][L] = block(Xb[rb[L]], [mb[j] for j in rb[L]], Xb[rb[L]], [mb[j] for j in rb[L]], True, a.boot, rng)
        for M in lb[i + 1:]:
            res["within_b"][f"{L}|{M}"] = block(Xb[rb[L]], [mb[j] for j in rb[L]], Xb[rb[M]], [mb[j] for j in rb[M]],
                                                False, a.boot, rng)
    for L in la:
        for M in lb:
            x = block(Xa[ra[L]], [ma[i] for i in ra[L]], Xb[rb[M]], [mb[j] for j in rb[M]], False, a.boot, rng)
            res["cross"][f"{L}|{M}"] = x
            pa = res["within_a"][L].get("diff_problem", {}).get("mean_ip"); pb = res["within_b"][M].get("diff_problem", {}).get("mean_ip")
            if pa and pb and pa > 0 and pb > 0 and "diff_problem" in x:
                res["rho"][f"{L}|{M}"] = x["diff_problem"]["mean_ip"] / (pa * pb) ** 0.5
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for k, v in res["cross"].items():
        print(k, {c: (round(v[c]["mean_cos"], 5), round(v[c]["conflict_rate"], 3), v[c]["n_pairs"]) for c in
                  ("diff_problem", "same_problem", "same_group") if c in v}, "rho", res["rho"].get(k))
    for k, v in res["within_b"].items():
        print("b", k, {c: (round(v[c]["mean_cos"], 5), v[c]["n_pairs"]) for c in ("diff_problem", "same_problem") if c in v})


if __name__ == "__main__":
    main()
