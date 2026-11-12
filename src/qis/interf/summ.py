"""Summaries of gradient geometry from the sketches of `qis.interf.grad_geom` (study Nov-12-2026-01). numpy only.

`python -m qis.interf.summ --grad DIR --out DIR [--delta-sketch sk_delta.npy]` writes
  gram.npz      Gram matrices of the mixed-group gradients: global (`G`), Adam-scaled (`Gs`, if present),
                per module type (`Gt`) and per layer (`Gl`), exact squared norms (`sq`), and `gd` = <g, tau_run>.
  summary.json  the statistics of `analyse` for the global, scaled, per-type and per-layer Gram matrices.
  small.npz     float16 cosine matrices and float32 norms (small enough to harvest).

Definitions (g_a = gradient of mixed group a; a group belongs to one problem and one domain):
  pairwise cosine   c_ab = <g_a, g_b> / (|g_a| |g_b|), norms exact, inner products from the count sketch.
  conflict rate     share of pairs with c_ab < 0. Independent directions give 50%.
  same-problem      pairs of the two groups of one problem (independent samples of the same policy gradient).
  within-domain     pairs of groups of different problems of one domain.
  cross-domain      pairs of groups of two domains.
  signal power      P_D = mean of <g_a, g_b> over pairs a != b of domain D. Sampling noise of different groups is
                    independent, so P_D estimates the squared norm of the expected domain gradient.
  corrected cosine  rho_DE = mean cross-domain <g_a, g_b> / sqrt(P_D P_E): the cosine between the expected domain
                    gradients, with sampling noise removed from the denominator. Undefined if P_D <= 0.
Confidence intervals: bootstrap over problems (resampled with replacement, 300 draws, 95% percentile interval).
"""
import argparse
import json
import os

import numpy as np


def load_groups(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    mixed = sorted([r for r in rows if r["row"] >= 0], key=lambda r: r["row"])
    return rows, mixed


def _pair_stats(C, G, ia, ib, same):
    """Mean cosine, conflict rate, mean inner product over pairs (a in ia, b in ib). same=True: a < b only."""
    if len(ia) == 0 or len(ib) == 0:
        return None
    c, g = C[np.ix_(ia, ib)], G[np.ix_(ia, ib)]
    if same:
        iu = np.triu_indices(len(ia), 1)
        c, g = c[iu], g[iu]
    c, g = c.ravel(), g.ravel()
    if c.size == 0:
        return None
    return {"n_pairs": int(c.size), "mean_cos": float(c.mean()), "median_cos": float(np.median(c)),
            "sd_cos": float(c.std()), "conflict_rate": float((c < 0).mean()), "mean_ip": float(g.mean())}


def analyse(G, norms, mixed, n_boot=300, seed=0, boot=True):
    """G: [n, n] inner products (sketch), norms: [n] exact norms, mixed: group rows in order."""
    n = len(mixed)
    C = G / np.outer(norms, norms)
    labels = sorted({m["label"] for m in mixed})
    lab = np.array([m["label"] for m in mixed])
    pid = np.array([m["label"] + "|" + m["prompt_id"] for m in mixed])
    upid = sorted(set(pid))
    idx_of = {p: np.where(pid == p)[0] for p in upid}

    def stats(sel):
        """sel: list of row indices (with repeats under bootstrap). Returns the nested statistics."""
        sel = np.asarray(sel)
        out = {"pairs": {}, "power": {}, "rho": {}}
        byl = {L: sel[lab[sel] == L] for L in labels}
        # exclude same-problem pairs from within-domain statistics, and self pairs everywhere
        P = {}
        for L in labels:
            ia = byl[L]
            if len(ia) < 2:
                P[L] = float("nan"); continue
            g = G[np.ix_(ia, ia)]
            off = ~np.eye(len(ia), dtype=bool) & (ia[:, None] != ia[None, :])
            P[L] = float(g[off].mean()) if off.any() else float("nan")
            out["power"][L] = P[L]
        for i, L in enumerate(labels):
            for M in labels[i:]:
                ia, ib = byl[L], byl[M]
                if L == M:
                    c = C[np.ix_(ia, ia)]; g = G[np.ix_(ia, ia)]
                    keep = (pid[ia][:, None] != pid[ia][None, :])
                    keep &= np.triu(np.ones_like(keep), 1)
                else:
                    c = C[np.ix_(ia, ib)]; g = G[np.ix_(ia, ib)]
                    keep = np.ones(c.shape, dtype=bool)
                if not keep.any():
                    continue
                cc, gg = c[keep], g[keep]
                out["pairs"][f"{L}|{M}"] = {"n_pairs": int(cc.size), "mean_cos": float(cc.mean()),
                                            "median_cos": float(np.median(cc)), "sd_cos": float(cc.std()),
                                            "conflict_rate": float((cc < 0).mean()), "mean_ip": float(gg.mean())}
                if L != M and P[L] > 0 and P[M] > 0:
                    out["rho"][f"{L}|{M}"] = float(gg.mean() / np.sqrt(P[L] * P[M]))
        return out

    base = stats(np.arange(n))
    # same-problem split-half pairs
    sp = {}
    for L in labels:
        cs = [C[i[0], i[1]] for p, i in idx_of.items() if p.startswith(L + "|") and len(i) == 2]
        if cs:
            sp[L] = {"n_pairs": len(cs), "mean_cos": float(np.mean(cs)), "median_cos": float(np.median(cs)),
                     "conflict_rate": float(np.mean(np.array(cs) < 0))}
    base["same_problem"] = sp
    base["n_groups"] = {L: int((lab == L).sum()) for L in labels}
    base["n_problems"] = {L: len([p for p in upid if p.startswith(L + "|")]) for L in labels}
    base["mean_norm"] = {L: float(norms[lab == L].mean()) for L in labels}
    base["median_norm"] = {L: float(np.median(norms[lab == L])) for L in labels}
    # cosine between the summed gradients of each domain (as a training batch would sum them; noise included)
    S = {L: np.where(lab == L)[0] for L in labels}
    base["sum_cos"] = {}
    for i, L in enumerate(labels):
        for M in labels[i + 1:]:
            num = G[np.ix_(S[L], S[M])].sum()
            den = np.sqrt(G[np.ix_(S[L], S[L])].sum() * G[np.ix_(S[M], S[M])].sum())
            base["sum_cos"][f"{L}|{M}"] = float(num / den)
    if boot:
        rng = np.random.default_rng(seed)
        acc = {}
        for _ in range(n_boot):
            draw = rng.choice(len(upid), len(upid), replace=True)
            sel = np.concatenate([idx_of[upid[j]] for j in draw])
            st = stats(sel)
            for k, v in st["rho"].items():
                acc.setdefault(("rho", k), []).append(v)
            for k, v in st["pairs"].items():
                acc.setdefault(("mean_cos", k), []).append(v["mean_cos"])
                acc.setdefault(("conflict_rate", k), []).append(v["conflict_rate"])
        base["ci95"] = {f"{a}:{k}": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)]
                        for (a, k), v in acc.items()}
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta-sketch", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.grad, "meta.json")))
    rows, mixed = load_groups(os.path.join(a.grad, "groups.jsonl"))
    n, nt, nl = len(mixed), len(meta["types"]), len(meta["layers"])
    sq = np.load(os.path.join(a.grad, "sq.npy"))[:n]
    norms = np.sqrt(sq[:, :nt].sum(1))

    def gram(path, sub=None):
        A = np.load(path, mmap_mode="r")
        X = np.asarray(A[:n] if sub is None else A[:n, sub], dtype=np.float64)
        return X @ X.T
    out = {"G": gram(os.path.join(a.grad, "sk_global.npy")), "sq": sq}
    res = {"meta": {k: meta[k] for k in ("ckpt", "optim", "types", "layers", "n_mixed", "n_groups", "log2_global",
                                         "log2_cat")},
           "global": analyse(out["G"], norms, mixed)}
    # sketch check: sketch norm against exact norm
    res["sketch_norm_ratio"] = [float(np.percentile(np.diag(out["G"]) / norms ** 2, q)) for q in (1, 50, 99)]
    sp = os.path.join(a.grad, "sk_scaled.npy")
    if os.path.exists(sp):
        out["Gs"] = gram(sp)
        res["scaled"] = analyse(out["Gs"], np.sqrt(sq[:, -1]), mixed)
    Gt = np.stack([gram(os.path.join(a.grad, "sk_type.npy"), i) for i in range(nt)])
    Gl = np.stack([gram(os.path.join(a.grad, "sk_layer.npy"), i) for i in range(nl)])
    out["Gt"], out["Gl"] = Gt, Gl
    res["by_type"] = {t: analyse(Gt[i], np.sqrt(sq[:, i]), mixed, boot=False) for i, t in enumerate(meta["types"])}
    res["by_layer"] = {l: analyse(Gl[i], np.sqrt(sq[:, nt + i]), mixed, boot=False)
                       for i, l in enumerate(meta["layers"])}
    res["norm_share_type"] = {t: float((sq[:, i] / sq[:, :nt].sum(1)).mean()) for i, t in enumerate(meta["types"])}
    res["norm_share_layer"] = {l: float((sq[:, nt + i] / sq[:, :nt].sum(1)).mean())
                               for i, l in enumerate(meta["layers"])}
    small = {"cos": (out["G"] / np.outer(norms, norms)).astype(np.float16), "norms": norms.astype(np.float32)}
    if "Gs" in out:
        ns = np.sqrt(sq[:, -1])
        small["cos_scaled"] = (out["Gs"] / np.outer(ns, ns)).astype(np.float16); small["norms_scaled"] = ns.astype(np.float32)
    if a.delta_sketch:
        D = np.load(a.delta_sketch).astype(np.float64)
        A = np.asarray(np.load(os.path.join(a.grad, "sk_global.npy"), mmap_mode="r")[:n], dtype=np.float64)
        out["gd"] = A @ D.T
        dn = np.sqrt((D ** 2).sum(1))
        small["cos_g_delta"] = (out["gd"] / np.outer(norms, dn)).astype(np.float32)
        small["delta_gram"] = (D @ D.T).astype(np.float64)
    np.savez_compressed(os.path.join(a.out, "gram.npz"), **out)
    np.savez_compressed(os.path.join(a.out, "small.npz"), **small)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"))
    slim = [{k: r[k] for k in ("label", "prompt_id", "half", "n_correct", "row")} | {"tok": int(sum(r["n_tokens"]))}
            for r in rows]
    json.dump(slim, open(os.path.join(a.out, "groups_slim.json"), "w"))
    g = res["global"]
    print(json.dumps({"n_groups": g["n_groups"], "rho": g["rho"], "same_problem": g["same_problem"],
                      "pairs": {k: [round(v["mean_cos"], 4), round(v["conflict_rate"], 3)] for k, v in g["pairs"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
