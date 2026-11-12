"""A26 pass B (b-eval): shared helpers. Independent of src/qis. Stdlib + numpy only.

Reads per-attempt records (records.jsonl.gz, rows.jsonl.gz) line by line. Never reads summary files.
"""
import gzip
import hashlib
import json
import math
import os

import numpy as np

P = "/data/user_data/vineethb/qis"
OUT = P + "/audit-a26/b-eval"
BOOT_DRAWS = 10000
BOOT_SEED = 20261111  # pass-B's own seed; not the project's 2027


def ids_hash(ids):
    return hashlib.sha1(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_records(path, keep_hash=False):
    """{(pid, sample): slim record}; also returns list of duplicate keys."""
    recs, dups = {}, []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            k = (r["prompt_id"], int(r["sample"]))
            slim = {"seed": r.get("seed"), "n_tokens": r["n_tokens"], "finish": r["finish"], "status": r["status"],
                    "reward": r["reward"], "extracted": r.get("extracted")}
            if keep_hash:
                slim["h"] = ids_hash(r["ids"])
                slim["len_ids"] = len(r["ids"])
            if k in recs:
                dups.append(k)
            recs[k] = slim
    return recs, dups


def panel_ids(path, key):
    return [it["id"] for it in json.load(open(path))["pools"][key]]


def integrity(recs, dups, order, n):
    """Missing / extra / duplicate attempts and consistency checks."""
    want = {(p, j) for p in order for j in range(n)}
    have = set(recs)
    bad_cap = sum(1 for r in recs.values() if r["finish"] == "length" and r["reward"] not in (0, None))
    seeds_per_pid = {}
    for (p, j), r in recs.items():
        seeds_per_pid.setdefault(p, set()).add(r["seed"])
    return {"attempts": len(recs), "expected": len(want), "missing": len(want - have), "extra": len(have - want),
            "duplicates": len(dups), "unknown": sum(1 for r in recs.values() if r["status"] == "unknown"),
            "reward_none": sum(1 for r in recs.values() if r["reward"] is None),
            "capped_with_nonzero_reward": bad_cap,
            "problems_with_more_than_one_seed": sum(1 for s in seeds_per_pid.values() if len(s) > 1),
            "status_counts": dict(sorted(_count(r["status"] for r in recs.values()).items()))}


def _count(it):
    c = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return c


def correct(r):
    return 1 if r["reward"] == 1 else 0  # unknown (None) counts as incorrect, as in the evaluation


def per_problem(recs, order, n, field="reward", fin="finish"):
    """Arrays over problems: c (correct), t (truncated), n."""
    c = np.zeros(len(order)); t = np.zeros(len(order))
    for i, p in enumerate(order):
        for j in range(n):
            r = recs[(p, j)]
            c[i] += 1 if r[field] == 1 else 0
            t[i] += 1 if r[fin] == "length" else 0
    return c, t


def passk(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def metrics(c, t, n):
    N = len(c)
    att = N * n
    corr = float(c.sum()); trunc = float(t.sum()); fin = att - trunc
    m = {"problems": N, "attempts_per_problem": n, "attempts": att, "correct": corr, "truncated": trunc,
         "finished": fin, "pass1": float(np.mean(c / n)), "truncation_share": trunc / att,
         "acc_among_finished": corr / fin if fin else None,
         "solved_any": int((c > 0).sum()), "coverage_at_n": float(np.mean(c > 0))}
    if n >= 8:
        m["pass8_unbiased"] = float(np.mean([passk(n, int(x), 8) for x in c]))
    return m


def boot_diff(xa, xb, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Paired problem bootstrap of mean(xa - xb): same resampled problems for both. Percentile 95%."""
    d = np.asarray(xa, float) - np.asarray(xb, float)
    rng = np.random.default_rng(seed)
    N = len(d)
    idx = rng.integers(0, N, size=(draws, N))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "draws": draws, "seed": seed, "N": N,
            "share_draws_gt0": float(np.mean(bs > 0))}


def decompose(ca, ta, cb, tb, n):
    """pass1_A - pass1_B = -(tA-tB)*abar + (1-tbar)*(aA-aB), exact (midpoint form).
    t = truncated/attempts; a = correct/finished (pooled over attempts; capped attempts score 0)."""
    att = len(ca) * n
    tA, tB = ta.sum() / att, tb.sum() / att
    aA = ca.sum() / (att - ta.sum()); aB = cb.sum() / (att - tb.sum())
    abar, tbar = (aA + aB) / 2, (tA + tB) / 2
    trunc_term = -(tA - tB) * abar
    acc_term = (1 - tbar) * (aA - aB)
    total = ca.sum() / att - cb.sum() / att
    return {"tA": tA, "tB": tB, "aA": aA, "aB": aB, "truncation_term": trunc_term, "accuracy_term": acc_term,
            "sum_terms": trunc_term + acc_term, "diff": total, "residual": total - (trunc_term + acc_term)}


def coverage_split(ca, cb, n):
    """Split mean(c_A/n - c_B/n) by problem class: solved (>=1 of n) by both / A only / B only / neither."""
    N = len(ca)
    d = (ca - cb) / n
    out = {}
    for name, mask in (("both", (ca > 0) & (cb > 0)), ("A_only", (ca > 0) & (cb == 0)),
                       ("B_only", (ca == 0) & (cb > 0)), ("neither", (ca == 0) & (cb == 0))):
        out[name] = {"problems": int(mask.sum()), "contribution": float(d[mask].sum() / N),
                     "mean_diff_within": float(d[mask].mean()) if mask.any() else None}
    out["total"] = float(d.mean())
    out["coverage_part"] = out["A_only"]["contribution"] + out["B_only"]["contribution"]
    out["reliability_part"] = out["both"]["contribution"]
    return out


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
