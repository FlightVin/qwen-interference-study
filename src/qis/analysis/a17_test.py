"""EXPLORATORY (A17): sealed-test seed reversal, coverage, sampling null and endpoint reproducibility.

Inputs: `qis.evaluate_ckpt` outputs, `<dir>/records.jsonl.gz` (one JSON line per attempt: prompt_id, seed, sample 0-31,
n_tokens, finish, extracted, status, reward, ids) and `<dir>/summary.json` (per_problem c/n, used only as a cross-check).
- EVAL_ROOT/qis-test-base and EVAL_ROOT/qis-test-{H,R,U}-s{101,202}-sym. H seed 202 is read from `--h202 NAME`
  (default qis-test-H-s202-sym), looked up in EVAL_ROOT, then in CGPU_ROOT; absent cells are written "missing".
- CGPU_ROOT/qis-cgpu-{R,U}-s{101,202} (optional): L40S reruns with the original request seeds. H-s101 and base already
  ran on L40S, and the H-s202 endpoint is common-GPU, so those cells are shared by both tables.
Outputs in OUT_DIR (JSON + Markdown; every delta is written as "A − B = d" with both absolute values):
1. reversal: per problem R − U (c/32) per seed with binomial SE; sign agreement across seeds; per band (per_problem.py
   bands from base samples 0-15) absolute R and U, band R − U, contribution (size/128 × band R − U) and reversal
   contribution (seed 202 − seed 101); B1 items sorted by |change|, top-3/top-5 share, Spearman within B1 and overall.
2. coverage: per policy pass@1, pass@8, problems solved ≥ 1 of 32, expected distinct solved at k = 1 and 8, 0/32 count,
   truncation share (finish == "length"); decomposition against the held-out base baseline (base samples 16-31);
   paired R vs U per seed split into coverage (exactly one reaches) and reliability (both reach) parts.
3. sampling_null: expected (hypergeometric) and empirical (fixed disjoint splits) "lost"/"gained" counts between two
   groups of m = 8 or 16 attempts at one checkpoint; the curve P_loss(p, m) = (1 − (1 − p)^m)(1 − p)^m.
4. endpoints: original table (primary) and common-GPU table (reproducibility check): pass@1, pass@8, paired R − U, R − H,
   U − H with 95% paired problem bootstrap (10,000 draws, seed 2027; original-table streams are the A9 streams of
   report_tables, so they reproduce its intervals); per checkpoint original-vs-cgpu agreement.
6. truncation: per policy truncation share t (finish == "length"), accuracy among finished attempts a, pass@1 = (1 − t)·a,
   mean n_tokens by outcome; R − U per seed and band split exactly into −Δt·ā + (1 − t̄)·Δa with problem-bootstrap
   intervals (streams a9/a17-trunc/...); per-problem Δt vs Δsuccess correlations; success within 8,192 tokens.
`--figures FIG_DIR` (or `plot(OUT_DIR, FIG_DIR)` from the matplotlib venv) draws endpoints.png, reversal_scatter.png,
coverage.png, sampling_null.png and truncation.png from the JSON outputs.
Usage: python -m qis.analysis.a17_test EVAL_ROOT OUT_DIR [--cgpu-root CGPU_ROOT] [--h202 NAME] [--figures FIG_DIR]
"""
import argparse
import gzip
import hashlib
import json
import os
from math import comb

import numpy as np

from qis import seeds
from qis.analysis.per_problem import BANDS, N, band_of, pass8
from qis.analysis.report_tables import DRAWS
from qis.analysis.report_tables import boot as boot_a9       # seeds.rng(2027, "a9/<name>"), 10,000 problem draws

ARMS, SEEDS = ("H", "R", "U"), (101, 202)
CONTRASTS = (("R-U", "R", "U"), ("R-H", "R", "H"), ("U-H", "U", "H"))
SPLITS = {8: (((0, 8), (8, 16)), ((16, 24), (24, 32))), 16: (((0, 16), (16, 32)),)}
C_BINS = ((0, 0), (1, 2), (3, 4), (5, 8), (9, 16), (17, 24), (25, 31), (32, 32))
DEV_PANEL = {"problems": 53, "n": 8, "source": "reports/tables/report_data.json: all 28 panel=dev eval rows have n=8, problems=53"}


# ---------------------------------------------------------------- loading
def load(d):
    """Raw records of one evaluation directory, or None when records.jsonl.gz is absent."""
    p = os.path.join(d, "records.jsonl.gz")
    if not os.path.exists(p):
        return None
    rec, lines = {}, 0
    for l in gzip.open(p, "rt"):
        r = json.loads(l); lines += 1
        ids = r.get("ids")
        rec.setdefault(r["prompt_id"], {})[r["sample"]] = (
            r["reward"] == 1, r.get("finish"), r.get("seed"),
            hashlib.sha1(json.dumps(ids).encode()).hexdigest() if ids is not None else None, r.get("n_tokens"))
    full = {k: v for k, v in rec.items() if all(i in v for i in range(N))}
    s = os.path.join(d, "summary.json")
    sc = {x["id"]: x["c"] for x in json.load(open(s))["per_problem"]} if os.path.exists(s) else None
    return {"rec": full, "incomplete": sorted(set(rec) - set(full)), "lines": lines, "summary_c": sc, "dir": d}


def cell(raw, probs):
    """Arrays over (problem in probs, sample 0-31), or a string saying why the cell is unusable."""
    if raw is None:
        return "missing"
    if sorted(raw["rec"]) != probs:
        return f"incomplete: {len(raw['rec'])} problems with 32 samples, {len(set(raw['rec']) & set(probs))} of {len(probs)} base problems"
    R = raw["rec"]
    get = lambda j: [[R[p][i][j] for i in range(N)] for p in probs]          # noqa: E731
    y, fin = np.array(get(0), dtype=bool), get(1)
    trunc = np.array([[f == "length" for f in r] for r in fin]) if all(f is not None for r in fin for f in r) else None
    nt = get(4)
    ntok = np.array(nt, dtype=np.int64) if all(v is not None for r in nt for v in r) else None
    c = y.sum(1)
    mism = None if raw["summary_c"] is None else sum(raw["summary_c"].get(p) != int(c[i]) for i, p in enumerate(probs))
    return {"y": y, "c": c, "trunc": trunc, "ntok": ntok, "seed": get(2), "hash": get(3), "dir": raw["dir"], "lines": raw["lines"],
            "summary_c_mismatch": mism}


def status(x):
    if isinstance(x, str):
        return x
    return {"dir": x["dir"], "record_lines": x["lines"], "summary_c_mismatch_problems": x["summary_c_mismatch"],
            "finish_field": x["trunc"] is not None}


# ---------------------------------------------------------------- helpers
def ok(x):
    return isinstance(x, dict)


def num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def f3(x, f=".3f"):
    return format(x, f) if num(x) else ("missing" if x is None else str(x))


def dl(a, b, d=None, ci=None, f=".3f"):
    """'a − b = d [lo, hi]': the delta next to both absolute values."""
    if not (num(a) and num(b)):
        return "missing"
    d = a - b if d is None else d
    return f"{a:{f}} − {b:{f}} = {d:+{f}}" + (f" [{ci['lo']:+.3f}, {ci['hi']:+.3f}]" if ci else "")


def frac(k, n):
    return f"{int(k)}/{int(n)}"


def mean_or_none(v):
    return float(np.mean(v)) if len(v) else None


def avg_rank(x):
    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind="mergesort"); xs = x[o]; r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(a, b):
    ra, rb = avg_rank(a), avg_rank(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 and ra.std() > 0 and rb.std() > 0 else None


def p_loss(p, m):
    """Independent sampling: P(group A of m has ≥ 1 success and group B of m has none)."""
    return (1 - (1 - p) ** m) * (1 - p) ** m


def p_lost_hyper(c, m, n=N):
    """Two disjoint random groups of m from n attempts with c successes: P(A ≥ 1 success, B = 0) = P(B=0) − P(A∪B=0)."""
    return comb(n - c, m) / comb(n, m) - comb(n - c, 2 * m) / comb(n, 2 * m)


def dump(out, name, res, md):
    json.dump(res, open(os.path.join(out, name + ".json"), "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    open(os.path.join(out, name + ".md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[:40]))


# ---------------------------------------------------------------- 1. reversal
def reversal(pol, probs, kb):
    n = len(probs)
    need = [f"{a}-s{s}" for a in "RU" for s in SEEDS]
    miss = [k for k in need if not ok(pol[k])]
    if miss:
        return {"status": "missing", "missing": miss}, ["# Reversal", "", f"missing: {', '.join(miss)}"]
    c = {k: pol[k]["c"] for k in need}
    p = {k: v / N for k, v in c.items()}
    dc = {s: c[f"R-s{s}"] - c[f"U-s{s}"] for s in SEEDS}                   # integer c differences (exact ties)
    d = {s: dc[s] / N for s in SEEDS}
    se = {s: np.sqrt(p[f"R-s{s}"] * (1 - p[f"R-s{s}"]) / N + p[f"U-s{s}"] * (1 - p[f"U-s{s}"]) / N) for s in SEEDS}
    a, b = dc[101], dc[202]
    band = np.array([band_of(int(k)) for k in kb])
    allc = np.stack([c[k] for k in need])
    agree = {"n": n, "same_sign": int(np.sum(a * b > 0)), "opposite_sign": int(np.sum(a * b < 0)),
             "tie_in_either_seed": int(np.sum((a == 0) | (b == 0))), "tie_in_both_seeds": int(np.sum((a == 0) & (b == 0))),
             "all_four_0of32": int(np.sum((allc == 0).all(0))), "all_four_32of32": int(np.sum((allc == N).all(0))),
             "rule": "sign of R − U in c/32; same/opposite need both seeds nonzero; tie = difference exactly 0"}
    overall = {f"s{s}": {"R": float(p[f"R-s{s}"].mean()), "U": float(p[f"U-s{s}"].mean()), "R-U": float(d[s].mean())} for s in SEEDS}
    overall["reversal_s202_minus_s101"] = overall["s202"]["R-U"] - overall["s101"]["R-U"]
    bands = {}
    for bn, lo, hi in BANDS:
        m = band == bn; k = int(m.sum())
        row = {"rule": f"base samples 0-15 correct in [{lo},{hi}] of 16", "size": k, "share": k / n}
        for s in SEEDS:
            if k:
                R_, U_ = float(p[f"R-s{s}"][m].mean()), float(p[f"U-s{s}"][m].mean())
                row[f"s{s}"] = {"R": R_, "U": U_, "R-U": R_ - U_, "contribution": k / n * (R_ - U_)}
            else:
                row[f"s{s}"] = None
        row["reversal_contribution_s202_minus_s101"] = (row["s202"]["contribution"] - row["s101"]["contribution"]) if k else None
        bands[bn] = row
    check = {f"s{s}": sum(v[f"s{s}"]["contribution"] for v in bands.values() if v[f"s{s}"]) for s in SEEDS}
    per = [{"id": probs[i], "band": band[i], "base_k16": int(kb[i]), "c": {k: int(c[k][i]) for k in need},
            "d101": float(d[101][i]), "se101": float(se[101][i]), "d202": float(d[202][i]), "se202": float(se[202][i])}
           for i in range(n)]
    m1 = np.where(band == "B1")[0]
    items = sorted((dict(per[i], change_s202_minus_s101=per[i]["d202"] - per[i]["d101"]) for i in m1),
                   key=lambda x: (-abs(x["change_s202_minus_s101"]), x["id"]))
    tot = sum(x["change_s202_minus_s101"] for x in items)
    share = lambda k: sum(x["change_s202_minus_s101"] for x in items[:k]) / tot if tot else None   # noqa: E731
    b1 = {"size": len(items), "reversal_sum_of_item_changes": tot,
          "reversal_contribution_to_overall": tot / n,
          "share_top3": share(3), "share_top5": share(5),
          "share_note": "signed share of the band's summed (seed-202 − seed-101) item change; items sorted by |change|",
          "spearman_within_B1": spearman(d[101][m1], d[202][m1]) if len(m1) else None, "items": items}
    res = {"n_problems": n, "overall": overall, "agreement": agree, "bands": bands,
           "band_contribution_sum_check": check, "B1": b1,
           "spearman_overall": spearman(d[101], d[202]), "per_problem": per,
           "se_rule": "sqrt(pR(1-pR)/32 + pU(1-pU)/32) per problem and seed"}
    o = overall
    md = ["# Seed reversal of R − U (sealed test, 128 problems × 32 attempts)", "",
          "| Seed | R pass@1 | U pass@1 | R − U (R minus U) |", "|---|---|---|---|"]
    md += [f"| {s} | {o[f's{s}']['R']:.3f} | {o[f's{s}']['U']:.3f} | {dl(o[f's{s}']['R'], o[f's{s}']['U'])} |" for s in SEEDS]
    md += ["", f"Reversal (seed-202 R − U minus seed-101 R − U): {dl(o['s202']['R-U'], o['s101']['R-U'])}.", "",
           f"Per-problem sign of R − U across seeds (of {n}): same sign {agree['same_sign']}, opposite sign "
           f"{agree['opposite_sign']}, tie in either seed {agree['tie_in_either_seed']}, tie in both seeds "
           f"{agree['tie_in_both_seeds']} (all four 0/32: {agree['all_four_0of32']}; all four 32/32: {agree['all_four_32of32']}).",
           f"Spearman rank correlation of per-problem R − U, seed 101 vs seed 202: overall {f3(res['spearman_overall'])}, "
           f"within B1 {f3(b1['spearman_within_B1'])}.", "",
           "| Band | Problems | R − U s101 | Contribution s101 | R − U s202 | Contribution s202 | Reversal contribution (s202 − s101) |",
           "|---|---|---|---|---|---|---|"]
    for bn, v in bands.items():
        if not v["size"]:
            md.append(f"| {bn} | 0/{n} | – | – | – | – | – |"); continue
        x1, x2 = v["s101"], v["s202"]
        md.append(f"| {bn} | {v['size']}/{n} ({v['share']:.3f}) | {dl(x1['R'], x1['U'])} | {v['size']}/{n} × {x1['R-U']:+.3f} = "
                  f"{x1['contribution']:+.4f} | {dl(x2['R'], x2['U'])} | {v['size']}/{n} × {x2['R-U']:+.3f} = {x2['contribution']:+.4f} | "
                  f"{dl(x2['contribution'], x1['contribution'], f='.4f')} |")
    md += [f"| Sum | {n}/{n} | | {check['s101']:+.4f} | | {check['s202']:+.4f} | {check['s202'] - check['s101']:+.4f} |", "",
           f"B1: {b1['size']} problems; summed item change (seed-202 R − U minus seed-101 R − U) {tot:+.3f} "
           f"(= {tot / n:+.4f} of the overall reversal). Top 3 items carry {f3(b1['share_top3'])} of it, top 5 carry "
           f"{f3(b1['share_top5'])}. Top 10 by |change| (full list in reversal.json):", "",
           "| Problem | Base k/16 | R − U s101 (c/32) | R − U s202 (c/32) | Change (s202 − s101) |", "|---|---|---|---|---|"]
    for x in items[:10]:
        cc = x["c"]
        md.append(f"| {x['id']} | {x['base_k16']}/16 | {cc['R-s101']}/32 − {cc['U-s101']}/32 = {x['d101']:+.3f} | "
                  f"{cc['R-s202']}/32 − {cc['U-s202']}/32 = {x['d202']:+.3f} | {dl(x['d202'], x['d101'])} |")
    return res, md


# ---------------------------------------------------------------- 2. coverage
def decompose(pp, hold, n):
    """pp: policy pass@1 per problem; hold: base successes in samples 16-31 (independent baseline)."""
    U = hold == 0; bp = hold / 16; ch = pp - bp; became = U & (pp > 0)
    return {"baseline": "base samples 16-31 (16 attempts)",
            "unreachable_at_baseline": frac(U.sum(), n), "became_reachable": frac(became.sum(), U.sum()),
            "became_reachable_mean_pass1": mean_or_none(pp[became]), "unreachable_mean_pass1": mean_or_none(pp[U]),
            "reachable_at_baseline": frac((~U).sum(), n), "reachable_now_zero": frac(((~U) & (pp == 0)).sum(), (~U).sum()),
            "reachable_mean_baseline_pass1": mean_or_none(bp[~U]), "reachable_mean_pass1": mean_or_none(pp[~U]),
            "reachable_mean_change": mean_or_none(ch[~U]),
            "overall_change_vs_baseline": float(ch.mean()),
            "contribution_unreachable": float(ch[U].sum() / n), "contribution_reachable": float(ch[~U].sum() / n)}


def coverage(pol, probs):
    n, base = len(probs), pol["base"]
    hold = base["y"][:, 16:].sum(1)
    res = {"n_problems": n, "policies": {}, "paired_R_vs_U": {}}
    for k, x in pol.items():
        if not ok(x):
            res["policies"][k] = x; continue
        c = x["c"]; pp = c / N; p8 = np.array([pass8(int(v)) for v in c])
        t = x["trunc"]
        row = {"pass1": float(pp.mean()), "pass8": float(p8.mean()), "solved_any": frac((c > 0).sum(), n),
               "solved_any_n": int((c > 0).sum()), "zero": frac((c == 0).sum(), n),
               "expected_distinct_k1": float(pp.sum()), "expected_distinct_k8": float(p8.sum()),
               "truncated_share": float(t.mean()) if t is not None else "unavailable (records lack finish)",
               "truncated_attempts": frac(t.sum(), t.size) if t is not None else "unavailable"}
        if k == "base":
            row["decomposition"] = dict(decompose(base["y"][:, :16].mean(1), hold, n),
                                        note="null reference: base samples 0-15 (16 attempts) against samples 16-31; "
                                             "not 1:1 comparable with 32-attempt policies")
        else:
            row["decomposition"] = decompose(pp, hold, n)
        res["policies"][k] = row
    for s in SEEDS:
        r, u = pol[f"R-s{s}"], pol[f"U-s{s}"]
        if not (ok(r) and ok(u)):
            res["paired_R_vs_U"][f"s{s}"] = "missing"; continue
        cR, cU = r["c"], u["c"]; d = (cR - cU) / N
        oR, oU, both = (cR > 0) & (cU == 0), (cU > 0) & (cR == 0), (cR > 0) & (cU > 0)
        res["paired_R_vs_U"][f"s{s}"] = {
            "R": float((cR / N).mean()), "U": float((cU / N).mean()), "R-U": float(d.mean()),
            "coverage": {"R_only": frac(oR.sum(), n), "U_only": frac(oU.sum(), n),
                         "contribution": float(d[oR | oU].sum() / n), "contribution_R_only": float(d[oR].sum() / n),
                         "contribution_U_only": float(d[oU].sum() / n)},
            "reliability": {"both_reach": frac(both.sum(), n), "contribution": float(d[both].sum() / n),
                            "mean_R_minus_U_within": mean_or_none(d[both])},
            "neither": frac((~(oR | oU | both)).sum(), n),
            "rule": "coverage = exactly one of R/U has 0/32; reliability = both > 0/32; contribution = sum of (R − U) / 128"}
    P = res["policies"]
    md = ["# Coverage (sealed test, 32 attempts per problem)", "",
          "| Policy | pass@1 | pass@8 | Solved ≥ 1 of 32 | 0/32 | E[distinct solved], k=1 | E[distinct solved], k=8 | Truncated attempts |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in P.items():
        if not ok(v):
            md.append(f"| {k} | {v} | | | | | | |"); continue
        md.append(f"| {k} | {v['pass1']:.3f} | {v['pass8']:.3f} | {v['solved_any']} | {v['zero']} | {v['expected_distinct_k1']:.1f} | "
                  f"{v['expected_distinct_k8']:.1f} | {v['truncated_attempts']} ({f3(v['truncated_share'])}) |")
    md += ["", "Decomposition against the held-out baseline (base samples 16-31). Base row: samples 0-15 vs 16-31 (null reference).", "",
           "| Policy | Unreachable at baseline | Became reachable (c > 0) | Their mean pass@1 | Reachable at baseline | "
           "Reachable: policy − baseline pass@1 | Reachable now 0 | Overall change = unreachable part + reachable part |",
           "|---|---|---|---|---|---|---|---|"]
    for k, v in P.items():
        if ok(v):
            x = v["decomposition"]
            md.append(f"| {k} | {x['unreachable_at_baseline']} | {x['became_reachable']} | {f3(x['became_reachable_mean_pass1'])} | "
                      f"{x['reachable_at_baseline']} | {dl(x['reachable_mean_pass1'], x['reachable_mean_baseline_pass1'])} | "
                      f"{x['reachable_now_zero']} | {x['overall_change_vs_baseline']:+.3f} = {x['contribution_unreachable']:+.3f} + "
                      f"{x['contribution_reachable']:+.3f} |")
    md += ["", "Paired R vs U: R − U split into coverage (exactly one of R/U has 0/32) and reliability (both reach).", "",
           "| Seed | R − U (R minus U) | R only / U only | Coverage part (R only + U only) | Both reach | Reliability part | Neither |",
           "|---|---|---|---|---|---|---|"]
    for s, v in res["paired_R_vs_U"].items():
        if not ok(v):
            md.append(f"| {s} | {v} | | | | | |"); continue
        cv, rl = v["coverage"], v["reliability"]
        md.append(f"| {s} | {dl(v['R'], v['U'])} | {cv['R_only']} / {cv['U_only']} | {cv['contribution']:+.3f} = "
                  f"{cv['contribution_R_only']:+.3f} + {cv['contribution_U_only']:+.3f} | {rl['both_reach']} | "
                  f"{rl['contribution']:+.3f} | {v['neither']} |")
    return res, md


# ---------------------------------------------------------------- 3. sampling null
def sampling_null(pol, probs):
    n = len(probs)
    res = {"n_problems": n, "policies": {}, "dev_panel": DEV_PANEL}
    pool = {m: {b: [0, 0, 0, 0.0, 0.0, 0] for b in C_BINS} for m in SPLITS}  # problems, lost, gained, sum p, sum hyper, splits
    for k, x in pol.items():
        if not ok(x):
            res["policies"][k] = x; continue
        y, c, row = x["y"], x["c"], {}
        for m, splits in SPLITS.items():
            hyp = np.array([p_lost_hyper(int(v), m) for v in c])
            emp = []
            for (a0, a1), (b0, b1) in splits:
                A, B = y[:, a0:a1].any(1), y[:, b0:b1].any(1)
                lost, gained = A & ~B, ~A & B
                emp.append({"A": f"samples {a0}-{a1 - 1}", "B": f"samples {b0}-{b1 - 1}",
                            "lost_A_some_B_zero": int(lost.sum()), "gained_A_zero_B_some": int(gained.sum())})
                for i, v in enumerate(c):
                    q = pool[m][next(bb for bb in C_BINS if bb[0] <= v <= bb[1])]
                    q[0] += 1; q[1] += int(lost[i]); q[2] += int(gained[i]); q[3] += v / N; q[4] += hyp[i]
            row[f"m{m}"] = {"expected_lost_hypergeometric": float(hyp.sum()), "expected_gained_hypergeometric": float(hyp.sum()),
                            "expected_lost_binomial_plugin": float(sum(p_loss(v / N, m) for v in c)),
                            "empirical": emp, "empirical_mean_lost": float(np.mean([e["lost_A_some_B_zero"] for e in emp])),
                            "empirical_mean_gained": float(np.mean([e["gained_A_zero_B_some"] for e in emp])),
                            "per_problem_rate_hypergeometric": float(hyp.mean())}
        res["policies"][k] = row
    res["pooled_by_c_bin"] = {f"m{m}": [{"c_range": list(b), "problem_splits": q[0], "mean_p": q[3] / q[0],
                                          "empirical_lost_rate": q[1] / q[0], "empirical_gained_rate": q[2] / q[0],
                                          "hypergeometric_rate": q[4] / q[0], "p_loss_at_mean_p": p_loss(q[3] / q[0], m)}
                                         for b, q in pool[m].items() if q[0]] for m in SPLITS}
    grid = np.linspace(0, 1, 101)
    res["curve"] = {"p": grid.tolist(), **{f"m{m}": [p_loss(p, m) for p in grid] for m in SPLITS},
                    "max": {f"m{m}": {"p": 1 - 2 ** (-1 / m), "P_loss": 0.25} for m in SPLITS},
                    "formula": "P_loss(p, m) = (1 − (1 − p)^m)(1 − p)^m; P_gain(p, m) is the same by symmetry"}
    res["notes"] = [
        "Hypergeometric: expectation over random disjoint splits of the observed 32 attempts (conditional on c); lost and gained are equal by symmetry.",
        "Empirical: fixed index splits. The two m = 8 splits use disjoint attempts; the m = 16 split reuses the same 32 attempts, so the m = 8 and m = 16 counts are not independent.",
        "Pooled bins combine all policies; the same problem appears once per policy, so pooled rows are not independent.",
        f"Dev panel: {DEV_PANEL['problems']} problems × {DEV_PANEL['n']} attempts per evaluation ({DEV_PANEL['source']}). The per-problem rate "
        "depends on p, and the dev panel has a different p distribution, so test-panel counts must not be rescaled to the dev panel; use P_loss(p, 8)."]
    md = ["# Sampling null: problems that look lost or gained at an unchanged checkpoint", "",
          "Lost = group A has ≥ 1 success and group B has none; gained = the reverse. Counts are out of "
          f"{n} problems.", "",
          "| Policy | m | Expected lost = expected gained (hypergeometric) | Binomial plug-in (p = c/32) | Empirical lost per split | Empirical gained per split |",
          "|---|---|---|---|---|---|"]
    for k, v in res["policies"].items():
        if not ok(v):
            md.append(f"| {k} | | {v} | | | |"); continue
        for m in SPLITS:
            x = v[f"m{m}"]
            md.append(f"| {k} | {m} | {x['expected_lost_hypergeometric']:.2f} | {x['expected_lost_binomial_plugin']:.2f} | "
                      f"{', '.join(str(e['lost_A_some_B_zero']) for e in x['empirical'])} | "
                      f"{', '.join(str(e['gained_A_zero_B_some']) for e in x['empirical'])} |")
    ps = (0.01, 0.02, 0.05, 1 - 2 ** (-1 / 16), 1 - 2 ** (-1 / 8), 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    md += ["", "P_loss(p, m) under independent sampling (per problem; maximum 0.25 at p = 1 − 2^(−1/m)):", "",
           "| p | " + " | ".join(f"{p:.3f}" for p in ps) + " |", "|---|" + "---|" * len(ps)]
    md += [f"| m = {m} | " + " | ".join(f"{p_loss(p, m):.3f}" for p in ps) + " |" for m in SPLITS]
    md += [""] + [f"- {t}" for t in res["notes"]]
    return res, md


# ---------------------------------------------------------------- 4. endpoints
def table(cells, prefix):
    rows, D = [], {name: [] for name, _, _ in CONTRASTS}
    for s in SEEDS:
        row = {"seed": s}
        for a in ARMS:
            x = cells.get(f"{a}-s{s}")
            row[a] = float(x["c"].mean() / N) if ok(x) else "missing"
            row[a + "_pass8"] = float(np.mean([pass8(int(v)) for v in x["c"]])) if ok(x) else "missing"
        for name, a, b in CONTRASTS:
            xa, xb = cells.get(f"{a}-s{s}"), cells.get(f"{b}-s{s}")
            if ok(xa) and ok(xb):
                dr = (xa["c"] - xb["c"]) / N
                row[name] = dict(boot_a9([dr], f"{prefix}{name}/s{s}"), A=row[a], B=row[b], direction=f"{a} − {b}")
                D[name].append(dr)
            else:
                row[name] = "missing"
        rows.append(row)
    mean = {}
    for a in ARMS:
        v = [r[a] for r in rows if r[a] != "missing"]
        mean[a] = float(np.mean(v)) if len(v) == len(SEEDS) else "missing"
    for name, a, b in CONTRASTS:
        mean[name] = (dict(boot_a9(D[name], f"{prefix}{name}/mean"), A=mean[a], B=mean[b], direction=f"{a} − {b}")
                      if len(D[name]) == len(SEEDS) else "missing")
    return {"rows": rows, "mean": mean}


def repro(o, g, key):
    if not (ok(o) and ok(g)):
        return "missing"
    tot = o["y"].size
    same_seed = sum(a == b for ra, rb in zip(o["seed"], g["seed"]) for a, b in zip(ra, rb))
    hashes = all(h is not None for r in o["hash"] + g["hash"] for h in r)
    same_tok = sum(a == b for ra, rb in zip(o["hash"], g["hash"]) for a, b in zip(ra, rb)) if hashes else None
    dc = g["c"] - o["c"]
    po, pg = float(o["c"].mean() / N), float(g["c"].mean() / N)
    return {"attempts_identical_reward": frac((o["y"] == g["y"]).sum(), tot), "attempts_identical_request_seed": frac(same_seed, tot),
            "attempts_identical_token_ids": frac(same_tok, tot) if hashes else "unavailable (records lack ids)",
            "problems_identical_c": frac((dc == 0).sum(), len(dc)), "max_abs_delta_c": int(np.abs(dc).max()),
            "original_pass1": po, "cgpu_pass1": pg,
            "delta_pass1_cgpu_minus_original": dict(boot_a9([dc / N], f"a17-repro/{key}"), A=pg, B=po, direction="cgpu − original")}


def endpoints(pol, cg, src):
    orig = {k: v for k, v in pol.items() if k != "base"}
    common = {"H-s101": pol["H-s101"], "H-s202": pol["H-s202"], **cg}
    csrc = {"H-s101": "original evaluation (ran on L40S)", "H-s202": f"endpoint {src['H-s202']} (common-GPU)",
            **{k: f"qis-cgpu-{k}" for k in cg}, "base": "original evaluation (ran on L40S)"}
    res = {"primary_original": table(orig, ""), "reproducibility_check_common_gpu": table(common, "a17-cgpu/"),
           "common_gpu_sources": csrc,
           "base": {"pass1": float(pol["base"]["c"].mean() / N), "pass8": float(np.mean([pass8(int(v)) for v in pol["base"]["c"]]))},
           "reproducibility": {k: repro(pol[k], cg.get(k), k) for k in ("R-s101", "R-s202", "U-s101", "U-s202")},
           "interval_note": "95% paired bootstrap over the 128 problems (10,000 draws, seed 2027; streams a9/<contrast>/s<seed> "
                            "for the original table as in report_tables, a9/a17-cgpu/... and a9/a17-repro/... otherwise); "
                            "conditional on these trained policies"}
    md = ["# Endpoints (sealed test, pass@1, 128 problems × 32 attempts)", "",
          f"Base model: pass@1 {res['base']['pass1']:.3f}, pass@8 {res['base']['pass8']:.3f}."]
    for title, key in (("Primary: original evaluations", "primary_original"),
                       ("Reproducibility check: common GPU (L40S, original request seeds)", "reproducibility_check_common_gpu")):
        T = res[key]
        md += ["", f"## {title}", "", "| Seed | H | R | U | R − U | R − H | U − H | pass@8 H / R / U |", "|---|---|---|---|---|---|---|---|"]
        for r in T["rows"] + [dict(T["mean"], seed="Mean")]:
            p8 = " / ".join(f3(r.get(a + "_pass8")) for a in ARMS) if "H_pass8" in r else "–"
            md.append(f"| {r['seed']} | {f3(r['H'])} | {f3(r['R'])} | {f3(r['U'])} | " +
                      " | ".join(dl(r[nm]["A"], r[nm]["B"], r[nm]["est"], r[nm]) if ok(r[nm]) else "missing" for nm, _, _ in CONTRASTS) +
                      f" | {p8} |")
    md += ["", "Common-GPU cell sources: " + "; ".join(f"{k}: {v}" for k, v in csrc.items()) + ".", "",
           "## Reproducibility per checkpoint (cgpu vs original)", "",
           "| Checkpoint | Identical reward | Identical request seed | Identical token ids | Identical c/32 | max abs Δc | pass@1 cgpu − original |",
           "|---|---|---|---|---|---|---|"]
    for k, v in res["reproducibility"].items():
        if not ok(v):
            md.append(f"| {k} | {v} | | | | | |"); continue
        dd = v["delta_pass1_cgpu_minus_original"]
        md.append(f"| {k} | {v['attempts_identical_reward']} | {v['attempts_identical_request_seed']} | {v['attempts_identical_token_ids']} | "
                  f"{v['problems_identical_c']} | {v['max_abs_delta_c']} | {dl(dd['A'], dd['B'], dd['est'], dd)} |")
    md += ["", res["interval_note"] + "."]
    return res, md


# ---------------------------------------------------------------- 6. truncation
def tdec(TR, CR, TU, CU, tot):
    """Totals (scalars or bootstrap vectors) → t_R, t_U, a_R, a_U, truncation term −Δt·ā, accuracy term (1 − t̄)·Δa."""
    tR, tU = TR / tot, TU / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        aR, aU = CR / (tot - TR), CU / (tot - TU)
    return tR, tU, aR, aU, -(tR - tU) * (aR + aU) / 2, (1 - (tR + tU) / 2) * (aR - aU)


def tblock(r, u, ix, name):
    """Exact R − U split over problems ix; both terms bootstrapped on one shared problem draw (stream a9/<name>)."""
    TR, CR, TU, CU = r["trunc"][ix].sum(1), r["y"][ix].sum(1), u["trunc"][ix].sum(1), u["y"][ix].sum(1)
    tot = N * len(ix)
    tR, tU, aR, aU, tt, at = tdec(TR.sum(), CR.sum(), TU.sum(), CU.sum(), tot)
    pR, pU = CR.sum() / tot, CU.sum() / tot
    idx = seeds.rng(seeds.BOOTSTRAP_SEED, f"a9/{name}").integers(0, len(ix), (DRAWS, len(ix)))
    b = tdec(TR[idx].sum(1), CR[idx].sum(1), TU[idx].sum(1), CU[idx].sum(1), tot)
    ci = lambda v, e: {"est": float(e), "lo": float(np.nanquantile(v, .025)), "hi": float(np.nanquantile(v, .975))}  # noqa: E731
    return {"n_problems": len(ix), "R": {"t": float(tR), "a": float(aR), "pass1": float(pR)},
            "U": {"t": float(tU), "a": float(aU), "pass1": float(pU)}, "R-U": float(pR - pU),
            "t_bar": float((tR + tU) / 2), "a_bar": float((aR + aU) / 2), "delta_t": float(tR - tU), "delta_a": float(aR - aU),
            "truncation_term": ci(b[4], tt), "accuracy_term": ci(b[5], at),
            "sum_of_terms": float(tt + at), "identity_residual": float(tt + at - (pR - pU)), "stream": f"a9/{name}"}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None


def truncation(pol, probs, kb):
    n = len(probs)
    band = np.array([band_of(int(k)) for k in kb])
    res = {"n_problems": n, "identity": "pass1 = (1 − t)·a per policy; R − U = (1−t_R)a_R − (1−t_U)a_U = −Δt·ā + (1 − t̄)·Δa exactly, "
           "Δ = R − U, bars = mean of R and U; t = truncated/attempts, a = correct/finished",
           "policies": {}, "decomposition": {}, "bands": {}, "per_problem": {}, "within_8192": {}}
    usable = lambda x: ok(x) and x["trunc"] is not None                 # noqa: E731
    for k, x in pol.items():
        if not usable(x):
            res["policies"][k] = x if not ok(x) else "unavailable (records lack finish)"; continue
        y, t, nt = x["y"], x["trunc"], x["ntok"]
        tot, T, C = y.size, int(t.sum()), int(y.sum())
        F = tot - T
        row = {"truncated": frac(T, tot), "t": T / tot, "finished": frac(F, tot), "correct": frac(C, tot),
               "a_finished": C / F if F else None, "pass1": C / tot, "pass1_check_1mt_times_a": (1 - T / tot) * (C / F) if F else None,
               "truncated_and_correct": int((t & y).sum())}
        if nt is not None:
            row["mean_tokens"] = {"finished_correct": mean_or_none(nt[~t & y]), "finished_wrong": mean_or_none(nt[~t & ~y]),
                                  "truncated": mean_or_none(nt[t])}
            c8 = ((nt <= 8192) & y).sum(1)
            row["within_8192"] = {"correct": frac(c8.sum(), tot), "pass1": float(c8.sum() / tot)}
        else:
            row["mean_tokens"] = row["within_8192"] = "unavailable (records lack n_tokens)"
        res["policies"][k] = row
    allix = np.arange(n)
    for s in SEEDS:
        r, u = pol[f"R-s{s}"], pol[f"U-s{s}"]
        if not (usable(r) and usable(u)):
            for sec in ("decomposition", "per_problem", "within_8192"):
                res[sec][f"s{s}"] = "missing"
            res["bands"][f"s{s}"] = "missing"; continue
        res["decomposition"][f"s{s}"] = tblock(r, u, allix, f"a17-trunc/R-U/s{s}")
        res["bands"][f"s{s}"] = {bn: tblock(r, u, np.where(band == bn)[0], f"a17-trunc/{bn}/R-U/s{s}") if (band == bn).any() else None
                                 for bn, _, _ in BANDS}
        dt = (r["trunc"].sum(1) - u["trunc"].sum(1)) / N
        ds = (r["c"] - u["c"]) / N
        res["per_problem"][f"s{s}"] = {
            "pearson_dt_dsuccess": corr(dt, ds), "spearman_dt_dsuccess": spearman(dt, ds),
            "success_same_sign_as_minus_dt": frac((ds * -dt > 0).sum(), n), "success_opposite_sign_to_minus_dt": frac((ds * -dt < 0).sum(), n),
            "either_zero": frac(((ds == 0) | (dt == 0)).sum(), n),
            "items": [{"id": probs[i], "band": band[i], "dt": float(dt[i]), "dsuccess": float(ds[i]),
                       "t_R": f"{int(r['trunc'][i].sum())}/32", "t_U": f"{int(u['trunc'][i].sum())}/32",
                       "c_R": f"{int(r['c'][i])}/32", "c_U": f"{int(u['c'][i])}/32"} for i in range(n)]}
        if r["ntok"] is not None and u["ntok"] is not None:
            cR, cU = ((r["ntok"] <= 8192) & r["y"]).sum(1), ((u["ntok"] <= 8192) & u["y"]).sum(1)
            res["within_8192"][f"s{s}"] = {
                "R-U_within_8192": dict(boot_a9([(cR - cU) / N], f"a17-trunc/8192/R-U/s{s}"), A=float(cR.mean() / N),
                                        B=float(cU.mean() / N), direction="R − U"),
                "R-U_cap_12288": dict(boot_a9([ds], f"a17-trunc/12288/R-U/s{s}"), A=float(r["c"].mean() / N),
                                      B=float(u["c"].mean() / N), direction="R − U")}
        else:
            res["within_8192"][f"s{s}"] = "unavailable (records lack n_tokens)"
    md = ["# Truncation and the R − U difference (sealed test, cap 12,288 tokens)", "",
          "Truncated = finish == \"length\" (always reward 0). t = truncated / attempts; a = correct / finished; pass@1 = (1 − t)·a.",
          "Identity (exact): R − U = (1 − t_R)·a_R − (1 − t_U)·a_U = −(t_R − t_U)·ā + (1 − t̄)·(a_R − a_U), with ā, t̄ the R/U means.", "",
          "| Policy | Truncated (t) | Correct / finished (a) | pass@1 | (1 − t)·a | Truncated and correct | Mean tokens: correct / wrong / truncated | pass@1 within 8,192 |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in res["policies"].items():
        if not ok(v):
            md.append(f"| {k} | {v} | | | | | | |"); continue
        mt, w8 = v["mean_tokens"], v["within_8192"]
        mts = " / ".join(f3(mt[x], ".0f") for x in ("finished_correct", "finished_wrong", "truncated")) if ok(mt) else mt
        md.append(f"| {k} | {v['truncated']} ({v['t']:.4f}) | {v['correct'].split('/')[0]}/{v['finished'].split('/')[0]} ({f3(v['a_finished'], '.4f')}) | "
                  f"{v['pass1']:.4f} | {f3(v['pass1_check_1mt_times_a'], '.4f')} | {v['truncated_and_correct']} | {mts} | "
                  f"{f3(w8['pass1']) if ok(w8) else w8} |")

    def term_cells(x):
        tt, at = x["truncation_term"], x["accuracy_term"]
        return (f"−({x['R']['t']:.4f} − {x['U']['t']:.4f})·{x['a_bar']:.4f} = {tt['est']:+.4f} [{tt['lo']:+.4f}, {tt['hi']:+.4f}]",
                f"(1 − {x['t_bar']:.4f})·({x['R']['a']:.4f} − {x['U']['a']:.4f}) = {at['est']:+.4f} [{at['lo']:+.4f}, {at['hi']:+.4f}]")
    md += ["", "## R − U split per seed (95% problem bootstrap, 10,000 draws, seed 2027, streams a9/a17-trunc/...)", "",
           "| Seed | R − U pass@1 (R minus U) | Truncation term −Δt·ā | Accuracy term (1 − t̄)·Δa | Sum of terms (residual) |", "|---|---|---|---|---|"]
    for s, x in res["decomposition"].items():
        if not ok(x):
            md.append(f"| {s} | {x} | | | |"); continue
        a1, a2 = term_cells(x)
        md.append(f"| {s} | {dl(x['R']['pass1'], x['U']['pass1'])} | {a1} | {a2} | {x['sum_of_terms']:+.4f} ({x['identity_residual']:.1e}) |")
    md += ["", "## Per band (bands from base samples 0-15)", "", "| Seed | Band | Problems | R − U pass@1 | Truncation term | Accuracy term |", "|---|---|---|---|---|---|"]
    for s, B in res["bands"].items():
        if not ok(B):
            md.append(f"| {s} | {B} | | | | |"); continue
        for bn, x in B.items():
            if x is None:
                md.append(f"| {s} | {bn} | 0/{n} | – | – | – |"); continue
            a1, a2 = term_cells(x)
            md.append(f"| {s} | {bn} | {x['n_problems']}/{n} | {dl(x['R']['pass1'], x['U']['pass1'])} | {a1} | {a2} |")
    md += ["", "## Per problem: Δt = t_R − t_U vs Δsuccess = c_R/32 − c_U/32", "",
           "| Seed | Pearson | Spearman | Δsuccess same sign as −Δt | Opposite sign | Either zero |", "|---|---|---|---|---|---|"]
    for s, x in res["per_problem"].items():
        md.append(f"| {s} | {x} | | | | |" if not ok(x) else
                  f"| {s} | {f3(x['pearson_dt_dsuccess'])} | {f3(x['spearman_dt_dsuccess'])} | {x['success_same_sign_as_minus_dt']} | "
                  f"{x['success_opposite_sign_to_minus_dt']} | {x['either_zero']} |")
    md += ["", "## Does the reversal exist within 8,192 tokens? (success = reward 1 and n_tokens ≤ 8,192; same traces)", "",
           "| Seed | R − U at cap 12,288 | R − U within 8,192 |", "|---|---|---|"]
    for s, x in res["within_8192"].items():
        if not ok(x):
            md.append(f"| {s} | {x} | |"); continue
        g = lambda z: dl(z["A"], z["B"], z["est"], z)                    # noqa: E731
        md.append(f"| {s} | {g(x['R-U_cap_12288'])} | {g(x['R-U_within_8192'])} |")
    return res, md


# ---------------------------------------------------------------- 5. figures
def plot(out_dir, fig_dir):
    """Draw the four A17 figures from the JSON outputs in out_dir (matplotlib venv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qis.analysis.figures import ARM_COLOR
    os.makedirs(fig_dir, exist_ok=True)
    J = lambda n: json.load(open(os.path.join(out_dir, n + ".json"))) if os.path.exists(os.path.join(out_dir, n + ".json")) else None  # noqa: E731
    drawn, BC = [], dict(zip([b for b, _, _ in BANDS], plt.get_cmap("Purples")(np.linspace(0.35, 1.0, len(BANDS)))))
    E = J("endpoints")
    if E:
        T = {"original": E["primary_original"], "common GPU": E["reproducibility_check_common_gpu"]}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
        cap = []
        for t, (lab, tab) in enumerate(T.items()):
            cap.append(f"{lab}: " + "; ".join(f"s{r['seed']} " + ", ".join(f"{a} {f3(r[a])}" for a in ARMS) for r in tab["rows"]))
            for i, r in enumerate(tab["rows"]):
                for j, a in enumerate(ARMS):
                    x = i * 8 + j * 2 + t * 0.8
                    if num(r[a]):
                        a1.bar(x, r[a], 0.8, color=ARM_COLOR[a], hatch="//" if t else None, edgecolor="white",
                               label=f"{a} {lab}" if i == 0 else None)
                        a1.text(x, r[a] + 0.01, f"{r[a]:.3f}", ha="center", fontsize=6, rotation=90)
                    else:
                        a1.text(x, 0.02, "missing", ha="center", fontsize=6, rotation=90)
        a1.axhline(E["base"]["pass1"], color="black", lw=0.8, ls="--", label=f"base {E['base']['pass1']:.3f}")
        a1.set_xticks([2.4, 10.4]); a1.set_xticklabels([f"seed {s}" for s in SEEDS]); a1.set_ylim(0, 1.05)
        a1.set_ylabel("target_test pass@1 (128 × 32)"); a1.legend(fontsize=6, ncol=2, loc="lower right")
        k, ticks = 0, []
        for name, _, _ in CONTRASTS:
            for row_key in ("101", "202", "mean"):
                for t, (lab, tab) in enumerate(T.items()):
                    r = tab["mean"] if row_key == "mean" else next(x for x in tab["rows"] if str(x["seed"]) == row_key)
                    x = r[name]
                    if ok(x):
                        xx = k + (t - 0.5) * 0.3
                        a2.errorbar(xx, x["est"], yerr=[[x["est"] - x["lo"]], [x["hi"] - x["est"]]], fmt="os"[t],
                                    color=("black", "grey")[t], capsize=3, label=lab if k == 0 else None)
                        a2.annotate(f"{x['A']:.3f}−{x['B']:.3f}", (xx, x["hi"]), fontsize=5, ha="center", va="bottom", rotation=90)
                ticks.append(f"{name}\n{row_key}"); k += 1
        a2.axhline(0, color="grey", lw=0.8); a2.set_xticks(range(k)); a2.set_xticklabels(ticks, fontsize=6)
        a2.set_ylabel("paired difference in pass@1 (95% problem bootstrap)"); a2.legend(fontsize=7)
        fig.suptitle("\n".join(cap), fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "endpoints.png"), dpi=150); plt.close(fig)
        drawn.append("endpoints")
    R = J("reversal")
    if R and "per_problem" in R:
        P = R["per_problem"]
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        ax.errorbar([p["d101"] for p in P], [p["d202"] for p in P], xerr=[p["se101"] for p in P], yerr=[p["se202"] for p in P],
                    fmt="none", ecolor="lightgrey", elinewidth=0.6, alpha=0.6, zorder=1)
        for bn, _, _ in BANDS:
            for tie in (False, True):
                q = [p for p in P if p["band"] == bn and ((p["d101"] == 0 or p["d202"] == 0) == tie)]
                if q:
                    ax.scatter([p["d101"] for p in q], [p["d202"] for p in q], s=22, marker="x" if tie else "o", color=BC[bn],
                               edgecolors="white" if not tie else None, linewidths=0.5 if not tie else 1.2, zorder=3,
                               label=f"{bn}" + (" (tie in a seed)" if tie else ""))
        lim = max(0.05, max(abs(v) for p in P for v in (p["d101"], p["d202"])) + 0.05)
        ax.plot([-lim, lim], [-lim, lim], ls=":", color="grey", lw=0.8); ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        q = lambda f: sum(f(p["d101"], p["d202"]) for p in P)                # noqa: E731
        for (sx, sy), cnt in (((1, 1), q(lambda a, b: a > 0 and b > 0)), ((-1, 1), q(lambda a, b: a < 0 and b > 0)),
                              ((-1, -1), q(lambda a, b: a < 0 and b < 0)), ((1, -1), q(lambda a, b: a > 0 and b < 0))):
            ax.text(sx * 0.95 * lim, sy * 0.95 * lim, f"{cnt}/{len(P)}", ha="right" if sx > 0 else "left",
                    va="top" if sy > 0 else "bottom", fontsize=8)
        ag = R["agreement"]
        ax.text(0.02, 0.5, f"tie in either seed: {ag['tie_in_either_seed']}/{ag['n']}\ntie in both: {ag['tie_in_both_seeds']}/{ag['n']}",
                transform=ax.transAxes, fontsize=7, va="center")
        o = R["overall"]
        ax.plot(o["s101"]["R-U"], o["s202"]["R-U"], "*", color="black", ms=11, zorder=4,
                label=f"mean: s101 {o['s101']['R']:.3f}−{o['s101']['U']:.3f}, s202 {o['s202']['R']:.3f}−{o['s202']['U']:.3f}")
        ax.set_xlabel("seed 101: R − U pass@1 per problem (c/32; ±1 SE)"); ax.set_ylabel("seed 202: R − U pass@1 per problem (c/32; ±1 SE)")
        ax.legend(fontsize=6, loc="lower right"); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "reversal_scatter.png"), dpi=150); plt.close(fig)
        drawn.append("reversal_scatter")
    C = J("coverage")
    if C:
        ks = [k for k, v in C["policies"].items() if ok(v)]
        col = [ARM_COLOR.get(k.split("-")[0], "grey") if k != "base" else "grey" for k in ks]
        hat = ["//" if k.endswith("s202") else None for k in ks]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
        for ax, f, lab, txt in ((a1, "solved_any_n", f"problems solved ≥ 1 of 32 (of {C['n_problems']})", lambda v: v["solved_any"]),
                                (a2, "pass1", "pass@1 (128 × 32)", lambda v: f"{v['pass1']:.3f}")):
            for i, k in enumerate(ks):
                v = C["policies"][k]
                ax.bar(i, v[f], color=col[i], hatch=hat[i], edgecolor="white")
                ax.text(i, v[f], txt(v), ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=7, rotation=30); ax.set_ylabel(lab)
        a1.set_ylim(0, C["n_problems"] * 1.08); a2.set_ylim(0, 1.05)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "coverage.png"), dpi=150); plt.close(fig)
        drawn.append("coverage")
    S = J("sampling_null")
    if S:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for m, ls, mk in ((8, "-", "o"), (16, "--", "s")):
            ax.plot(S["curve"]["p"], S["curve"][f"m{m}"], ls, color="black", lw=1.2, label=f"P_loss(p, {m}), independent sampling")
            B = S["pooled_by_c_bin"][f"m{m}"]
            ax.plot([b["mean_p"] for b in B], [b["empirical_lost_rate"] for b in B], mk, color="#dd8452", ms=6,
                    label=f"observed lost rate, m = {m} (pooled policies, binned by c/32)")
            ax.plot([b["mean_p"] for b in B], [b["hypergeometric_rate"] for b in B], mk, mfc="none", color="#4c72b0", ms=8,
                    label=f"hypergeometric expectation, m = {m}")
        ax.set_xlabel("problem success probability p (observed bins: c/32)"); ax.set_ylabel("P(A ≥ 1 success, B = 0) per problem")
        ax.set_title(f"dev panel: {S['dev_panel']['problems']} problems × {S['dev_panel']['n']} attempts → m = 8 curve applies", fontsize=8)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "sampling_null.png"), dpi=150); plt.close(fig)
        drawn.append("sampling_null")
    Tr = J("truncation")
    if Tr and any(ok(v) for v in Tr["per_problem"].values()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
        for ax, s in zip(axes[:2], SEEDS):
            x = Tr["per_problem"].get(f"s{s}")
            if not ok(x):
                ax.text(0.5, 0.5, f"seed {s}: missing", transform=ax.transAxes, ha="center"); continue
            for bn, _, _ in BANDS:
                q = [i for i in x["items"] if i["band"] == bn]
                if q:
                    ax.scatter([i["dt"] for i in q], [i["dsuccess"] for i in q], s=20, color=BC[bn], edgecolors="white",
                               linewidths=0.5, label=bn)
            ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
            ax.set_title(f"seed {s}: Pearson {f3(x['pearson_dt_dsuccess'])}, Spearman {f3(x['spearman_dt_dsuccess'])}; "
                         f"same sign as −Δt {x['success_same_sign_as_minus_dt']}", fontsize=7)
            ax.set_xlabel("Δt = truncated share R − U (per problem, of 32)"); ax.set_ylabel("Δsuccess = c/32 R − U")
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
        ax, k, ticks = axes[2], 0, []
        for s in SEEDS:
            x = Tr["decomposition"].get(f"s{s}")
            if ok(x):
                for j, (term, col) in enumerate((("truncation_term", "#8c8c8c"), ("accuracy_term", "#4c4c4c"))):
                    v = x[term]
                    ax.bar(k + j * 0.4, v["est"], 0.38, color=col, label=term.replace("_", " ") if k == 0 else None)
                    ax.errorbar(k + j * 0.4, v["est"], yerr=[[v["est"] - v["lo"]], [v["hi"] - v["est"]]], fmt="none", color="black", capsize=3)
                ax.plot(k + 0.2, x["R-U"], "*", color="black", ms=9, label="observed R − U" if k == 0 else None)
                ticks.append(f"seed {s}\nR {x['R']['pass1']:.3f} − U {x['U']['pass1']:.3f}")
            else:
                ticks.append(f"seed {s}\nmissing")
            k += 1
        ax.axhline(0, color="grey", lw=0.8); ax.set_xticks([i + 0.2 for i in range(k)]); ax.set_xticklabels(ticks, fontsize=7)
        ax.set_ylabel("R − U pass@1 = −Δt·ā + (1 − t̄)·Δa (95% problem bootstrap)"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "truncation.png"), dpi=150); plt.close(fig)
        drawn.append("truncation")
    json.dump({"drawn": drawn, "inputs": out_dir}, open(os.path.join(fig_dir, "a17_figures.json"), "w"), indent=1)
    print("drawn:", drawn)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root"); ap.add_argument("out")
    ap.add_argument("--cgpu-root"); ap.add_argument("--h202", default="qis-test-H-s202-sym"); ap.add_argument("--figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    braw = load(os.path.join(a.eval_root, "qis-test-base"))
    if braw is None:
        raise SystemExit(f"base records not found under {a.eval_root}/qis-test-base")
    probs = sorted(braw["rec"])
    pol, src = {"base": cell(braw, probs)}, {"base": "qis-test-base"}
    for arm in ARMS:
        for s in SEEDS:
            k = f"{arm}-s{s}"
            if k == "H-s202":
                roots = [r for r in (a.eval_root, a.cgpu_root) if r and os.path.exists(os.path.join(r, a.h202, "records.jsonl.gz"))]
                d = os.path.join(roots[0] if roots else a.eval_root, a.h202)
            else:
                d = os.path.join(a.eval_root, f"qis-test-{k}-sym")
            pol[k], src[k] = cell(load(d), probs), d
    cg = {f"{arm}-s{s}": cell(load(os.path.join(a.cgpu_root, f"qis-cgpu-{arm}-s{s}")), probs) if a.cgpu_root else "missing (no --cgpu-root)"
          for arm in "RU" for s in SEEDS}
    inputs = {"n_problems": len(probs), "base_incomplete_problems": braw["incomplete"],
              "original": {k: status(v) for k, v in pol.items()}, "cgpu": {k: status(v) for k, v in cg.items()}}
    kb = pol["base"]["y"][:, :16].sum(1)
    for name, (res, md) in (("reversal", reversal(pol, probs, kb)), ("coverage", coverage(pol, probs)),
                            ("sampling_null", sampling_null(pol, probs)), ("endpoints", endpoints(pol, cg, src)),
                            ("truncation", truncation(pol, probs, kb))):
        dump(a.out, name, dict(res, inputs=inputs), md)
    if a.figures:
        plot(a.out, a.figures)


if __name__ == "__main__":
    main()
