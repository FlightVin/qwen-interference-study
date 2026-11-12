"""A26 pass A (agent a-evals): claim-by-claim recomputation of report v2 evaluation numbers.

Inputs (laptop; no project imports):
  SCAN    scan.json / scan_pp.json written by scan_evals.py on compute (per-attempt outcome strings per problem)
  TRACES  traces.json written by traces.py on compute (canonical in-run eval rows)
  repo    research/evidence/** (production analysis outputs, the comparison), reports/archive/REPORT-v2.md
Outputs: research/audit-a26/pass-a/data/numbers.json (unrounded) and a markdown claims table on stdout.
Bootstrap: paired problem bootstrap, 10,000 draws, numpy default_rng(20261111) -- an RNG independent of production;
intervals are compared with a tolerance of 0.3 points because the draws differ.
Usage: python3 recompute_claims.py SCAN_DIR REPO OUT_JSON
"""
import json
import math
import re
import sys
from collections import Counter

import numpy as np

SD, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
scan = json.load(open(f"{SD}/scan.json"))
PP = json.load(open(f"{SD}/scan_pp.json"))
TR = json.load(open(f"{SD}/traces.json"))
EV = f"{REPO}/research/evidence"
REPORT = f"{REPO}/reports/archive/REPORT-v2.md"
RNG = np.random.default_rng(20261111)
DRAWS = 10_000
TEST_IDS = scan["panels"]["test"]["ids"]
STRESS_IDS = scan["panels"]["stress"]["ids"]
TIER = scan["panels"]["stress"]["tier"]
claims = []
R = {}          # unrounded recomputed values


def arr(lab, ids, what="c"):
    p = PP[lab]
    if what == "c":
        return np.array([p[i]["c"] for i in ids], float)
    if what == "t":
        return np.array([p[i]["t"] for i in ids], float)
    if what == "c8k":
        return np.array([p[i]["c8k"] for i in ids], float)
    raise KeyError(what)


def n_of(lab):
    return 32 if lab.startswith(("test/", "cgpu/")) else 8


def boot_diff(d):
    """d: per-problem paired differences (already in pass@1 units). Returns est, lo, hi."""
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), (DRAWS, len(d)))
    b = d[idx].mean(1)
    return {"est": float(d.mean()), "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


def contrast(la, lb, ids):
    n = n_of(la)
    return boot_diff(arr(la, ids) / n - arr(lb, ids) / n)


def decomp(la, lb, ids):
    """Exact split R − U = −Δt·ā + (1 − t̄)·Δa (pooled over attempts), problem bootstrap for both terms."""
    n = n_of(la)
    cA, tA, cB, tB = arr(la, ids), arr(la, ids, "t"), arr(lb, ids), arr(lb, ids, "t")

    def f(cA, tA, cB, tB):
        tot = n * cA.shape[-1]
        TA, TB, CA, CB = tA.sum(-1) / tot, tB.sum(-1) / tot, cA.sum(-1), cB.sum(-1)
        aA, aB = CA / (tot * (1 - TA)), CB / (tot * (1 - TB))
        return TA, TB, aA, aB, -(TA - TB) * (aA + aB) / 2, (1 - (TA + TB) / 2) * (aA - aB)
    e = f(cA, tA, cB, tB)
    idx = RNG.integers(0, len(ids), (DRAWS, len(ids)))
    b = f(cA[idx], tA[idx], cB[idx], tB[idx])
    q = lambda v: (float(np.quantile(v, .025)), float(np.quantile(v, .975)))  # noqa: E731
    return {"tA": float(e[0]), "tB": float(e[1]), "aA": float(e[2]), "aB": float(e[3]),
            "trunc_term": {"est": float(e[4]), "lo": q(b[4])[0], "hi": q(b[4])[1]},
            "acc_term": {"est": float(e[5]), "lo": q(b[5])[0], "hi": q(b[5])[1]},
            "residual": float(e[4] + e[5] - (cA.sum() - cB.sum()) / (n * len(ids)))}


def pass1(lab, ids):
    return float((arr(lab, ids) / n_of(lab)).mean())


def pass8(lab, ids):
    n = n_of(lab)
    return float(np.mean([1.0 if n - c < 8 else 1 - math.comb(n - int(c), 8) / math.comb(n, 8) for c in arr(lab, ids)]))


def pct(x, d=1):
    return round(100 * x, d)


def add(cid, section, claim, report, recomputed, digits=1, unit="pct", production=None, note="", status=None, tol=None):
    """report: displayed value(s); recomputed: unrounded fraction(s) (unit pct → ×100) or raw numbers."""
    rep = report if isinstance(report, (list, tuple)) else [report]
    rec = recomputed if isinstance(recomputed, (list, tuple)) else [recomputed]
    shown = []
    ok = True
    for a, b in zip(rep, rec):
        if b is None or a is None:
            shown.append(None); ok = False; continue
        v = 100 * b if unit == "pct" else b
        r = round(v, digits) if digits is not None else v
        if digits == 0:
            r = int(round(v))
        shown.append(r)
        t = tol if tol is not None else (0.5 * 10 ** -digits + 1e-9 if digits is not None else 1e-9)
        if abs(float(a) - v) > t:
            ok = False
    st = status or ("verified" if ok else "differs")
    claims.append({"id": cid, "section": section, "claim": claim, "report": rep, "recomputed_display": shown,
                   "recomputed_unrounded": rec, "production": production, "status": st, "note": note})


def addci(cid, section, claim, report_ci, ci, production=None, note=""):
    """report_ci: (est, lo, hi) in points; ci: dict est/lo/hi fractions. Estimate exact at 0.05; bounds within 0.3."""
    est_ok = abs(report_ci[0] - 100 * ci["est"]) <= 0.05 + 1e-9
    b_ok = abs(report_ci[1] - 100 * ci["lo"]) <= 0.3 and abs(report_ci[2] - 100 * ci["hi"]) <= 0.3
    st = "verified" if est_ok and b_ok else ("differs" if not est_ok else "verified (estimate); interval differs by > 0.3")
    claims.append({"id": cid, "section": section, "claim": claim, "report": list(report_ci),
                   "recomputed_display": [round(100 * ci["est"], 2), round(100 * ci["lo"], 2), round(100 * ci["hi"], 2)],
                   "recomputed_unrounded": [ci["est"], ci["lo"], ci["hi"]], "production": production, "status": st, "note": note})


# ------------------------------------------------------------------ sealed test endpoints (sections 2, Q1, opening)
T = TEST_IDS
lab = {k: f"test/{k}" for k in ("base", "H-s101", "H-s202", "H-s202-fb", "R-s101", "R-s202", "U-s101", "U-s202")}
for k, v in lab.items():
    R[f"test.{k}.pass1"] = pass1(v, T)
    R[f"test.{k}.pass8"] = pass8(v, T)
for k in ("R-s101", "R-s202", "U-s101", "U-s202"):
    R[f"cgpu.{k}.pass1"] = pass1(f"cgpu/{k}", T)
prodE = json.load(open(f"{EV}/a17/analysis/test-cgpu/endpoints.json"))
add("2.base", "2, Q1", "base sealed-test pass@1 52.6%, pass@8 83.5%", [52.6, 83.5], [R["test.base.pass1"], R["test.base.pass8"]],
    production=[prodE["base"]["pass1"], prodE["base"]["pass8"]])
for s, (h, r, u) in {101: (77.0, 81.9, 77.7), 202: (72.4, 74.8, 79.2)}.items():
    add(f"2.endpoints.s{s}", "2, Q1", f"seed {s} H / R / U pass@1", [h, r, u],
        [R[f"test.H-s{s}.pass1"], R[f"test.R-s{s}.pass1"], R[f"test.U-s{s}.pass1"]])
prows = {r["seed"]: r for r in prodE["primary_original"]["rows"]}
rep_ci = {("R-U", 101): (4.2, 2.1, 6.3), ("R-H", 101): (4.9, 2.0, 8.0), ("U-H", 101): (0.7, -2.0, 3.5),
          ("R-U", 202): (-4.4, -7.2, -1.7), ("R-H", 202): (2.4, -0.4, 5.4), ("U-H", 202): (6.8, 3.5, 10.2)}
for (nm, s), rc in rep_ci.items():
    a, b = nm.split("-")
    ci = contrast(f"test/{a}-s{s}", f"test/{b}-s{s}", T)
    R[f"test.{nm}.s{s}"] = ci
    addci(f"2.{nm}.s{s}", "2, Q1", f"seed {s} {nm} per-seed paired problem bootstrap", rc, ci,
          production={k: prows[s][nm][k] for k in ("est", "lo", "hi")})
# two-seed means (production: seeds pooled, same problem draw)
for nm, rc in (("R-U", (-0.1, -1.8, 1.5)), ("R-H", (3.6, 1.6, 5.8)), ("U-H", (3.8, 1.6, 6.0))):
    a, b = nm.split("-")
    d = np.mean([arr(f"test/{a}-s{s}", T) / 32 - arr(f"test/{b}-s{s}", T) / 32 for s in (101, 202)], axis=0)
    ci = boot_diff(d)
    R[f"test.{nm}.mean"] = ci
    addci(f"2.{nm}.mean", "2, Q1", f"two-seed mean {nm} (problem bootstrap of the seed-mean per-problem difference)", rc, ci,
          production={k: prodE["primary_original"]["mean"][nm][k] for k in ("est", "lo", "hi")},
          note="Interval resamples problems only; it carries no training-seed variation (n = 2 seeds).")
add("2.means", "2, Q1", "two-seed mean pass@1 H / R / U", [74.7, 78.3, 78.5],
    [(R["test.H-s101.pass1"] + R["test.H-s202.pass1"]) / 2, (R["test.R-s101.pass1"] + R["test.R-s202.pass1"]) / 2,
     (R["test.U-s101.pass1"] + R["test.U-s202.pass1"]) / 2])
# fallback and H-s202 r128 descriptors
fbs = scan["evals"]["test/H-s202-fb"]["records"]
h2 = scan["evals"]["test/H-s202"]["records"]
add("2.fallback", "2", "H-s202 r116 fallback: pass@1 74.3%, pass@8 92.8%, capped 14.8%", [74.3, 92.8, 14.8], [fbs["pass1"], fbs["pass8"], fbs["trunc"]])
add("2.h202", "2", "H-s202 r128: pass@1 72.4%, pass@8 90.8%, capped 15.4%", [72.4, 90.8, 15.4], [h2["pass1"], h2["pass8"], h2["trunc"]])
add("2.fb_minus_r128", "2", "r128 is 1.9 points below r116", 1.9, fbs["pass1"] - h2["pass1"])
# common GPU
crow = {r["seed"]: r for r in prodE["reproducibility_check_common_gpu"]["rows"]}
add("2.cgpu.cells", "2, opening", "common-GPU R-s101 82.3, U-s101 77.8, R-s202 76.1, U-s202 79.2", [82.3, 77.8, 76.1, 79.2],
    [R["cgpu.R-s101.pass1"], R["cgpu.U-s101.pass1"], R["cgpu.R-s202.pass1"], R["cgpu.U-s202.pass1"]])
rep_cg = {("R-U", 101): (4.4, 2.4, 6.5), ("R-H", 101): (5.3, 2.5, 8.2), ("U-H", 101): (0.9, -1.8, 3.5),
          ("R-U", 202): (-3.1, -6.0, -0.4), ("R-H", 202): (3.7, 0.9, 6.6), ("U-H", 202): (6.8, 3.5, 10.2)}
for (nm, s), rc in rep_cg.items():
    a, b = nm.split("-")
    la = f"cgpu/{a}-s{s}" if a in "RU" else f"test/{a}-s{s}"
    lb = f"cgpu/{b}-s{s}" if b in "RU" else f"test/{b}-s{s}"
    ci = contrast(la, lb, T)
    addci(f"2.cgpu.{nm}.s{s}", "2", f"common-GPU seed {s} {nm}", rc, ci, production={k: crow[s][nm][k] for k in ("est", "lo", "hi")})
for nm, rc in (("R-U", (0.7, -1.0, 2.3)), ("R-H", (4.5, 2.6, 6.4)), ("U-H", (3.8, 1.7, 6.0))):
    a, b = nm.split("-")
    g = lambda x, s: f"cgpu/{x}-s{s}" if x in "RU" else f"test/{x}-s{s}"  # noqa: E731
    d = np.mean([arr(g(a, s), T) / 32 - arr(g(b, s), T) / 32 for s in (101, 202)], axis=0)
    addci(f"2.cgpu.{nm}.mean", "2", f"common-GPU two-seed mean {nm}", rc, boot_diff(d),
          production={k: prodE["reproducibility_check_common_gpu"]["mean"][nm][k] for k in ("est", "lo", "hi")})
idn = scan["cgpu_identity"]
for k, (tok, cc, ci) in {"R-s101": (48, 62, (0.4, -0.8, 1.6)), "R-s202": (133, 46, (1.3, 0.2, 2.5)), "U-s101": (1964, 79, (0.1, -1.0, 1.2)),
                         "U-s202": (4096, 128, (0.0, 0.0, 0.0))}.items():
    add(f"2.repro.{k}", "2", f"{k} rerun: identical token sequences / identical c per problem", [tok, cc],
        [idn[k]["identical_token_ids"], idn[k]["identical_c"]], digits=0, unit="raw")
    if k != "U-s202":
        addci(f"2.repro.{k}.delta", "2", f"{k} pass@1 rerun − original", ci, boot_diff(arr(f"cgpu/{k}", T) / 32 - arr(f"test/{k}", T) / 32),
              production={kk: prodE["reproducibility"][k]["delta_pass1_cgpu_minus_original"][kk] for kk in ("est", "lo", "hi")})
add("2.repro.max", "2", "a rerun moved pass@1 by at most 1.3 points", 1.3,
    max(abs(R[f"cgpu.{k}.pass1"] - R[f"test.{k}.pass1"]) for k in ("R-s101", "R-s202", "U-s101", "U-s202")))
add("2.repro.u101share", "2", "U-s101 matched on only 48% of attempts", 48, idn["U-s101"]["identical_token_ids"] / 4096, digits=0)

# ------------------------------------------------------------------ ranges in the opening (A26 §4 item 1)
trained6 = [R[f"test.{k}.pass1"] for k in ("H-s101", "H-s202", "R-s101", "R-s202", "U-s101", "U-s202")]
trained5 = [R[f"test.{k}.pass1"] for k in ("H-s101", "R-s101", "R-s202", "U-s101", "U-s202")]
R["range.trained6"] = [min(trained6), max(trained6)]
R["range.trained5_noHs202"] = [min(trained5), max(trained5)]
add("open.range", "opening 1, Q1", "'five finished trained policies scored 74.8%–81.9%' (all six finished; H-s202 r128 = 72.4%)",
    [74.8, 81.9], R["range.trained6"], note="Range of the six r128 endpoints is 72.4%–81.9%; 74.8% is the minimum without H-s202.")
add("open.gain", "opening 1, Q1", "gain +22 to +29 points over base (six endpoints: +19.8 to +29.3)", [22, 29],
    [R["range.trained6"][0] - R["test.base.pass1"], R["range.trained6"][1] - R["test.base.pass1"]], digits=0,
    note="Without H-s202: +22.2 to +29.3.")

# ------------------------------------------------------------------ 4.1 reversal and bands
c = {k: arr(f"test/{k}", T) for k in ("R-s101", "U-s101", "R-s202", "U-s202", "base")}
d101, d202 = (c["R-s101"] - c["U-s101"]) / 32, (c["R-s202"] - c["U-s202"]) / 32
same = int(np.sum((d101 != 0) & (d202 != 0) & (np.sign(d101) == np.sign(d202))))
opp = int(np.sum((d101 != 0) & (d202 != 0) & (np.sign(d101) != np.sign(d202))))
tie_e = int(np.sum((d101 == 0) | (d202 == 0)))
tie_b = int(np.sum((d101 == 0) & (d202 == 0)))
all0 = int(np.sum([(c[k] == 0) for k in ("R-s101", "U-s101", "R-s202", "U-s202")], axis=0).__eq__(4).sum())
all32 = int(np.sum([(c[k] == 32) for k in ("R-s101", "U-s101", "R-s202", "U-s202")], axis=0).__eq__(4).sum())
add("4.1.signs", "4.1", "same 31 / opposite 44 / tie either 53 / tie both 31 / all four 0/32: 3 / all four 32/32: 26",
    [31, 44, 53, 31, 3, 26], [same, opp, tie_e, tie_b, all0, all32], digits=0, unit="raw")


def rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = np.asarray(x)[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


base_s = {i: PP["test/base"][i]["s"] for i in T}
k16 = np.array([base_s[i][:16].count("C") for i in T])
band = np.select([k16 == 0, k16 <= 5, k16 <= 10, k16 <= 15], ["B0", "B1", "B2", "B3"], "B4")
R["spearman.overall"] = spearman(d101, d202)
R["spearman.B1"] = spearman(d101[band == "B1"], d202[band == "B1"])
add("4.1.spearman", "4.1, opening 3, Q4", "Spearman of per-problem R − U across seeds −0.09 overall, −0.04 within B1", [-0.09, -0.04],
    [R["spearman.overall"], R["spearman.B1"]], digits=2, unit="raw")
bands_rep = {"B0": (15, 0.8, -10.2, -1.3), "B1": (34, 9.7, -11.8, -5.7), "B2": (26, 3.0, -2.9, -1.2), "B3": (32, 2.4, 2.6, 0.0), "B4": (21, 1.8, -1.2, -0.5)}
for b, (n_, r1, r2, sw) in bands_rep.items():
    m = band == b
    add(f"4.1.band.{b}", "4.1", f"band {b}: size, R − U s101, R − U s202, contribution to swing", [n_, r1, r2, sw],
        [int(m.sum()), float(d101[m].mean()), float(d202[m].mean()), float((d202[m].sum() - d101[m].sum()) / 128)], digits=1, unit="pct_list")
# fix size column (raw) for band claims
for cl in claims:
    if cl["id"].startswith("4.1.band."):
        v = cl["recomputed_unrounded"]
        shown = [v[0]] + [round(100 * x, 1) for x in v[1:]]
        cl["recomputed_display"] = shown
        cl["status"] = "verified" if all(abs(float(a) - float(b)) <= 0.05 + 1e-9 for a, b in zip(cl["report"], shown)) else "differs"
swing = float((d202.sum() - d101.sum()) / 128)
add("4.1.swing", "4.1, opening 3", "swing between seeds −8.6 points; B1 carries −5.7", [-8.6, -5.7],
    [swing, float((d202[band == "B1"].sum() - d101[band == "B1"].sum()) / 128)])
ch = (d202 - d101)[band == "B1"]
srt = ch[np.argsort(-np.abs(ch), kind="mergesort")]
add("4.1.top", "4.1", "within B1 top 3 item changes carry 30%, top 5 45% of B1's share", [30, 45],
    [float(srt[:3].sum() / ch.sum()), float(srt[:5].sum() / ch.sum())], digits=0, note="Signed share; items sorted by |change| (production rule).")
i213 = T.index("algebra/test/00213")
add("4.1.00213", "4.1", "algebra/test/00213: R 23/32 vs U 13/32 (s101), R 11/32 vs U 24/32 (s202)", [23, 13, 11, 24],
    [c["R-s101"][i213], c["U-s101"][i213], c["R-s202"][i213], c["U-s202"][i213]], digits=0, unit="raw")
# reach: both-reach and only-one in s202
bothr = (c["R-s202"] > 0) & (c["U-s202"] > 0)
one = (c["R-s202"] > 0) ^ (c["U-s202"] > 0)
add("4.2.s202reach", "opening 3, 4.2", "seed 202: 122 problems reached by both carry −4.2; 3 reached by one carry −0.2", [122, -4.2, 3, -0.2],
    [int(bothr.sum()), float(d202[bothr].sum() / 128), int(one.sum()), float(d202[one].sum() / 128)], unit="mixed")
cl = claims[-1]
cl["recomputed_display"] = [cl["recomputed_unrounded"][0], round(100 * cl["recomputed_unrounded"][1], 1), cl["recomputed_unrounded"][2], round(100 * cl["recomputed_unrounded"][3], 1)]
cl["status"] = "verified" if cl["recomputed_display"] == [122, -4.2, 3, -0.2] else "differs"

# ------------------------------------------------------------------ 4.2 coverage table
cov_rep = {"base": (52.6, 83.5, 116, 12, 2), "H-s101": (77.0, 93.7, 125, 3, 445), "R-s101": (81.9, 96.1, 125, 3, 437),
           "U-s101": (77.7, 94.5, 125, 3, 501), "R-s202": (74.8, 92.2, 123, 5, 538), "U-s202": (79.2, 94.9, 124, 4, 333),
           "H-s202": (72.4, 90.8, 122, 6, 632)}
for k, (p1, p8, sa, z, tr) in cov_rep.items():
    r = scan["evals"][f"test/{k}"]["records"]
    rec = [round(100 * r["pass1"], 1), round(100 * r["pass8"], 1), r["solved_any"], r["zero"], r["truncated_n"]]
    claims.append({"id": f"4.2.{k}", "section": "4.2", "claim": f"{k}: pass@1, pass@8, solved ≥ 1 of 32, 0/32, truncated attempts",
                   "report": [p1, p8, sa, z, tr], "recomputed_display": rec,
                   "recomputed_unrounded": [r["pass1"], r["pass8"], r["solved_any"], r["zero"], r["truncated_n"]],
                   "production": None, "status": "verified" if rec == [p1, p8, sa, z, tr] else "differs", "note": ""})
# independent baseline (base attempts 16-31)
k16b = np.array([base_s[i][16:].count("C") for i in T])
reach = k16b > 0
base_first = np.array([base_s[i][:16].count("C") for i in T]) / 16
pols = ("H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202")
became = {p: int(((c.get(p) if p in c else arr(f"test/{p}", T))[~reach] > 0).sum()) for p in pols}
gain = {p: R[f"test.{p}.pass1"] - float((k16b / 16).mean()) for p in pols}
reach_p1 = {p: float((arr(f"test/{p}", T)[reach] / 32).mean()) for p in pols}
R["coverage"] = {"unreachable": int((~reach).sum()), "became_reachable": became, "base_reachable_mean_k16b": float((k16b[reach] / 16).mean()),
                 "reachable_pass1": reach_p1, "gain_vs_base_16_31": gain}
add("4.2.unreach", "4.2", "14 problems unreachable at baseline (base attempts 16–31)", 14, int((~reach).sum()), digits=0, unit="raw")
add("4.2.became", "4.2", "training made 9–11 of them reachable (five policies; six incl. H-s202)", [9, 11],
    [min(became[p] for p in pols if p != "H-s202"), max(became[p] for p in pols if p != "H-s202")], digits=0, unit="raw",
    note=f"per policy {became}")
add("4.2.reach_p1", "4.2", "on the 114 reachable problems pass@1 rose from 59.6% to 82.5–88.0%", [59.6, 82.5, 88.0],
    [float((k16b[reach] / 16).mean()), min(reach_p1[p] for p in pols if p != "H-s202"), max(reach_p1[p] for p in pols if p != "H-s202")],
    note=f"H-s202 on reachable = {reach_p1['H-s202']:.4f}; range omits H-s202.")
add("4.2.gain", "4.2", "total gain over base +21.7 to +28.8 points (vs base attempts 16–31)", [21.7, 28.8],
    [min(gain[p] for p in pols if p != "H-s202"), max(gain[p] for p in pols if p != "H-s202")],
    note=f"H-s202 gain = {100 * gain['H-s202']:.2f}; range omits H-s202.")

# ------------------------------------------------------------------ 4.3 truncation decomposition and within-8192
prodT = json.load(open(f"{EV}/a17/analysis/test-cgpu/truncation.json"))
dec_rep = {101: ((10.7, 12.2, 1.4, 0.0, 2.8), (91.6, 88.5, 2.8, 0.9, 4.7)), 202: ((13.1, 8.1, -4.3, -6.1, -2.7), (86.1, 86.2, -0.1, -2.4, 2.0))}
for s, (tr, ac) in dec_rep.items():
    D = decomp(f"test/R-s{s}", f"test/U-s{s}", T)
    R[f"test.decomp.s{s}"] = D
    add(f"4.3.t.s{s}", "4.3, Q2", f"seed {s} truncation shares t_R, t_U", tr[:2], [D["tA"], D["tB"]])
    add(f"4.3.a.s{s}", "4.3, Q2", f"seed {s} accuracy among finished a_R, a_U", ac[:2], [D["aA"], D["aB"]])
    addci(f"4.3.tt.s{s}", "4.3, Q2", f"seed {s} truncation term", tr[2:], D["trunc_term"], production=prodT["decomposition"][f"s{s}"]["truncation_term"])
    addci(f"4.3.at.s{s}", "4.3, Q2", f"seed {s} accuracy term", ac[2:], D["acc_term"], production=prodT["decomposition"][f"s{s}"]["accuracy_term"])
for s, (rc, pr) in {101: ((5.1, 3.1, 7.3), (76.5, 71.4)), 202: ((-7.0, -10.0, -4.2), (68.6, 75.7))}.items():
    a8, b8 = arr(f"test/R-s{s}", T, "c8k") / 32, arr(f"test/U-s{s}", T, "c8k") / 32
    add(f"4.3.w8k.cells.s{s}", "4.3", f"seed {s} within-8192 R, U", pr, [float(a8.mean()), float(b8.mean())])
    addci(f"4.3.w8k.s{s}", "4.3", f"seed {s} within-8192 R − U", rc, boot_diff(a8 - b8), production=prodT["within_8192"][f"s{s}"]["R-U_within_8192"])
tt = {s: (arr(f"test/R-s{s}", T, "t") - arr(f"test/U-s{s}", T, "t")) / 32 for s in (101, 202)}
add("4.3.pearson", "4.3", "per-problem Pearson(R − U truncation, R − U success) −0.51 (s101), −0.59 (s202)", [-0.51, -0.59],
    [float(np.corrcoef(tt[101], d101)[0, 1]), float(np.corrcoef(tt[202], d202)[0, 1])], digits=2, unit="raw")
add("4.3.r202trunc", "4.3", "R-s202 has 538 truncated answers", 538, scan["evals"]["test/R-s202"]["records"]["truncated_n"], digits=0, unit="raw")

# ------------------------------------------------------------------ 5 length extension (sealed test and stress)
L = scan["length"]
lt_rep = {"R-s101": (81.9, 84.2, 437, 94, 21.5, 50, 293, 67.0, 15689), "U-s101": (77.7, 80.6, 501, 118, 23.6, 67, 316, 63.1, 13845),
          "R-s202": (74.8, 76.9, 538, 87, 16.2, 104, 347, 64.5, 20476), "U-s202": (79.2, 80.8, 333, 66, 19.8, 65, 202, 60.7, 16674),
          "H-s101": (77.0, 79.2, 445, 93, 20.9, 141, 211, 47.4, 14002)}
prodLT = json.load(open(f"{EV}/a19/length/test/length.json"))
for k, rep in lt_rep.items():
    s = L[f"test/{k}"]["sets"]["all"]
    rec = [round(100 * s["pass1_1"], 1), round(100 * s["pass1_2"], 1), s["capped_1"], s["recovered"], round(100 * s["recovered"] / s["capped_1"], 1),
           s["finished_wrong"], s["still_capped"], round(100 * s["still_capped"] / s["capped_1"], 1), round(s["new_tokens"] / s["recovered"])]
    claims.append({"id": f"5.2.{k}", "section": "5.2", "claim": f"{k} sealed test 12,288 → 16,384 row", "report": list(rep),
                   "recomputed_display": rec, "recomputed_unrounded": s, "production": prodLT["sets"]["all"]["cells"]["all"].get(f"{k}-r128"),
                   "status": "verified" if rec == list(rep) else "differs",
                   "note": "'Extra tokens per recovered answer' = all new tokens spent on extended attempts / recovered attempts "
                           f"(recovered attempts themselves used {s['new_tokens_recovered_only'] / s['recovered']:.0f} new tokens each on average)."})
ls_rep = {"R-s101": (69.5, 72.3, 362, 49, 13.5, 61, 252, 69.6, 25435), "U-s101": (65.3, 68.2, 406, 51, 12.6, 64, 291, 71.7, 28075),
          "R-s202": (63.2, 66.0, 407, 50, 12.3, 96, 261, 64.1, 27437), "U-s202": (65.8, 68.2, 268, 41, 15.3, 81, 146, 54.5, 19914),
          "H-s101": (64.1, 67.3, 313, 55, 17.6, 126, 132, 42.2, 15933)}
for k, rep in ls_rep.items():
    s = L[f"stress/{k}"]["sets"]["all"]
    rec = [round(100 * s["pass1_1"], 1), round(100 * s["pass1_2"], 1), s["capped_1"], s["recovered"], round(100 * s["recovered"] / s["capped_1"], 1),
           s["finished_wrong"], s["still_capped"], round(100 * s["still_capped"] / s["capped_1"], 1), round(s["new_tokens"] / s["recovered"])]
    claims.append({"id": f"5.3.{k}", "section": "5.3", "claim": f"{k} stress 12,288 → 16,384 row", "report": list(rep), "recomputed_display": rec,
                   "recomputed_unrounded": s, "production": None, "status": "verified" if rec == list(rep) else "differs", "note": ""})
t1_rep = {"R-s101": (71.5, 75.8, 54, 11, 5, 38, 16805), "U-s101": (67.6, 69.9, 47, 6, 10, 31, 27348),
          "R-s202": (67.6, 71.1, 56, 9, 13, 34, 20331), "U-s202": (69.5, 73.0, 41, 9, 11, 21, 12910)}
for k, rep in t1_rep.items():
    s = L[f"stress/{k}"]["sets"]["t1"]
    rec = [round(100 * s["pass1_1"], 1), round(100 * s["pass1_2"], 1), s["capped_1"], s["recovered"], s["finished_wrong"], s["still_capped"],
           round(s["new_tokens"] / s["recovered"])]
    claims.append({"id": f"5.4.{k}", "section": "5.4", "claim": f"{k} tranche 1 row", "report": list(rep), "recomputed_display": rec,
                   "recomputed_unrounded": s, "production": None, "status": "verified" if rec == list(rep) else "differs", "note": ""})


def lpp(kind, ck, setname, which):
    pp = L[f"{kind}/{ck}"]["per_problem"][setname]
    return pp, which


def lext_contrast(kind, a, b, setname, ids=None):
    pa, pb = L[f"{kind}/{a}"]["per_problem"][setname], L[f"{kind}/{b}"]["per_problem"][setname]
    ids = sorted(pa) if ids is None else [i for i in sorted(pa) if i in ids]
    A1 = np.array([pa[i][1] / pa[i][0] for i in ids]); A2 = np.array([pa[i][2] / pa[i][0] for i in ids])
    B1 = np.array([pb[i][1] / pb[i][0] for i in ids]); B2 = np.array([pb[i][2] / pb[i][0] for i in ids])
    return boot_diff(A1 - B1), boot_diff(A2 - B2), boot_diff((A2 - B2) - (A1 - B1)), (A1.mean(), A2.mean(), B1.mean(), B2.mean())


lrep = {("test", "R-s101", "U-s101", "all", None): ((4.2, 2.1, 6.2), (3.6, 1.5, 5.7), (-0.6, -1.6, 0.4)),
        ("test", "R-s202", "U-s202", "all", None): ((-4.4, -7.3, -1.8), (-3.9, -6.7, -1.3), (0.5, -0.3, 1.3)),
        ("test", "R-s101", "H-s101", "all", None): ((4.9, 2.0, 8.0), (4.9, 2.0, 8.0), (0.0, -0.8, 0.9)),
        ("test", "U-s101", "H-s101", "all", None): ((0.7, -2.1, 3.6), (1.3, -1.2, 3.9), (0.6, -0.2, 1.5)),
        ("stress", "R-s101", "U-s101", "all", None): ((4.2, 1.6, 6.8), (4.1, 1.4, 6.7), (-0.1, -1.1, 1.0)),
        ("stress", "R-s101", "U-s101", "all", "harder"): ((4.1, 0.5, 7.8), (4.2, 0.7, 7.8), (0.1, -1.1, 1.4)),
        ("stress", "R-s101", "U-s101", "all", "superhard"): ((4.3, 0.6, 8.0), (4.0, 0.2, 7.8), (-0.3, -1.8, 1.3)),
        ("stress", "R-s202", "U-s202", "all", None): ((-2.7, -5.1, -0.2), (-2.1, -4.6, 0.3), (0.5, -0.5, 1.5)),
        ("stress", "R-s202", "U-s202", "all", "harder"): ((-3.2, -6.6, 0.3), (-1.6, -4.6, 1.4), (1.6, 0.1, 3.2)),
        ("stress", "R-s202", "U-s202", "all", "superhard"): ((-2.3, -5.6, 1.1), (-2.6, -6.2, 1.0), (-0.3, -1.7, 1.1)),
        ("stress", "R-s101", "H-s101", "all", None): ((5.4, 2.5, 8.3), (5.0, 2.3, 7.8), (-0.3, -1.5, 0.8)),
        ("stress", "U-s101", "H-s101", "all", None): ((1.2, -1.5, 3.9), (1.0, -1.8, 3.7), (-0.2, -1.4, 0.9)),
        ("stress", "R-s101", "U-s101", "t1", None): ((3.9, -2.0, 10.2), (5.9, -0.4, 12.5), (2.0, -1.2, 5.1)),
        ("stress", "R-s202", "U-s202", "t1", None): ((-2.0, -9.0, 4.3), (-2.0, -8.6, 4.3), (0.0, -3.1, 3.1))}
for (kind, a, b, st, tier), reps in lrep.items():
    ids = None if tier is None else {i for i in STRESS_IDS if TIER[i] == tier}
    c1, c2, cd, means = lext_contrast(kind, a, b, st, ids)
    tag = f"{kind}/{st}/{tier or 'all'}/{a}-{b}"
    R[f"length.{tag}"] = {"at12288": c1, "at16384": c2, "change": cd, "means": [float(x) for x in means]}
    for nm, ci, rep in (("at 12,288", c1, reps[0]), ("at 16,384", c2, reps[1]), ("change", cd, reps[2])):
        addci(f"5.{tag}.{nm}", "5, Q2, Q4", f"{tag} {nm}", rep, ci)
# ranges stated in section 5 prose
g_t = {k: L[f"test/{k}"]["sets"]["all"]["pass1_2"] - L[f"test/{k}"]["sets"]["all"]["pass1_1"] for k in lt_rep}
g_s = {k: L[f"stress/{k}"]["sets"]["all"]["pass1_2"] - L[f"stress/{k}"]["sets"]["all"]["pass1_1"] for k in ls_rep}
add("5.2.gainrange", "5.2", "4,096 more tokens add 1.6–2.9 points to every checkpoint (sealed test)", [1.6, 2.9], [min(g_t.values()), max(g_t.values())])
add("5.2.recrange", "5.2", "recover 16–24% of capped answers (sealed test)", [16, 24],
    [min(L[f"test/{k}"]["sets"]["all"]["recovered"] / L[f"test/{k}"]["sets"]["all"]["capped_1"] for k in lt_rep),
     max(L[f"test/{k}"]["sets"]["all"]["recovered"] / L[f"test/{k}"]["sets"]["all"]["capped_1"] for k in lt_rep)], digits=0)
add("5.3.gainrange", "5.3", "every checkpoint gains 2.4–2.9 points (stress)", [2.4, 2.9],
    [min(g_s.values()), max(g_s.values())], note=f"unrounded gains: { {k: round(100 * v, 2) for k, v in g_s.items()} }; "
    "U-s202 gains 2.32 and H-s101 3.11. The stated range uses differences of rounded scores.")
add("5.3.H", "5.3", "H-s101 stress gains 3.2 points (64.1 → 67.3)", 3.2, g_s["H-s101"], note="difference of rounded values is 3.2; unrounded 3.11")
sc = {k: L[f"stress/{k}"]["sets"]["all"] for k in ("R-s101", "U-s101", "R-s202", "U-s202")}
add("5.3.recrange", "5.3", "only 12–15% of capped answers recover; 55–72% still capped (R/U stress)", [12, 15, 55, 72],
    [min(v["recovered"] / v["capped_1"] for v in sc.values()), max(v["recovered"] / v["capped_1"] for v in sc.values()),
     min(v["still_capped"] / v["capped_1"] for v in sc.values()), max(v["still_capped"] / v["capped_1"] for v in sc.values())], digits=0)

# ------------------------------------------------------------------ 7 stress panels
S = STRESS_IDS
st_rep = {"base": (37.3, 25.8, 52.5, 0.1, 37.3, 152), "H-s101-r128": (64.1, 52.6, 79.5, 17.7, 77.9, 183), "R-s101-r128": (69.5, 58.4, 84.2, 20.5, 87.4, 184),
          "U-s101-r128": (65.3, 54.2, 80.1, 23.0, 84.8, 184), "H-s202-r128": (60.1, 49.4, 74.3, 25.5, 80.7, 177), "R-s202-r128": (63.2, 53.3, 76.3, 23.0, 82.1, 180),
          "U-s202-r128": (65.8, 55.6, 79.5, 15.2, 77.6, 188), "R-s101-r64": (64.3, 53.5, 78.6, 26.1, 87.0, 183), "H-s202-r64": (62.4, 52.1, 76.1, 15.1, 73.5, 183)}
for k, rep in st_rep.items():
    e = scan["evals"][f"stress/{k}"]
    a, sh, hd = e["records"], e["tiers"]["superhard"], e["tiers"]["harder"]
    rec = [round(100 * a["pass1"], 1), round(100 * sh["pass1"], 1), round(100 * hd["pass1"], 1), round(100 * a["trunc"], 1), round(100 * a["acc_finished"], 1), a["solved_any"]]
    claims.append({"id": f"7.2.{k}", "section": "7.2", "claim": f"{k}: all pass@1, SuperHard, Harder, truncated, acc finished, solved ≥ 1 of 8",
                   "report": list(rep), "recomputed_display": rec, "recomputed_unrounded": [a["pass1"], sh["pass1"], hd["pass1"], a["trunc"], a["acc_finished"], a["solved_any"]],
                   "production": None, "status": "verified" if rec == list(rep) else "differs", "note": ""})
tier_rep = {("base", "harder"): (52.5, 399, 760, 78, 17, 0.1, 52.6), ("H-s101-r128", "harder"): (79.5, 604, 760, 88, 7, 9.1, 87.4),
            ("R-s101-r128", "harder"): (84.2, 640, 760, 90, 5, 9.9, 93.4), ("U-s101-r128", "harder"): (80.1, 609, 760, 90, 5, 11.4, 90.5),
            ("R-s202-r128", "harder"): (76.3, 580, 760, 89, 6, 13.4, 88.1), ("U-s202-r128", "harder"): (79.5, 604, 760, 91, 4, 9.2, 87.5),
            ("H-s202-r128", "harder"): (74.3, 565, 760, 88, 7, 15.1, 87.6), ("R-s101-r64", "harder"): (78.6, 597, 760, 89, 6, 14.6, 92.0),
            ("H-s202-r64", "harder"): (76.1, 578, 760, 89, 6, 8.7, 83.3), ("base", "superhard"): (25.8, 260, 1008, 74, 52, 0.0, 25.8),
            ("H-s101-r128", "superhard"): (52.6, 530, 1008, 95, 31, 24.2, 69.4), ("R-s101-r128", "superhard"): (58.4, 589, 1008, 94, 32, 28.5, 81.7),
            ("U-s101-r128", "superhard"): (54.2, 546, 1008, 94, 32, 31.6, 79.2), ("R-s202-r128", "superhard"): (53.3, 537, 1008, 91, 35, 30.3, 76.4),
            ("U-s202-r128", "superhard"): (55.6, 560, 1008, 97, 29, 19.6, 69.1), ("H-s202-r128", "superhard"): (49.4, 498, 1008, 89, 37, 33.3, 74.1),
            ("R-s101-r64", "superhard"): (53.5, 539, 1008, 94, 32, 34.8, 82.0), ("H-s202-r64", "superhard"): (52.1, 525, 1008, 94, 32, 19.9, 65.1)}
for (k, t), rep in tier_rep.items():
    x = scan["evals"][f"stress/{k}"]["tiers"][t]
    rec = [round(100 * x["pass1"], 1), x["correct"], x["attempts"], x["solved_any"], x["zero"], round(100 * x["trunc"], 1), round(100 * x["acc_finished"], 1)]
    claims.append({"id": f"7.2.tier.{k}.{t}", "section": "7.2, Q3", "claim": f"{k} {t}: pass@1, correct/attempts, solved ≥ 1 of 8, 0/8, truncated, acc finished",
                   "report": list(rep), "recomputed_display": rec, "recomputed_unrounded": x, "production": None,
                   "status": "verified" if rec == list(rep) else "differs", "note": ""})
prodS = json.load(open(f"{EV}/a17/analysis/stress/stress.json"))
ct_rep = {("R-s101-r128", "U-s101-r128"): ((4.2, 1.5, 6.8), (20.5, 23.0, 2.1, 0.4, 3.9), (87.4, 84.8, 2.0, -0.1, 4.1), (83.3, 83.3)),
          ("R-s202-r128", "U-s202-r128"): ((-2.7, -5.1, -0.2), (23.0, 15.2, -6.3, -8.2, -4.3), (82.1, 77.6, 3.6, 1.2, 6.1), (81.4, 85.1)),
          ("R-s101-r128", "H-s101-r128"): ((5.4, 2.5, 8.2), (20.5, 17.7, -2.3, -4.6, 0.0), (87.4, 77.9, 7.7, 5.2, 10.1), (83.3, 82.8)),
          ("U-s101-r128", "H-s101-r128"): ((1.2, -1.4, 3.8), (23.0, 17.7, -4.3, -6.3, -2.2), (84.8, 77.9, 5.5, 2.9, 8.0), (83.3, 82.8)),
          ("R-s202-r128", "H-s202-r128"): ((3.1, -0.1, 6.3), (23.0, 25.5, 2.0, -0.3, 4.4), (82.1, 80.7, 1.0, -1.8, 4.0), (81.4, 80.1)),
          ("U-s202-r128", "H-s202-r128"): ((5.7, 2.5, 9.2), (15.2, 25.5, 8.2, 6.0, 10.4), (77.6, 80.7, -2.5, -5.6, 0.8), (85.1, 80.1)),
          ("R-s101-r128", "R-s101-r64"): ((5.3, 3.0, 7.6), (20.5, 26.1, 4.9, 3.0, 6.9), (87.4, 87.0, 0.3, -1.4, 2.0), (83.3, 82.8))}
for (a, b), (p, t, ac, p8) in ct_rep.items():
    D = decomp(f"stress/{a}", f"stress/{b}", S)
    ci = contrast(f"stress/{a}", f"stress/{b}", S)
    pr = prodS["contrasts"]["all"].get(f"{a} vs {b}")
    addci(f"7.2.c.{a}-{b}", "7.2, Q3, opening 5", f"stress all-221 {a} − {b}", p, ci, production=pr and pr["d_pass1"])
    add(f"7.2.c.{a}-{b}.t", "7.2", f"{a} vs {b} truncation shares", t[:2], [D["tA"], D["tB"]])
    addci(f"7.2.c.{a}-{b}.tt", "7.2", f"{a} vs {b} truncation term", t[2:], D["trunc_term"], production=pr and pr["trunc_term"])
    add(f"7.2.c.{a}-{b}.a", "7.2", f"{a} vs {b} accuracy among finished", ac[:2], [D["aA"], D["aB"]])
    addci(f"7.2.c.{a}-{b}.at", "7.2", f"{a} vs {b} accuracy term", ac[2:], D["acc_term"], production=pr and pr["acc_term"])
    add(f"7.2.c.{a}-{b}.p8", "7.2", f"{a}, {b} pass@8 (n = 8: observed coverage at 8)", p8, [pass8(f"stress/{a}", S), pass8(f"stress/{b}", S)])
for s, (h, sh) in {101: ((4.1, 0.7, 7.8), (4.3, 0.5, 8.0)), 202: ((-3.2, -6.7, 0.3), (-2.3, -5.6, 1.0))}.items():
    for t, rc in (("harder", h), ("superhard", sh)):
        ids = [i for i in S if TIER[i] == t]
        pr = prodS["contrasts"][t].get(f"R-s{s}-r128 vs U-s{s}-r128")
        addci(f"Q3.tier.{t}.s{s}", "Q3", f"stress {t} R − U seed {s}", rc, contrast(f"stress/R-s{s}-r128", f"stress/U-s{s}-r128", ids), production=pr and pr["d_pass1"])
# stress ranges (A26 §4 item 1)
tr_all = {k: scan["evals"][f"stress/{k}"]["records"]["pass1"] for k in st_rep if k != "base"}
tr_sh = {k: scan["evals"][f"stress/{k}"]["tiers"]["superhard"]["pass1"] for k in st_rep if k != "base"}
add("open5.range", "opening 5", "'trained checkpoints 62.4–69.5%' on 221 stress problems (all 8 trained cells: 60.1–69.5)", [62.4, 69.5],
    [min(tr_all.values()), max(tr_all.values())], note="Minimum is H-s202 r128 60.1%; 62.4% is H-s202 r64.")
add("open5.sh", "opening 5, 7.2", "'SuperHard 25.8% → 52.1–58.4%' (all 8 trained cells: 49.4–58.4)", [52.1, 58.4],
    [min(tr_sh.values()), max(tr_sh.values())], note="Minimum is H-s202 r128 49.4%.")
base_all = scan["evals"]["stress/base"]["records"]["pass1"]
add("7.2.gain", "7.2", "'every trained checkpoint gains 25–32 points over base' (8 trained cells: 22.8–32.2)", [25, 32],
    [min(tr_all.values()) - base_all, max(tr_all.values()) - base_all], digits=0, note="H-s202 r128 gains 22.8 points.")
cov = prodS["coverage"]["all"]
gained = {k: v["gained"] for k, v in cov.items()}
lost = {k: v["lost"] for k, v in cov.items()}
# recompute gained/lost from records
bs = {i: PP["stress/base"][i]["c"] > 0 for i in S}
g2 = {k: sum(1 for i in S if not bs[i] and PP[f"stress/{k}"][i]["c"] > 0) for k in tr_all}
l2 = {k: sum(1 for i in S if bs[i] and PP[f"stress/{k}"][i]["c"] == 0) for k in tr_all}
add("7.2.gainedlost", "7.2", "each checkpoint solves 33–38 problems base never solved; loses 2–5 base solved", [33, 38, 2, 5],
    [min(g2.values()), max(g2.values()), min(l2.values()), max(l2.values())], digits=0, unit="raw",
    production={"gained": gained, "lost": lost}, note=f"lost per cell {l2}; H-s202 r128 lost 8.")
p8 = {k: pass8(f"stress/{k}", S) for k in tr_all}
add("7.2.p8gap", "7.2", "pass@8 differs by at most 3.7 points between arms (same seed, r128)", 3.7,
    max(abs(p8[f"R-s{s}-r128"] - p8[f"U-s{s}-r128"]) for s in (101, 202)),
    note=f"R vs U only; with H: max |Δ| = {100 * max(abs(p8[f'{x}-s{s}-r128'] - p8[f'{y}-s{s}-r128']) for s in (101, 202) for x in 'HRU' for y in 'HRU'):.1f} (U − H seed 202).")
add("7.2.h202drop", "7.2", "H-s202 fell 2.3 points from r64 to r128 (62.4 → 60.1)", 2.3, tr_all["H-s202-r64"] - tr_all["H-s202-r128"])

# ------------------------------------------------------------------ 8 cue study
cue = {k: scan["evals"][f"cue/{k}"]["records"]["pass1"] for k in [f"{p}-{f}" for p in ("base", "R-s101", "R-s202", "U-s101", "U-s202") for f in ("uncued", "cued")]}
add("8.cells", "8", "R-s202 62.0 uncued / 57.3 cued; U-s202 64.6 / 69.3", [62.0, 57.3, 64.6, 69.3],
    [cue["R-s202-uncued"], cue["R-s202-cued"], cue["U-s202-uncued"], cue["U-s202-cued"]])
cprod = json.load(open(f"{EV}/a15/cue_analysis.json"))
add("8.gain", "8", "cue gain R-s202 −4.7, U-s202 +4.7; difference in cue gain +2.1 (s101), −9.4 (s202)", [-4.7, 4.7, 2.1, -9.4],
    [cue["R-s202-cued"] - cue["R-s202-uncued"], cue["U-s202-cued"] - cue["U-s202-uncued"],
     (cue["R-s101-cued"] - cue["R-s101-uncued"]) - (cue["U-s101-cued"] - cue["U-s101-uncued"]),
     (cue["R-s202-cued"] - cue["R-s202-uncued"]) - (cue["U-s202-cued"] - cue["U-s202-uncued"])],
    production={"cue_gain": cprod["cue_gain"], "did": cprod["did"]}, note="Intervals [−9.9, 0.0], [−0.5, +10.9], [−6.3, +12.0], [−19.3, −1.6] match the production cue_analysis.json (not re-bootstrapped).")
mf = {p: cprod["policies"][p]["cued"]["mentions_factor"] - cprod["policies"][p]["uncued"]["mentions_factor"] for p in cprod["policies"]}
add("8.factor", "8", "cue added +9 to +15 points more mentions of 'factor' (production output; text not re-decoded)", [9, 15],
    [min(mf.values()), max(mf.values())], digits=0, status=None, note="checked against production cue_analysis.json only")
claims[-1]["status"] = "claimed (production output only)" if claims[-1]["status"] == "verified" else claims[-1]["status"]

# ------------------------------------------------------------------ 6 local diagnostic
def jl(p):
    return [json.loads(x) for x in open(p)]


diag = {"r0": jl(f"{EV}/a15/diag_r0.jsonl"), "r64": jl(f"{EV}/a15/diag_r64.jsonl")}
ref = {"r0": jl(f"{EV}/a15/diag_r0_ref.jsonl"), "r64": jl(f"{EV}/a15/diag_r64_ref.jsonl")}
for rnd, rep in {"r0": (40.0, 43.2, 41.5, 42.7, 40.5, -2.2, -0.9), "r64": (60.6, 63.1, 60.6, 61.6, 60.9, -0.7, 0.2)}.items():
    rows = diag[rnd]
    jh = lambda r: r["eval"]["probe_J_H"]["pass1"]  # noqa: E731
    none = [jh(r) for r in rows if r["branch"] == "none"]
    refv = [jh(r) for r in ref[rnd] if r.get("branch") == "none"] or [jh(r) for r in ref[rnd]]
    m = {b: float(np.mean([jh(r) for r in rows if r["branch"] == b])) for b in ("hard", "related", "unrelated")}
    rec = [none[0], refv[0], m["hard"], m["related"], m["unrelated"], m["unrelated"] - m["related"], m["unrelated"] - m["hard"]]
    add(f"6.{rnd}", "6", f"{rnd}: no-update J_H original / rerun, hard, related, unrelated means, unrelated − related, unrelated − hard", list(rep), rec,
        note=f"draws per branch: { {b: sum(r['branch'] == b for r in rows) for b in ('hard', 'related', 'unrelated')} }; n per problem 16, cap 8,192 (DEFINITIONS §4 item 1)")
    R[f"diag.{rnd}"] = {"none": none, "ref": refv, "means": m,
                        "draws": {b: [(r["seed"], jh(r), r["eval"]["probe_J_R"]["pass1"], r["eval"]["probe_J_U"]["pass1"]) for r in rows if r["branch"] == b] for b in ("hard", "related", "unrelated")}}
zg = [d for b in ("related", "unrelated") for d in R["diag.r64"]["draws"][b] if (b, d[0]) in (("related", 4101), ("related", 4102), ("unrelated", 4103))]
add("6.zero", "6", "three zero-gradient r64 draws (related 4101, 4102; unrelated 4103) give J_H 61.1, J_R 89.5, J_U 93.2 each",
    [61.1, 89.5, 93.2] * 3, [x for d in zg for x in d[1:]])
ur = [d[1] for d in R["diag.r64"]["draws"]["unrelated"]]
add("6.spread", "6", "unrelated at r64 single draws 59.2%–63.3%", [59.2, 63.3], [min(ur), max(ur)])

# ------------------------------------------------------------------ dev and probes (3.1, 3.2, Q3) from canonical metrics rows
DV = TR["dev_probe_canonical"]
dev_rep = {(101, 0): (41.7, 42.2, -0.5), (101, 32): (54.2, 60.1, -5.9), (101, 64): (73.6, 71.0, 2.6), (101, 96): (74.8, 68.4, 6.4),
           (101, 128): (72.9, 69.1, 3.8), (202, 0): (40.3, 42.2, -1.9), (202, 32): (59.9, 61.1, -1.2), (202, 64): (61.3, 67.9, -6.6),
           (202, 96): (64.2, 67.9, -3.8), (202, 128): (65.6, 70.3, -4.7)}   # report table 3.2, R − U column as printed
for (s, r), rep in dev_rep.items():
    a, b = DV[f"R-s{s}"][f"dev@{r}"]["pass1"], DV[f"U-s{s}"][f"dev@{r}"]["pass1"]
    add(f"3.2.dev.s{s}.r{r}", "3.2, Q3", f"dev seed {s} round {r}: R, U, R − U", list(rep), [a, b, a - b],
        note="table cell R − U disagrees with the difference of its displayed scores" if abs(round(rep[0] - rep[1], 1) - rep[2]) > 1e-9 else "")
h = DV["H-s202"]
add("3.1.h202dev", "3.1", "H-s202 (a2-r60) dev at rounds 64/96/128: 63.7/66.0/67.0", [63.7, 66.0, 67.0],
    [h["dev@64"]["pass1"], h["dev@96"]["pass1"], h["dev@128"]["pass1"]])
order = {}
for s in (101, 202):
    for pan in ("dev", "probe_J_H", "probe_J_R", "probe_J_U"):
        seq = [(r, DV[f"R-s{s}"][f"{pan}@{r}"]["pass1"] - DV[f"U-s{s}"][f"{pan}@{r}"]["pass1"]) for r in (0, 32, 64, 96, 128)]
        end = np.sign(seq[-1][1])
        first = next((r for r, d in seq if np.sign(d) == end), None)
        persist = next((seq[i][0] for i in range(len(seq)) if all(np.sign(d) == end for _, d in seq[i:])), None)
        order[f"s{s}.{pan}"] = {"R_minus_U_by_round": seq, "endpoint_sign": int(end), "earliest_round_with_endpoint_sign": first,
                                "earliest_round_from_which_sign_holds": persist,
                                "n": DV[f"R-s{s}"][f"{pan}@128"]["n"], "problems": DV[f"R-s{s}"][f"{pan}@128"]["problems"],
                                "cap_by_round": [DV[f"R-s{s}"][f"{pan}@{r}"]["max_tokens"] for r in (0, 32, 64, 96, 128)]}
R["ordering"] = order
# probes: base ≈ 81% and gains 8–15 points (all six runs)
pr0 = [DV[f"{a}-s{s}"][f"{p}@0"]["pass1"] for a in "HRU" for s in (101, 202) for p in ("probe_J_R", "probe_J_U")]
prg = [DV[f"{a}-s{s}"][f"{p}@128"]["pass1"] - DV[f"{a}-s{s}"][f"{p}@0"]["pass1"] for a in "HRU" for s in (101, 202) for p in ("probe_J_R", "probe_J_U")]
add("3.1.probes", "3.1", "base solves about 81% of J_R and J_U; every arm gains 8–15 points on both (six runs, r0 → r128)", [81, 8, 15],
    [float(np.mean(pr0)), min(prg), max(prg)], digits=0, note=f"r0 values {sorted(round(100 * x, 1) for x in pr0)}; gains {sorted(round(100 * x, 1) for x in prg)}")
d128 = [DV[f"{a}-s{s}"]["dev@128"]["pass1"] for a in "HRU" for s in (101, 202)]
d0 = [DV[f"{a}-s{s}"]["dev@0"]["pass1"] for a in "HRU" for s in (101, 202)]
add("6.link3", "6", "dev pass@1 rose from 40–42% to 66–73% (six runs)", [40, 42, 66, 73], [min(d0), max(d0), min(d128), max(d128)], digits=0)

# ------------------------------------------------------------------ response audit denominators (7.3)
au = json.load(open(f"{EV}/a17/audit/audit_merged.json"))
resp = list(au.values())
fails = [r for r in resp if r["reward"] == 0]
capped_f = [r for r in fails if r["finish"] == "length"]
pols_a = Counter(r["policy"] for r in resp)
reads = sum(len(r["reads"]) for r in resp)
fail_reads = [rd for r in fails for rd in r["reads"]]
fx = Counter(rd["practice_fixable"] for rd in fail_reads)
grader = [r for r in fails if all(rd["category"] in ("grader", "grader_error", "grader_issue") for rd in r["reads"])]
gen = [r for r in fails if r not in grader]
pair = Counter(tuple(sorted(rd["practice_fixable"] for rd in r["reads"])) for r in gen)
add("7.3.denoms", "7.3, opening 5", "audit: 96 responses, 192 reads, 35 failures, 13 capped failures, 33 genuine, 6 policies",
    [96, 192, 35, 13, 33, 6], [len(resp), reads, len(fails), len(capped_f), len(gen), len(pols_a)], digits=0, unit="raw",
    note=f"policies {dict(pols_a)}; grader-category failures {[(r['item'], r['policy']) for r in grader]}")
add("7.3.reads", "7.3", "failure reads 70 = yes 29 / no 27 / unclear 10 / n.a. 4", [70, 29, 27, 10, 4],
    [len(fail_reads), fx.get("yes", 0), fx.get("no", 0), fx.get("unclear", 0), fx.get("na", 0)], digits=0, unit="raw")
add("7.3.pairs", "7.3, opening 5", "of 33 genuine: both yes 13, both no 10; 29 of 66 judged reads", [13, 10, 29, 66],
    [pair.get(("yes", "yes"), 0), pair.get(("no", "no"), 0), fx.get("yes", 0), len(fail_reads) - fx.get("na", 0)], digits=0, unit="raw",
    note=f"pairs {dict((','.join(k), v) for k, v in pair.items())}")
R["audit"] = {"responses": len(resp), "reads": reads, "failures": len(fails), "capped_failures": len(capped_f), "genuine": len(gen), "pairs": {",".join(k): v for k, v in pair.items()}}

# ------------------------------------------------------------------ displayed-equation check (A26 §4 last item)
txt = open(REPORT).read().splitlines()
pat = re.compile(r"(?<![\d.])(\d+\.\d)\s*−\s*(\d+\.\d)\s*=\s*([+−-]?)(\d+\.\d)")
eqs = []
for ln, line in enumerate(txt, 1):
    for m in pat.finditer(line):
        a, b, sg, cval = float(m.group(1)), float(m.group(2)), m.group(3), float(m.group(4))
        cv = -cval if sg in ("−", "-") else cval
        disp = round(a - b, 1)
        eqs.append({"line": ln, "text": m.group(0), "a": a, "b": b, "c": cv, "a_minus_b_displayed": disp, "fails": abs(disp - cv) > 1e-9})
R["equations"] = {"total": len(eqs), "failing": [e for e in eqs if e["fails"]]}

# ------------------------------------------------------------------ write
status = Counter(c["status"] for c in claims)
json.dump({"claims": claims, "recomputed": R, "status_counts": status, "bootstrap": {"draws": DRAWS, "rng": "numpy default_rng(20261111)",
           "unit": "problem, paired; per-seed intervals; two-seed means resample problems of the seed-mean difference"}},
          open(OUT, "w"), indent=1, default=float)
print("status", dict(status))
for c in claims:
    if not c["status"].startswith("verified"):
        print(c["id"], "|", c["claim"], "| report", c["report"], "| recomputed", c["recomputed_display"], "|", c["status"], "|", c["note"][:200])
print("equations total", len(eqs), "failing", len(R["equations"]["failing"]))
for e in R["equations"]["failing"]:
    print("  line", e["line"], e["text"], "-> displayed a−b =", e["a_minus_b_displayed"])
