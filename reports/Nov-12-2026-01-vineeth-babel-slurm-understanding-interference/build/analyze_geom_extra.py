"""Derived gradient-geometry numbers for 01-understanding.md. Input: tables/geom_summary.json and the harvested
summary.json files. Run: python build/analyze_geom_extra.py <harness results dir>. Output: tables/geom_extra.json.

ratio_DE = mean cross-domain inner product / sqrt(within-D x within-E inner products), all over pairs of DIFFERENT
problems. It is the cosine between the expected domain gradients if problems are exchangeable inside a domain.
(The `rho` of summ.py divides by a power that also contains same-problem pairs, so it is biased towards zero.)
Pooled trained-checkpoint numbers weight each checkpoint by its number of pairs.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1]
g = json.load(open(os.path.join(HERE, "tables", "geom_summary.json")))
D = ["hard", "related", "unrelated", "control"]
key = lambda a, b: "|".join(sorted([a, b]))
out = {}
for ck in g:
    r = json.load(open(os.path.join(RES, f"qis-ui-geom-{ck}", "summary.json")))
    P = r["global"]["pairs"]
    e = {"ratio": {}, "cross_mean_cos": None, "within_mean_cos": None}
    for i, a in enumerate(D):
        for b in D[i + 1:]:
            k = key(a, b)
            if k in P and key(a, a) in P and key(b, b) in P:
                den = P[key(a, a)]["mean_ip"] * P[key(b, b)]["mean_ip"]
                e["ratio"][k] = P[k]["mean_ip"] / np.sqrt(den) if den > 0 else None
    cr = [(P[key(a, b)]["n_pairs"], P[key(a, b)]["mean_cos"], P[key(a, b)]["conflict_rate"]) for i, a in enumerate(D) for b in D[i + 1:] if key(a, b) in P]
    wi = [(P[key(a, a)]["n_pairs"], P[key(a, a)]["mean_cos"], P[key(a, a)]["conflict_rate"]) for a in D if key(a, a) in P]
    for nm, x in (("cross", cr), ("within", wi)):
        n = np.array([q[0] for q in x], float)
        e[nm] = {"n_pairs": int(n.sum()), "mean_cos": float((n * [q[1] for q in x]).sum() / n.sum()),
                 "conflict_rate": float((n * [q[2] for q in x]).sum() / n.sum())}
    sp = r["global"]["same_problem"]; n = np.array([sp[a]["n_pairs"] for a in sp], float)
    e["same"] = {"n_pairs": int(n.sum()), "mean_cos": float((n * [sp[a]["mean_cos"] for a in sp]).sum() / n.sum()),
                 "conflict_rate": float((n * [sp[a]["conflict_rate"] for a in sp]).sum() / n.sum())}
    if "scaled" in r:
        S = r["scaled"]["pairs"]; sps = r["scaled"]["same_problem"]
        x = [(S[k]["n_pairs"], S[k]["mean_cos"], S[k]["conflict_rate"]) for k in S if k.split("|")[0] != k.split("|")[1]]
        n = np.array([q[0] for q in x], float)
        e["scaled_cross"] = {"mean_cos": float((n * [q[1] for q in x]).sum() / n.sum()), "conflict_rate": float((n * [q[2] for q in x]).sum() / n.sum())}
        n = np.array([sps[a]["n_pairs"] for a in sps], float)
        e["scaled_same"] = {"mean_cos": float((n * [sps[a]["mean_cos"] for a in sps]).sum() / n.sum())}
    # by type and by depth: same-problem and cross-problem (all different-problem pairs) cosine
    for grp in ("by_type", "by_layer"):
        e[grp] = {}
        for t, x in r[grp].items():
            pp = x["pairs"]; n = np.array([pp[k]["n_pairs"] for k in pp], float)
            s2 = x["same_problem"]; m = np.array([s2[a]["n_pairs"] for a in s2], float)
            e[grp][t] = {"diff_problem": float((n * [pp[k]["mean_cos"] for k in pp]).sum() / n.sum()),
                         "same_problem": float((m * [s2[a]["mean_cos"] for a in s2]).sum() / m.sum())}
    e["norm_share_type"] = r["norm_share_type"]; e["norm_share_layer"] = r.get("norm_share_layer")
    out[ck] = e
tr = [c for c in out if "-r" in c]
pool = {}
for nm in ("cross", "within", "same"):
    n = np.array([out[c][nm]["n_pairs"] for c in tr], float)
    pool[nm] = {"n_pairs": int(n.sum()), "mean_cos": float((n * [out[c][nm]["mean_cos"] for c in tr]).sum() / n.sum()),
                "conflict_rate": float((n * [out[c][nm]["conflict_rate"] for c in tr]).sum() / n.sum())}
out["pooled_trained"] = {"checkpoints": tr, **pool}
json.dump(out, open(os.path.join(HERE, "tables", "geom_extra.json"), "w"), indent=1)
for c in out:
    e = out[c]
    if c == "pooled_trained":
        print(c, {k: (v["n_pairs"], round(v["mean_cos"], 5), round(v["conflict_rate"], 4)) for k, v in pool.items()}); continue
    print(c, "cross", round(e["cross"]["mean_cos"], 5), round(e["cross"]["conflict_rate"], 4), "within", round(e["within"]["mean_cos"], 5),
          "same", e["same"]["n_pairs"], round(e["same"]["mean_cos"], 4), "ratio", {k: (round(v, 2) if v else None) for k, v in e["ratio"].items()},
          "scaled", {k: round(list(v.values())[0], 5) for k, v in e.items() if k.startswith("scaled")})
e = out["base64"]
print("type", {t: (round(v["diff_problem"], 5), round(v["same_problem"], 4), round(e["norm_share_type"][t], 3)) for t, v in e["by_type"].items()})
print("layer", {t: (round(v["diff_problem"] * 1e4, 1), round(v["same_problem"] * 1e2, 1)) for t, v in e["by_layer"].items()})
