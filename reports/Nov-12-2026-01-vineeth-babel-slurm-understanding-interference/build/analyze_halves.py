"""Two derived tables for 01-understanding.md. Run: python build/analyze_halves.py <harness results dir>.
Output: tables/halves.json.
1. First-half and second-half weight updates. tau64 = theta_64 - theta_0, second half = theta_128 - theta_64, for the
   two runs with a round-64 checkpoint (R-s101, H-s202). Cosines come from the exact Gram matrix of qis-ui-wdelta.
2. Cosine between each mixed-group gradient at the base model and each run's total update (count sketch, 2^21 buckets;
   the sketch noise of one cosine is about 2^-10.5 = 6.9e-4).
"""
import gzip, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = sys.argv[1]
w = json.load(gzip.open(os.path.join(RES, "qis-ui-wdelta", "wdelta.json.gz")))
runs = w["runs"]; G = sum(np.array(t["gram"]) for t in w["tensors"]); ix = {r: i for i, r in enumerate(runs)}
def vec(d):
    v = np.zeros(len(runs))
    for k, c in d.items(): v[ix[k]] = c
    return v
ip = lambda a, b: float(a @ G @ b); cos = lambda a, b: ip(a, b) / np.sqrt(ip(a, a) * ip(b, b))
R1 = vec({"R-s101-r64": 1}); R2 = vec({"R-s101-r128": 1, "R-s101-r64": -1})
H1 = vec({"H-s202-r64": 1}); H2 = vec({"H-s202-r128": 1, "H-s202-r64": -1})
out = {"norms": {"R-s101 first": np.sqrt(ip(R1, R1)), "R-s101 second": np.sqrt(ip(R2, R2)), "H-s202 first": np.sqrt(ip(H1, H1)), "H-s202 second": np.sqrt(ip(H2, H2))},
       "cos": {"R first | H first": cos(R1, H1), "R second | H second": cos(R2, H2), "R first | H second": cos(R1, H2),
               "R second | H first": cos(R2, H1), "R first | R second": cos(R1, R2), "H first | H second": cos(H1, H2)},
       "other_runs": {}}
for o in ["H-s101-r128", "U-s101-r128", "R-s202-r128", "U-s202-r128"]:
    v = vec({o: 1}); out["other_runs"][o] = {"R first": cos(v, R1), "R second": cos(v, R2), "H first": cos(v, H1), "H second": cos(v, H2)}
six = [r for r in runs if r.endswith("r128")]; m = vec({r: 1 / 6 for r in six})
out["mean6"] = {"norm": np.sqrt(ip(m, m)), "cos_with_runs": {r: cos(m, vec({r: 1})) for r in six},
                "mean_run_norm": float(np.mean([np.sqrt(ip(vec({r: 1}), vec({r: 1}))) for r in six]))}
z = np.load(os.path.join(RES, "qis-ui-geom-base", "small.npz"))
c = z["cos_g_delta"].astype(float)
out["base_grad_vs_update"] = {"n_groups": int(c.shape[0]), "sd_single_cos": float(c.std()),
                              "mean_cos_per_run": {r: float(c[:, i].mean()) for i, r in enumerate(runs)},
                              "se_of_mean": float(c.std() / np.sqrt(c.shape[0]))}
json.dump(out, open(os.path.join(HERE, "tables", "halves.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:1800])
