"""E_related / E_unrelated one-to-one matching, balance repair and A5 technique coverage.

Committed before the authored-tranche screening counts were viewed (D39).

Inputs: eligible related (R) and unrelated (U) items with stage-2 counts (c of 32), prompt tokens,
mean response length, answer type, provenance (natural | authored), and for R the v1.3 primary
technique. H_train primary techniques for the coverage target.

Matching (project.md §5): within provenance only (D32), candidate pairs ranked by
  (1) |c_R - c_U| == 0 first, then same 4-count bin (12-15, 16-19, 20-23, 24), else not allowed;
  (2) same answer type preferred;
  (3) standardized length distance d = |z_prompt_R - z_prompt_U| + |z_len_R - z_len_U|
      (z over the pooled eligible R ∪ U);
  (4) ids (deterministic tie-break).
Greedy selection over the ranked pair list; R items of techniques in deficit (A5) are offered
first when a coverage repair is requested. Target 64 pairs.
Balance: standardized mean differences of c and mean response length between matched R and U;
while either |SMD| > 0.25, drop the pair with the largest contribution, re-check.
Coverage (A5): for each technique with share_H >= 10%, required_R = ceil(0.5 * share_H * |R|).
"""
import argparse
import json
import math
from collections import Counter

import numpy as np

AUX_LO, AUX_HI, TARGET = 12, 24, 64


def bin4(c):
    return min((c - 12) // 4, 3)


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0.0
    return 0.0 if sd == 0 else float((a.mean() - b.mean()) / sd)


def coverage(h_prims, r_prims):
    nH, nR = len(h_prims), len(r_prims)
    ch, cr = Counter(h_prims), Counter(r_prims)
    rows = []
    for t in sorted(ch):
        share = ch[t] / nH
        req = math.ceil(0.5 * share * nR) if share >= 0.10 else 0
        rows.append({"technique": t, "H_count": ch[t], "H_share": round(share, 4), "R_count": cr.get(t, 0),
                     "R_share": round(cr.get(t, 0) / nR, 4) if nR else 0.0, "required_R": req,
                     "deficit": max(0, req - cr.get(t, 0))})
    return rows


def match(R, U, priority=()):
    pool = R + U
    zp = {x["id"]: x for x in pool}
    for key in ("prompt_tokens", "resp_len_mean"):
        v = np.array([x[key] for x in pool], float)
        mu, sd = v.mean(), v.std() or 1.0
        for x in pool:
            x["z_" + key] = (x[key] - mu) / sd
    pairs = []
    for r in R:
        for u in U:
            if r["provenance"] != u["provenance"]:
                continue
            exact = r["c"] == u["c"]
            if not exact and bin4(r["c"]) != bin4(u["c"]):
                continue
            d = abs(r["z_prompt_tokens"] - u["z_prompt_tokens"]) + abs(r["z_resp_len_mean"] - u["z_resp_len_mean"])
            pri = 0 if r.get("primary") in priority else 1
            pairs.append((pri, 0 if exact else 1, 0 if r["answer_type"] == u["answer_type"] else 1, round(d, 6), r["id"], u["id"]))
    pairs.sort()
    used_r, used_u, out = set(), set(), []
    for pri, ex, at, d, ri, ui in pairs:
        if ri in used_r or ui in used_u:
            continue
        used_r.add(ri); used_u.add(ui)
        out.append({"r": ri, "u": ui, "exact": ex == 0, "same_answer_type": at == 0, "dist": d})
        if len(out) >= TARGET:
            break
    return out, zp


def balance(pairs, zp):
    log = []
    while len(pairs) > 2:
        cr = [zp[p["r"]]["c"] for p in pairs]; cu = [zp[p["u"]]["c"] for p in pairs]
        lr = [zp[p["r"]]["resp_len_mean"] for p in pairs]; lu = [zp[p["u"]]["resp_len_mean"] for p in pairs]
        s_c, s_l = smd(cr, cu), smd(lr, lu)
        if abs(s_c) <= 0.25 and abs(s_l) <= 0.25:
            return pairs, {"smd_success": s_c, "smd_resp_len": s_l, "dropped": log}
        worst = max(pairs, key=lambda p: abs(zp[p["r"]]["c"] - zp[p["u"]]["c"]) * (abs(s_c) > 0.25)
                    + abs(zp[p["r"]]["z_resp_len_mean"] - zp[p["u"]]["z_resp_len_mean"]) * (abs(s_l) > 0.25))
        pairs = [p for p in pairs if p is not worst]
        log.append({"r": worst["r"], "u": worst["u"], "smd_success": s_c, "smd_resp_len": s_l})
    return pairs, {"smd_success": None, "smd_resp_len": None, "dropped": log, "failed": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--related", required=True, help="jsonl: id, c, prompt_tokens, resp_len_mean, answer_type, provenance, primary")
    ap.add_argument("--unrelated", required=True)
    ap.add_argument("--h-primaries", required=True, help="json list of H_train primary techniques")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.related)]
    U = [json.loads(l) for l in open(a.unrelated)]
    R = [x for x in R if AUX_LO <= x["c"] <= AUX_HI]
    U = [x for x in U if AUX_LO <= x["c"] <= AUX_HI]
    H = json.load(open(a.h_primaries))
    history = []
    priority = ()
    for attempt in range(3):
        pairs, zp = match(R, U, priority)
        pairs, bal = balance(pairs, zp)
        cov = coverage(H, [zp[p["r"]]["primary"] for p in pairs])
        history.append({"attempt": attempt, "priority": list(priority), "n_pairs": len(pairs), "balance": bal, "coverage": cov})
        deficit = tuple(r["technique"] for r in cov if r["deficit"] > 0)
        if not deficit or set(deficit) <= set(priority):
            break
        priority = tuple(sorted(set(priority) | set(deficit)))
    pr = [zp[p["r"]]["c"] / 32 for p in pairs]
    pu = [zp[p["u"]]["c"] / 32 for p in pairs]
    mixed = lambda ps: float(np.mean([1 - p ** 8 - (1 - p) ** 8 for p in ps])) if ps else None  # noqa: E731
    res = {"pairs": pairs, "n_pairs": len(pairs), "eligible": {"related": len(R), "unrelated": len(U)},
           "by_provenance": dict(Counter(zp[p["r"]]["provenance"] for p in pairs)),
           "exact_count_matches": sum(p["exact"] for p in pairs),
           "mixed_group_prob": {"related": mixed(pr), "unrelated": mixed(pu)},
           "final_balance": history[-1]["balance"], "final_coverage": history[-1]["coverage"], "history": history}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_pairs", "eligible", "by_provenance", "exact_count_matches", "mixed_group_prob")}, indent=1))
    print(json.dumps(res["final_coverage"], indent=1))
    print(json.dumps({k: v for k, v in res["final_balance"].items() if k != "dropped"}))


if __name__ == "__main__":
    main()
