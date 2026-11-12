"""A26 pass A (agent a-evals): compare report v2 training-derived tables (4.4, 4.5) with the production outputs
signal.json and persistence.json. These numbers come from training records, not evaluations; a-evals does not
recompute them from raw records, it only checks that the report transcribes the production output correctly.
Usage: python3 production_compare.py REPO OUT_JSON   (adds key production_transcription_4_4_4_5 to OUT_JSON if it exists)
"""
import json
import sys

REPO, OUT = sys.argv[1], sys.argv[2]
EV = f"{REPO}/research/evidence/a17/analysis/train"
sig = json.load(open(f"{EV}/signal.json"))["runs"]
per = json.load(open(f"{EV}/persistence.json"))["runs"]
RUN = {"R-s101": "R-s101", "U-s101": "U-s101", "R-s202": "R-s202", "U-s202": "U-s202", "H-s101": "H-s101", "H-s202": "H-s202-r60"}
# report v2 section 4.4 first table: shared hard 1-32 -> 97-128 ; second pool 1-32 -> 97-128 (all-wrong / mixed / all-correct)
t44 = {"R-s101": ((27, 220, 9), (9, 119, 128), (0, 180, 76), (0, 25, 231)), "U-s101": ((22, 222, 12), (13, 122, 121), (0, 168, 88), (0, 20, 236)),
       "R-s202": ((27, 214, 15), (11, 130, 115), (0, 180, 76), (0, 44, 212)), "U-s202": ((28, 212, 16), (8, 119, 129), (0, 166, 90), (0, 29, 227)),
       "H-s101": ((32, 216, 8), (5, 111, 140), (31, 212, 13), (8, 102, 146)), "H-s202": ((14, 226, 16), (8, 112, 136), (23, 218, 15), (7, 121, 128))}
# second table: cumulative mixed groups and generated tokens (M) at rounds 32/64/96/128
t44b = {"H-s101": ((428, 800, 1076, 1289), (12.8, 31.6, 48.6, 68.6)), "R-s101": ((400, 658, 817, 961), (10.6, 25.9, 46.1, 65.1)),
        "U-s101": ((390, 625, 788, 930), (9.2, 23.0, 40.9, 60.0)), "R-s202": ((394, 690, 926, 1100), (10.0, 23.2, 37.7, 54.1)),
        "U-s202": ((378, 627, 799, 947), (9.3, 22.3, 37.7, 52.8)), "H-s202": ((444, 796, 1054, 1287), (12.7, 29.6, 51.2, 74.7))}
# section 4.5: PS -> all-wrong k / n ; capped 0/some/all ; next appearance all-wrong/mixed/all-correct/none (rounds 33-128)
t45 = {"R-s101": (29, 765, (12, 13, 4), (18, 10, 0, 1)), "U-s101": (38, 768, (7, 23, 8), (17, 18, 0, 3)), "R-s202": (19, 765, (13, 3, 3), (7, 10, 0, 2)),
       "U-s202": (26, 768, (18, 6, 2), (13, 11, 1, 1)), "H-s101": (13, 768, (11, 2, 0), (8, 4, 0, 1)), "H-s202": (21, 768, (14, 3, 4), (15, 5, 0, 1))}
rows = []


def trip(d):
    return (d["all_wrong"], d["mixed"], d["all_correct"])


for k, (h1, h4, s1, s4) in t44.items():
    w = sig[RUN[k]]["windows"]
    second = "hard_extra" if k.startswith("H") else "aux"
    got = (trip(w["r1-32"]["by_role"]["hard_common"]), trip(w["r97-128"]["by_role"]["hard_common"]),
           trip(w["r1-32"]["by_role"][second]), trip(w["r97-128"]["by_role"][second]))
    rows.append({"table": "4.4 groups", "run": k, "report": [h1, h4, s1, s4], "production": got, "match": got == (h1, h4, s1, s4)})
for k, (mx, tk) in t44b.items():
    w = sig[RUN[k]]["windows"]
    cm, ct, acc = [], [], 0
    for win in ("r1-32", "r33-64", "r65-96", "r97-128"):
        acc += w[win]["all_roles"]["mixed"]
        cm.append(acc)
        ct.append(round(w[win]["cum_gen_tokens_at_window_end"] / 1e6, 1))
    rows.append({"table": "4.4 cumulative", "run": k, "report": [mx, tk], "production": [cm, ct], "match": tuple(cm) == mx and tuple(ct) == tk})
# share of all-correct practice groups after round 32 (report: 62–90% for R, 74–92% for U; 62–92% overall)
share = {}
for k in ("R-s101", "U-s101", "R-s202", "U-s202"):
    w = sig[RUN[k]]["windows"]
    share[k] = [w[x]["by_role"]["aux"]["frac_all_correct"] for x in ("r33-64", "r65-96", "r97-128")]
rows.append({"table": "4.4 all-correct practice share after round 32", "run": "R,U", "report": "R 62–90%, U 74–92%",
             "production": {k: [round(100 * v, 1) for v in vs] for k, vs in share.items()},
             "match": None})
for k, (n_aw, n_ps, cap, nxt) in t45.items():
    ph = per[RUN[k]]["by_phase"]["r33-128"]
    got = (ph["previously_solvable_became_all_wrong"], ph["later_appearances_of_previously_solvable"],
           (ph["ps_all_wrong_by_capped"]["zero"], ph["ps_all_wrong_by_capped"]["some"], ph["ps_all_wrong_by_capped"]["all"]),
           (ph["next_outcome"]["all_wrong"], ph["next_outcome"]["mixed"], ph["next_outcome"]["all_correct"], ph["next_outcome"]["no_later_observation"]))
    rows.append({"table": "4.5", "run": k, "report": [n_aw, n_ps, cap, nxt], "production": got, "match": got == (n_aw, n_ps, cap, nxt)})
import os
base = json.load(open(OUT)) if os.path.exists(OUT) else {}
base["production_transcription_4_4_4_5"] = rows
json.dump(base, open(OUT, "w"), indent=1, default=str)
for r in rows:
    print(r["table"], r["run"], r["match"], "" if r["match"] else (r["report"], r["production"]))
