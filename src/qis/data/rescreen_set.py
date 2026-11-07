"""Item sets for the 8,192-token re-screen (D45).

Rule: an item whose 4,096-cap success already exceeds a band's upper limit is not re-screened
(success at a larger cap is >= success at the smaller cap path-wise). Everything else that could
fall in a band at 8,192 is re-screened with fresh streams.
  H        : audit-eligible native (hard, hard_reserve) and Omni items with 4k c <= 12/32.
  auxiliary: natural + authored (tranche 1, 2) related/unrelated items with 4k stage-1 <= 14/16
             (stage 1 at 8k, shortlist 4-14/16, then stage 2 at 8k). Annotation filters apply later.
  calib    : all 200 + 200 calibration candidates (16 samples at 8k).
Usage: python -m qis.data.rescreen_set PROJ_DATA ANN_DIR OUT_DIR
"""
import json
import os
import sys


def rd(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


def rows(path):
    return [json.loads(l) for l in open(path)]


def main(pd, ann, out):
    os.makedirs(out, exist_ok=True)
    p1 = rd(f"{ann}/pass1.jsonl"); p2 = rd(f"{ann}/pass2_hard.jsonl"); th = rd(f"{ann}/third_read.jsonl")
    tags = rd(f"{ann}/tags_v13.jsonl")
    o1, o2, ot = rd(f"{ann}/omni_pass1.jsonl"), rd(f"{ann}/omni_pass2.jsonl"), rd(f"{ann}/omni_third_read.jsonl")
    summary, H = {}, []
    # native H
    for src in ("hard", "hard_reserve"):
        c4 = rd(f"{pd}/screen/s2/{src}/s2_counts.jsonl")
        for r in rows(f"{pd}/data/candidates-v1-study/{src}.jsonl"):
            i = r["id"]; a, b = p1.get(i), p2.get(i)
            if a is None or b is None:
                continue
            agree = (a["label"], a["direct_practice"]) == (b["label"], b["direct_practice"])
            ok = (agree and b["label"] == "central" and b["direct_practice"] == "no") or \
                 (not agree and th.get(i, {}).get("resolved") and th[i]["label"] == "central" and th[i]["direct_practice"] == "no")
            ok = ok and b["confidence"] != "low" and b["figure"] == "no" and tags.get(i, {}).get("primary") not in (None, "none")
            if ok and i in c4 and c4[i]["c"] <= 12:
                H.append({**{k: r[k] for k in ("id", "problem", "gold")}, "source": src})
    # Omni H
    c4 = rd(f"{pd}/screen/s2/omni_hard/s2_counts.jsonl")
    key = lambda x: (x["label"], x["direct_practice"], x["proof_like"])  # noqa: E731
    for r in rows(f"{pd}/data/omni-v1/omni_hard.jsonl"):
        i = r["id"]; a, b = o1.get(i), o2.get(i)
        fin = b if key(a) == key(b) else (ot[i] if i in ot and ot[i]["resolved"] else None)
        ok = fin is not None and fin["label"] == "central" and fin["direct_practice"] == "no" and fin["proof_like"] == "no" \
            and a["confidence"] != "low" and b["confidence"] != "low" and b["figure"] == "no" and b["primary"] != "none"
        if ok and i in c4 and c4[i]["c"] <= 12:
            H.append({**{k: r[k] for k in ("id", "problem", "gold")}, "source": "omni"})
    with open(f"{out}/h_k8.jsonl", "w") as f:
        for r in H:
            f.write(json.dumps(r) + "\n")
    summary["h_k8"] = len(H)
    # auxiliary
    aux_src = {"related": [("related", "candidates-v1-study", "related"), ("auth_related", "candidates-authored-v1", "auth_related"),
                           ("auth2_related", "candidates-authored-v2", "auth2_related")],
               "unrelated": [("unrelated", "candidates-v1-study", "unrelated"), ("auth_unrelated", "candidates-authored-v1", "auth_unrelated"),
                             ("auth2_unrelated", "candidates-authored-v2", "auth2_unrelated")]}
    for arm, lst in aux_src.items():
        sel = []
        for pool, cdir, s1name in lst:
            s1 = rd(f"{pd}/screen/s1/{s1name}/s1_counts.jsonl")
            for r in rows(f"{pd}/data/{cdir}/{pool}.jsonl"):
                if r["id"] in s1 and s1[r["id"]]["c"] <= 14:
                    sel.append({**{k: r[k] for k in ("id", "problem", "gold")}, "pool": pool})
        with open(f"{out}/{arm}_k8.jsonl", "w") as f:
            for r in sel:
                f.write(json.dumps(r) + "\n")
        summary[f"{arm}_k8"] = len(sel)
    for sp in ("calib_train", "calib_val"):
        summary[sp] = sum(1 for _ in open(f"{pd}/data/candidates-v1-calib/{sp}.jsonl"))
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main(*sys.argv[1:4])
