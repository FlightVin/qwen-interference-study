"""Build E_related / E_unrelated matching inputs at the 8,192 cap (D45, D39).

Eligibility (final annotation = pass-1/pass-2 agreement, else resolved third read; confidence not low):
  related   : central, direct_practice=yes (natural: pass1 + pass2_aux_natural; authored t1/t2)
  unrelated : absent, modular=yes
Counts: 8k stage-2 (s2k8) for items shortlisted at 8k stage 1. Provenance: natural | authored.
Primary technique (related): natural -> tags_v13 primary; authored -> pass-2 primary.
Usage: python -m qis.data.aux_inputs PROJ_DATA ANN_DIR OUT_DIR
"""
import json
import os
import sys


def rd(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def final(a, b, t, keys=("label", "direct_practice", "modular")):
    if a is None or b is None:
        return None
    if all(a[k] == b[k] for k in keys):
        return b if b.get("confidence") != "low" and a.get("confidence") != "low" else None
    return t if t and t.get("resolved") and t.get("confidence") != "low" else None


def main(pd, ann, out):
    os.makedirs(out, exist_ok=True)
    p1 = rd(f"{ann}/pass1.jsonl"); p2n = rd(f"{ann}/pass2_aux_natural.jsonl"); tags = rd(f"{ann}/tags_v13.jsonl")
    a1p1, a1p2, a1t = rd(f"{ann}/authored_pass1.jsonl"), rd(f"{ann}/authored_pass2.jsonl"), rd(f"{ann}/authored_third_read.jsonl")
    a2p1, a2p2, a2t = rd(f"{ann}/authored2_pass1.jsonl"), rd(f"{ann}/authored2_pass2.jsonl"), rd(f"{ann}/authored2_third_read.jsonl")
    summary = {}
    for arm, rule in (("related", lambda f: f["label"] == "central" and f["direct_practice"] == "yes"),
                      ("unrelated", lambda f: f["label"] == "absent" and f["modular"] == "yes")):
        counts = rd(f"{pd}/screen/s2k8/{arm}_k8/s2k8_counts.jsonl")
        items = [json.loads(l) for l in open(f"{pd}/data/rescreen-k8-aux/{arm}_k8.jsonl")]
        rows, excl = [], {}
        for it in items:
            i = it["id"]
            if i.startswith("syn-R1") or i.startswith("syn-R2") or i.startswith("syn-U1") or i.startswith("syn-U2"):
                f, prov, prim_src = final(a1p1.get(i), a1p2.get(i), a1t.get(i)), "authored", a1p2.get(i)
            elif i.startswith("syn-"):
                f, prov, prim_src = final(a2p1.get(i), a2p2.get(i), a2t.get(i)), "authored", a2p2.get(i)
            else:
                f, prov, prim_src = final(p1.get(i), p2n.get(i), None), "natural", tags.get(i)
            c = counts.get(i)
            why = None
            if prov == "natural" and i not in p2n:
                why = "natural item without pass 2" + (" (IN BAND at 8k)" if c and 12 <= c["c"] <= 24 else "")
            elif f is None or not rule(f):
                why = "annotation not eligible"
            elif c is None:
                why = "not shortlisted at 8k stage 1"
            elif c["unknown"]:
                why = "verifier unknown"
            if why:
                excl[why] = excl.get(why, 0) + 1
                continue
            rows.append({"id": i, "c": c["c"], "prompt_tokens": c["prompt_tokens"], "resp_len_mean": c["resp_len_mean"],
                         "truncated": c["truncated"], "answer_type": f.get("answer_type", "other"), "provenance": prov,
                         "primary": (prim_src or {}).get("primary", "none"), "pool": it["pool"]})
        with open(f"{out}/{arm}.jsonl", "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
        inband = [r for r in rows if 12 <= r["c"] <= 24]
        summary[arm] = {"eligible_screened": len(rows), "in_band_12_24": len(inband),
                        "in_band_by_provenance": {p: sum(r["provenance"] == p for r in inband) for p in ("natural", "authored")},
                        "excluded": excl}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
