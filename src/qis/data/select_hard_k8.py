"""H_train selection at the 8,192 cap (D45): the frozen rule of D28/D31/D38 applied to the re-screen.

Candidates = the audit-eligible items re-screened at 8k (rescreen_set.h_k8; items with 4k c > 12/32
are above the band at 8k by monotonicity and are ineligible). Band from 32 fresh 8k attempts; no
verifier unknowns; truncated attempts < 50% of failures. Family = v1.3 primary (native: tags_v13;
Omni: pass-2 primary). Composition and ordering: select_hard.select (unchanged).
Usage: python -m qis.data.select_hard_k8 H_K8_SET H_K8_COUNTS ANN_DIR OUT_JSON
"""
import json
import sys
from collections import Counter

from qis.data.select_hard import band, select


def main(hset, counts, ann, out):
    rd = lambda f: {json.loads(l)["id"]: json.loads(l) for l in open(f)}  # noqa: E731
    c8, tags, o2 = rd(counts), rd(f"{ann}/tags_v13.jsonl"), rd(f"{ann}/omni_pass2.jsonl")
    elig, excl = [], []
    for line in open(hset):
        it = json.loads(line); i = it["id"]; c = c8.get(i)
        fam = o2[i]["primary"] if it["source"] == "omni" else tags[i]["primary"]
        why = None
        if c is None or c["n"] != 32:
            why = "no 32-attempt 8k count"
        elif c["unknown"]:
            why = "verifier unknown"
        elif band(c["c"]) is None:
            why = f"outside bands (c={c['c']}/32)"
        elif c["truncated"] >= 0.5 * max(1, c["n"] - c["c"]):
            why = "difficulty driven by truncation"
        if why:
            excl.append({"id": i, "source": it["source"], "reason": why}); continue
        elig.append({"id": i, "source": it["source"], "band": band(c["c"]), "c32": c["c"], "family": fam})
    sel = select(elig)
    json.dump({"selected": sel, "eligible": elig, "excluded": excl}, open(out, "w"), indent=1)
    print(json.dumps({"eligible": len(elig), "selected": len(sel), "by_band": Counter(e["band"] for e in sel),
                      "by_source": Counter(e["source"] for e in sel), "by_family": Counter(e["family"] for e in sel),
                      "excluded": Counter(e["reason"].split(" (")[0] for e in excl)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
