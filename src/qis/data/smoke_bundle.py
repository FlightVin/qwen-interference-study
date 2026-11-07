"""Development-only bundle for engineering smoke tests (not a study dataset).

Takes keyword-matched items from the CONSTRUCTION split only (never dev/test/calibration):
hard-like = algebra/intermediate_algebra level 4-5 with factoring keywords, related-like = level
1-3 with factoring keywords, unrelated-like = number theory modular. No annotation, no screening.
"""
import argparse
import json
import re

from qis.data.build import FACTOR_RE, MOD_RE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    pools = {"hard": [], "related": [], "unrelated": []}
    for line in open(a.manifest):
        r = json.loads(line)
        if r["split"] != "construction" or not r.get("gold_eligible") or "[asy]" in r["problem"]:
            continue
        it = {"id": r["id"], "problem": r["problem"], "gold": r["gold"]}
        lvl = int(re.sub(r"\D", "", r["level"]) or 0)
        if r["subject"] in ("algebra", "intermediate_algebra") and FACTOR_RE.search(r["solution"]):
            (pools["hard"] if lvl >= 4 else pools["related"]).append(it)
        elif r["subject"] == "number_theory" and MOD_RE.search(r["problem"] + r["solution"]):
            pools["unrelated"].append(it)
    n = a.n
    b = {"pools": {"hard": pools["hard"][:n], "related": pools["related"][:n], "unrelated": pools["unrelated"][:n],
                   "dev_smoke": pools["hard"][n:n + 8]},
         "note": "development-only smoke bundle from construction split; not a study dataset"}
    json.dump(b, open(a.out, "w"))
    print({k: len(v) for k, v in b["pools"].items()})


if __name__ == "__main__":
    main()
