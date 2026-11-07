"""Intake checks for authored auxiliary candidates (D32).

For each item: Math-Verify gold self-check (same verifier as training); prompt within the
1,024-token cap under the frozen template; near-duplicate check against all of MATH (train+test,
every split) with the same MinHash/Jaccard machinery as the split builder, plus an exact
max-Jaccard report; template families counted. Items failing any check are excluded with a reason.
Writes <out>/<arm>.jsonl (id, problem, gold, subject='authored', level) and intake_summary.json.
"""
import argparse
import collections
import json
import os

from qis import prompts
from qis.data import splits
from qis.verify import Verifier

DUP_JACCARD = 0.5     # stricter than the split builder (0.7): authored items must be clearly new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authored", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--also-dedupe", nargs="*", default=[], help="earlier authored jsonl files (problem field)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = [json.loads(l) for f in a.authored for l in open(f)]
    math_rows = [json.loads(l) for l in open(a.manifest)]
    msh = [(r["id"], r["split"], splits.shingles(r["problem"])) for r in math_rows]
    for f in a.also_dedupe:
        msh += [(r["id"], "authored-earlier", splits.shingles(r["problem"])) for r in (json.loads(l) for l in open(f))]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    v = Verifier(4)
    gold = v.score([{"gold": it["answer"], "response": "\\boxed{" + it["answer"] + "}", "truncated": False} for it in items])
    v.close()
    kept, excl = collections.defaultdict(list), []
    report = []
    for it, g in zip(items, gold):
        sh = splits.shingles(it["problem"])
        best = max(((len(sh & s) / max(1, len(sh | s)), i, sp) for i, sp, s in msh), key=lambda x: x[0])
        ntok = len(tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"])
        why = None
        if g["status"] != "correct":
            why = f"gold not verifier-eligible ({g['status']})"
        elif ntok > prompts.PROMPT_CAP:
            why = "prompt over cap"
        elif best[0] >= DUP_JACCARD:
            why = f"near-duplicate of {best[1]} ({best[2]}, J={best[0]:.2f})"
        report.append({"id": it["id"], "max_jaccard": round(best[0], 3), "nearest": best[1], "nearest_split": best[2],
                       "prompt_tokens": ntok, "excluded": why})
        if why:
            excl.append({"id": it["id"], "reason": why}); continue
        kept[it["arm"]].append({"id": it["id"], "problem": it["problem"], "gold": it["answer"], "subject": "authored",
                                "level": "authored", "template_family": it["template_family"],
                                "primary_by_author": it.get("primary") or it.get("subtype")})
    for arm, rows in kept.items():
        with open(os.path.join(a.out, f"{arm}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "intake_report.jsonl"), "w") as f:
        for r in report:
            f.write(json.dumps(r) + "\n")
    summ = {"n": len(items), "kept": {k: len(v) for k, v in kept.items()}, "excluded": excl,
            "families": {k: len({r["template_family"] for r in v}) for k, v in kept.items()},
            "max_jaccard_overall": max(r["max_jaccard"] for r in report)}
    json.dump(summ, open(os.path.join(a.out, "intake_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)[:2000])


if __name__ == "__main__":
    main()
