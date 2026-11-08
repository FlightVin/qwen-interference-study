"""A11 truncation study on 16,384-cap development traces (read-only; run in a CPU job).

Input: `qis.evaluate_ckpt` output directories (records.jsonl.gz + summary.json) of the dev panel at cap 16,384.
Per directory: share of responses crossing 8,192 tokens; among crossers, outcome = correct / wrong (finished, boxed) /
no_answer (finished, no box) / capped at 16,384; pass@1 at 16,384 vs completion within 8,192; loop score (share of
repeated 20-token windows in the last 1,000 tokens) of crossers by outcome; extra-token share (tokens beyond 8,192
/ all tokens) and wall time as the additional GPU cost. Fixed inspection sample: seeded (1701, "a11/sample"),
up to 8 crossing responses per directory, with problem text, gold, extracted answer, head and tail excerpts.
Usage: python -m qis.analysis.trunc_study BUNDLE MODEL OUT_DIR EVAL_DIR [EVAL_DIR ...]
"""
import gzip
import json
import os
import sys

from qis import seeds
from qis.diag_trunc import loop_score

LIM = 8192


def main(bundle, model, out, *dirs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    probs = {it["id"]: it for it in json.load(open(bundle))["pools"]["dev"]}
    os.makedirs(out, exist_ok=True)
    report, sample = {}, []
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        recs = [json.loads(l) for l in gzip.open(os.path.join(d, "records.jsonl.gz"), "rt")]
        summ = json.load(open(os.path.join(d, "summary.json")))
        cross = [r for r in recs if r["n_tokens"] > LIM]
        def outcome(r):
            if r["finish"] == "length":
                return "capped_16384"
            if r["reward"] == 1:
                return "correct"
            return "no_answer" if r["status"] == "no_answer" else "wrong"
        by = {}
        for r in cross:
            by.setdefault(outcome(r), []).append(r)
        toks = sum(r["n_tokens"] for r in recs)
        report[name] = {
            "responses": len(recs), "cross_8192": len(cross), "cross_frac": len(cross) / len(recs),
            "cross_outcomes": {k: len(v) for k, v in by.items()},
            "cross_then_correct_frac_of_all": len(by.get("correct", [])) / len(recs),
            "cross_then_correct_frac_of_crossers": len(by.get("correct", [])) / len(cross) if cross else None,
            "pass1_16384": summ["pass1"], "pass1_within_8192": summ.get("pass1_within_8192"),
            "capped_16384_frac": summ["truncated"],
            "loop_score_mean_by_outcome": {k: sum(loop_score(r["ids"]) for r in v) / len(v) for k, v in by.items()},
            "loop_gt_0.5_by_outcome": {k: sum(loop_score(r["ids"]) > 0.5 for r in v) for k, v in by.items()},
            "tokens": toks, "tokens_beyond_8192": sum(max(0, r["n_tokens"] - LIM) for r in recs),
            "extra_token_share": sum(max(0, r["n_tokens"] - LIM) for r in recs) / toks if toks else None,
            "eval_seconds": summ.get("seconds"), "tokens_per_second": summ.get("tokens_per_second")}
        rng = seeds.rng(seeds.CONSTRUCTION_SEED, f"a11/sample/{name}")
        for i in rng.permutation(len(cross))[:8].tolist():
            r = cross[i]
            text = tok.decode(r["ids"], skip_special_tokens=True)
            p = probs.get(r["prompt_id"], {})
            sample.append({"source": name, "prompt_id": r["prompt_id"], "sample": r["sample"], "n_tokens": r["n_tokens"],
                           "outcome": outcome(r), "loop_score": loop_score(r["ids"]), "problem": p.get("problem"),
                           "gold": p.get("gold"), "extracted": r["extracted"],
                           "head": text[:1500], "around_8192": tok.decode(r["ids"][LIM - 300:LIM + 300], skip_special_tokens=True),
                           "tail": text[-2500:]})
    json.dump(report, open(os.path.join(out, "trunc_study.json"), "w"), indent=1)
    with open(os.path.join(out, "inspection_sample.jsonl"), "w") as f:
        for s in sample:
            f.write(json.dumps(s) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
