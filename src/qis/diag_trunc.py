"""Diagnose truncated responses in evaluation records: repetition loops vs long reasoning.

For each record file: count truncated responses; for truncated ones compute the share of the last
1,000 tokens covered by repeats of a single 20-token window (loop score), and save short tail excerpts.
Usage: python -m qis.diag_trunc MODEL OUT_JSON FILE [FILE...]
"""
import gzip
import json
import sys


def loop_score(ids, w=20, tail=1000):
    t = ids[-tail:]
    if len(t) < 2 * w:
        return 0.0
    seen, rep = {}, 0
    for i in range(len(t) - w + 1):
        k = tuple(t[i:i + w])
        if k in seen:
            rep += 1
        seen[k] = i
    return rep / (len(t) - w + 1)


def main(model, out, *files):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    res = {}
    for f in files:
        recs = [json.loads(l) for l in gzip.open(f, "rt")]
        tr = [r for r in recs if r["finish"] == "length"]
        scores = [loop_score(r["ids"]) for r in tr]
        nontr = [r for r in recs if r["finish"] != "length"]
        res[f.split("/")[-1]] = {
            "n": len(recs), "truncated": len(tr),
            "trunc_loop_score_mean": sum(scores) / max(1, len(scores)),
            "trunc_loop_gt_0.5": sum(s > 0.5 for s in scores),
            "nontrunc_len_mean": sum(r["n_tokens"] for r in nontr) / max(1, len(nontr)),
            "correct_rate": sum(r["reward"] == 1 for r in recs) / len(recs),
            "per_problem_trunc": sorted(((r["prompt_id"]) for r in tr)),
            "tails": [tok.decode(r["ids"][-120:]) for r in tr[:6]]}
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
