"""EXPLORATORY (A19 §2, D154): length diagnostic. Would capped stress-v1 answers succeed with 16,384 response tokens?

Inputs: PANEL = data/stress/stress_v1.json (item id, tier) and ROOT/<label>/{t1,rest}/rows.jsonl.gz written by
qis.eval_extend (one row per attempt: reward_1 / finish_1 at the frozen cap 12,288; reward_2 / finish_2 at 16,384 on the same
trajectory; new_tokens). Labels: {R,U}-s{101,202}-r128. Sets: "t1" (fixed 64-problem × 4-attempt sample) and "all" (t1 + rest,
the full 221 × 8 panel); a set is used for a checkpoint only when its parts exist.
Outputs in OUT_DIR (length.json, length.md):
1. cells per set × tier (superhard, harder, all) × checkpoint: attempts, problems, pass@1 at both caps (mean over problems of
   correct / attempts), capped at 12,288, recovered (capped → correct), finished wrong, still capped at 16,384, extra tokens,
   extra tokens per recovered answer.
2. R − U per seed at both caps and the change (R − U at 16,384 minus R − U at 12,288): paired problem bootstrap (10,000
   draws; stream sha256("2027/a19-length/<set>/<tier>/<seed>")).
3. inspect.json: the preselected attempts (D154: per checkpoint the 3 t1 attempts capped at 12,288 with the smallest
   sha256("a19-inspect/<id>/<sample>")), with the last 3,000 characters before the cut and the continuation, decoded with
   the tokenizer at --model (skipped without it).
Usage: python -m qis.analysis.a19_length PANEL ROOT OUT_DIR [--model MODEL_DIR]
"""
import argparse
import gzip
import hashlib
import json
import os

import numpy as np

DRAWS = 10_000
LABELS = ["R-s101-r128", "U-s101-r128", "R-s202-r128", "U-s202-r128"]
TIERS = ["superhard", "harder", "all"]
SETS = {"t1": ["t1"], "all": ["t1", "rest"]}


def load(root, label, parts):
    rows = []
    for p in parts:
        f = os.path.join(root, label, p, "rows.jsonl.gz")
        if not os.path.exists(f):
            return None
        with gzip.open(f, "rt") as g:
            rows += [json.loads(line) for line in g]
    return rows


def cell(rows):
    by = {}
    for r in rows:
        by.setdefault(r["prompt_id"], []).append(r)
    ext = [r for r in rows if r["extended"]]
    rec = sum(r["reward_2"] == 1 for r in ext)
    new = sum(r["new_tokens"] for r in rows)
    return {"attempts": len(rows), "problems": len(by),
            "pass1_12288": float(np.mean([np.mean([r["reward_1"] == 1 for r in g]) for g in by.values()])),
            "pass1_16384": float(np.mean([np.mean([r["reward_2"] == 1 for r in g]) for g in by.values()])),
            "correct_12288": sum(r["reward_1"] == 1 for r in rows), "correct_16384": sum(r["reward_2"] == 1 for r in rows),
            "capped_12288": len(ext), "recovered": rec,
            "finished_wrong": sum(r["finish_2"] != "length" and r["reward_2"] != 1 for r in ext),
            "still_capped": sum(r["finish_2"] == "length" for r in ext),
            "extra_tokens": new, "extra_tokens_per_recovered": new / rec if rec else None}


def per_problem(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["prompt_id"], []).append(r[key] == 1)
    return {p: float(np.mean(v)) for p, v in by.items()}


def contrast(rr, ru, name):
    r1, r2 = per_problem(rr, "reward_1"), per_problem(rr, "reward_2")
    u1, u2 = per_problem(ru, "reward_1"), per_problem(ru, "reward_2")
    ids = sorted(set(r1) & set(u1))
    a = {k: np.array([d[i] for i in ids]) for k, d in (("r1", r1), ("r2", r2), ("u1", u1), ("u2", u2))}
    est = {"d_12288": lambda ix: (a["r1"][ix] - a["u1"][ix]).mean(), "d_16384": lambda ix: (a["r2"][ix] - a["u2"][ix]).mean(),
           "change": lambda ix: ((a["r2"][ix] - a["u2"][ix]) - (a["r1"][ix] - a["u1"][ix])).mean()}
    rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    n, full = len(ids), np.arange(len(ids))
    draws = [rng.integers(0, n, n) for _ in range(DRAWS)]
    out = {"problems": n, "R_12288": float(a["r1"].mean()), "U_12288": float(a["u1"].mean()),
           "R_16384": float(a["r2"].mean()), "U_16384": float(a["u2"].mean())}
    for k, f in est.items():
        b = np.array([f(ix) for ix in draws])
        out[k] = {"est": float(f(full)), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975))}
    return out


def inspect(root, rows_by_label, model):
    picks = []
    for lab, rows in rows_by_label.items():
        ext = [r for r in rows if r["extended"]]
        ext.sort(key=lambda r: hashlib.sha256(f"a19-inspect/{r['prompt_id']}/{r['sample']}".encode()).hexdigest())
        picks += [(lab, r) for r in ext[:3]]
    if not model:
        return [{"label": lab, "prompt_id": r["prompt_id"], "sample": r["sample"]} for lab, r in picks]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    out = []
    for lab, r in picks:
        orig = None
        with gzip.open(os.path.join(root, "..", "stress-eval", lab, "records.jsonl.gz"), "rt") as g:
            for line in g:
                x = json.loads(line)
                if x["prompt_id"] == r["prompt_id"] and x["sample"] == r["sample"]:
                    orig = x["ids"]
                    break
        out.append({"label": lab, "prompt_id": r["prompt_id"], "sample": r["sample"], "reward_2": r["reward_2"],
                    "finish_2": r["finish_2"], "new_tokens": r["new_tokens"], "extracted_2": r.get("extracted_2"),
                    "before_cut_tail": tok.decode(orig, skip_special_tokens=True)[-3000:],
                    "continuation": tok.decode(r["new_ids"], skip_special_tokens=True)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel"); ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tier = {it["id"]: it["tier"] for it in json.load(open(a.panel))["pools"]["stress"]}
    res, md = {"sets": {}}, ["# Length diagnostic (A19 §2, D154; EXPLORATORY)", ""]
    for s, parts in SETS.items():
        rows = {lab: load(a.root, lab, parts) for lab in LABELS}
        S = res["sets"][s] = {"available": [lab for lab, r in rows.items() if r is not None], "cells": {}, "contrasts": {}}
        md += [f"## Set {s} ({' + '.join(parts)}); available: {', '.join(S['available']) or 'none'}", "",
               "| Tier | Checkpoint | Problems | Attempts | pass@1 12,288 | pass@1 16,384 | Capped at 12,288 | Recovered | "
               "Finished wrong | Still capped | Extra tokens per recovered |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in TIERS:
            for lab in LABELS:
                if rows[lab] is None:
                    continue
                c = S["cells"].setdefault(t, {})[lab] = cell([r for r in rows[lab] if t == "all" or tier[r["prompt_id"]] == t])
                tpr = f"{c['extra_tokens_per_recovered']:,.0f}" if c["extra_tokens_per_recovered"] else "—"
                md.append(f"| {t} | {lab} | {c['problems']} | {c['attempts']} | {c['pass1_12288']:.3f} | {c['pass1_16384']:.3f} | "
                          f"{c['capped_12288']} | {c['recovered']} | {c['finished_wrong']} | {c['still_capped']} | {tpr} |")
        md += ["", "| Tier | Seed | Problems | R − U at 12,288 | R − U at 16,384 | Change |", "|---|---|---|---|---|---|"]
        for t in TIERS:
            for seed in ("s101", "s202"):
                R, U = f"R-{seed}-r128", f"U-{seed}-r128"
                if rows[R] is None or rows[U] is None:
                    continue
                sel = lambda x: [r for r in x if t == "all" or tier[r["prompt_id"]] == t]  # noqa: E731
                c = S["contrasts"].setdefault(t, {})[seed] = contrast(sel(rows[R]), sel(rows[U]), f"2027/a19-length/{s}/{t}/{seed}")
                f = lambda k: f"{c[k]['est']:+.3f} [{c[k]['lo']:+.3f}, {c[k]['hi']:+.3f}]"  # noqa: E731
                md.append(f"| {t} | {seed} | {c['problems']} | {c['R_12288']:.3f} − {c['U_12288']:.3f} = {f('d_12288')} | "
                          f"{c['R_16384']:.3f} − {c['U_16384']:.3f} = {f('d_16384')} | {f('change')} |")
        md.append("")
        if s == "t1":
            ins = inspect(a.root, {lab: r for lab, r in rows.items() if r is not None}, a.model)
            json.dump(ins, open(os.path.join(a.out, "inspect.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(res, open(os.path.join(a.out, "length.json"), "w"), indent=1)
    open(os.path.join(a.out, "length.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
