"""A26 pass A (agent a-evals): end-to-end traces of representative records. Read-only on $P.

No project imports. Uses the Qwen tokenizer (transformers) only to decode stored token ids, and a re-implementation of
seeds.rng / grpo.minibatch_plan (src/qis/seeds.py, src/qis/train/grpo.py:44-64) to recover a step's four groups.
Usage: python traces.py OUT.json
"""
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

P = "/data/user_data/vineethb/qis"
MODEL = f"{P}/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def rng(master, name):
    d = hashlib.sha256(name.encode()).digest()
    key = tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))
    return np.random.default_rng(np.random.SeedSequence(entropy=master, spawn_key=key))


def minibatch_plan(roles, n_mb, g):
    by_role = {}
    for i, r in enumerate(roles):
        by_role.setdefault(r, []).append(i)
    plan = [[] for _ in range(n_mb)]
    for r in sorted(by_role):
        gs = by_role[r]
        perm = g.permutation(len(gs))
        k = len(gs) // n_mb
        for m in range(n_mb):
            plan[m] += [gs[i] for i in perm[m * k:(m + 1) * k]]
    order = g.permutation(n_mb)
    return [sorted(plan[m]) for m in order]


def boxes(s):
    out, i = [], 0
    while True:
        j = s.find("\\boxed{", i)
        if j < 0:
            return out
        k, d = j + 7, 1
        while k < len(s) and d:
            d += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        out.append(s[j + 7:k - 1])
        i = k


def recs(path, pred=None):
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if pred is None or pred(r):
                yield r


def metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"kind": "unparsable"})
    return rows


def main(out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    T = {it["id"]: it for it in json.load(open(f"{P}/data/main-v1/target_test_sealed.json"))["pools"]["test"]}
    S = {it["id"]: it for it in json.load(open(f"{P}/data/stress/stress_v1.json"))["pools"]["stress"]}
    res = {}

    # (a) ordinary correct sealed-test attempt: R-s101, first record with reward 1
    d = f"{P}/test-eval/qis-test-R-s101-sym"
    r = next(recs(f"{d}/records.jsonl.gz", lambda r: r["reward"] == 1))
    txt = dec(r["ids"])
    summ = json.load(open(f"{d}/summary.json"))
    pp = {p["id"]: p for p in summ["per_problem"]}[r["prompt_id"]]
    allp = [x for x in recs(f"{d}/records.jsonl.gz", lambda x: x["prompt_id"] == r["prompt_id"])]
    res["a_ordinary"] = {"file": f"{d}/records.jsonl.gz", "record": {k: v for k, v in r.items() if k != "ids"},
                         "gold": T[r["prompt_id"]]["gold"], "problem_head": T[r["prompt_id"]]["problem"][:300],
                         "boxes": boxes(txt), "text_tail": txt[-400:],
                         "problem_records": len(allp), "problem_c_from_records": sum(x["reward"] == 1 for x in allp),
                         "summary_per_problem": pp, "summary_pass1": summ["pass1"], "summary_problems": summ["problems"]}

    # (b) capped attempt and its length extension: R-s202 sealed test, first extended row that was recovered,
    #     plus the first extended row that stayed capped
    d = f"{P}/length-test/R-s202-r128/all"
    rows = list(recs(f"{d}/rows.jsonl.gz", lambda r: r["extended"]))
    rec_row = next(x for x in rows if x["reward_2"] == 1)
    cap_row = next(x for x in rows if x["finish_2"] == "length")
    out_b = {}
    for name, row in (("recovered", rec_row), ("still_capped", cap_row)):
        o = next(recs(f"{P}/test-eval/qis-test-R-s202-sym/records.jsonl.gz",
                      lambda x: x["prompt_id"] == row["prompt_id"] and x["sample"] == row["sample"]))
        full = dec(o["ids"] + row["new_ids"])
        out_b[name] = {"original": {k: v for k, v in o.items() if k != "ids"}, "original_tail": dec(o["ids"])[-300:],
                       "row": {k: v for k, v in row.items() if k != "new_ids"}, "gold": T[row["prompt_id"]]["gold"],
                       "continuation_tail": full[-400:], "boxes_full": boxes(full)[-3:],
                       "prefix_preserved": len(o["ids"]) == row["n_tokens_1"] == 12288}
    out_b["summary"] = json.load(open(f"{d}/summary.json"))
    res["b_capped_and_extension"] = out_b

    # (c) multiple-box answer: stress math:precalculus/train/00152 (gold 2,4), attempt 0 of the audited policies,
    #     and every attempt in every stress cell whose boxes contain both 2 and 4 separately
    pid = "math:precalculus/train/00152"
    out_c = {"gold": S[pid]["gold"], "cells": {}}
    for cell in sorted(os.listdir(f"{P}/stress-eval")):
        f = f"{P}/stress-eval/{cell}/records.jsonl.gz"
        if not os.path.exists(f):
            continue
        rs = sorted(recs(f, lambda x: x["prompt_id"] == pid), key=lambda x: x["sample"])
        cc = []
        for x in rs:
            bx = boxes(dec(x["ids"]))
            cc.append({"sample": x["sample"], "reward": x["reward"], "finish": x["finish"], "extracted": x["extracted"],
                       "status": x["status"], "n_boxes": len(bx), "boxes": bx[-4:]})
        two = [c for c in cc if c["reward"] == 0 and c["finish"] != "length" and {"2", "4"} <= {b.strip() for b in c["boxes"]}]
        out_c["cells"][cell] = {"attempts": len(cc), "correct": sum(c["reward"] for c in cc), "sample0": cc[0] if cc else None,
                                "wrong_with_separate_2_and_4_boxes": len(two)}
    res["c_multibox"] = out_c

    # (g) sharded endpoint: H-s202 round 128, sealed test; problem index 3 (shard k3, RTX A6000)
    ids = list(T)
    pid = ids[3]
    sh = f"{P}/test-eval/qis-test-H-s202-sym-shards/k3"
    srs = sorted(recs(f"{sh}/records.jsonl.gz", lambda x: x["prompt_id"] == pid), key=lambda x: x["sample"])
    mrs = sorted(recs(f"{P}/test-eval/qis-test-H-s202-sym/records.jsonl.gz", lambda x: x["prompt_id"] == pid), key=lambda x: x["sample"])
    ssum = json.load(open(f"{sh}/summary.json"))
    msum = json.load(open(f"{P}/test-eval/qis-test-H-s202-sym/summary.json"))
    res["g_sharded"] = {"prompt_id": pid, "panel_index": 3, "shard": "k3", "shard_panel_ids": [it["id"] for it in json.load(open(f"{sh}/panel_3.json"))["pools"]["test"]][:4],
                        "shard_records": len(srs), "shard_samples": [x["sample"] for x in srs], "shard_seeds": sorted({x["seed"] for x in srs}),
                        "shard_c": sum(x["reward"] == 1 for x in srs), "shard_trunc": sum(x["finish"] == "length" for x in srs),
                        "shard_summary_pp": {p["id"]: p for p in ssum["per_problem"]}[pid],
                        "merged_records": len(mrs), "merged_identical_to_shard": [json.dumps(a, sort_keys=True) for a in srs] == [json.dumps(b, sort_keys=True) for b in mrs],
                        "merged_summary_pp": {p["id"]: p for p in msum["per_problem"]}[pid], "merged_pass1": msum["pass1"],
                        "merged_shards": msum.get("shards"), "seed_label": msum.get("seed_label")}

    # (d) resumed runs: H-s202 (a2-orig vs a2-r60) and U-s202 (reruns of 65-80 / 65-81)
    out_d = {}
    for run in ("qis-main-H-s202-sym", "qis-main-U-s202-sym", "qis-main-R-s202-sym", "qis-main-R-s101-sym"):
        rows = metrics(f"{P}/runs/{run}/metrics.jsonl")
        ev = [{"row": i, "what": r.get("what"), "resume_round": r.get("resume_round", r.get("round")), "job": r.get("job"),
               "host": r.get("host"), "gpu": r.get("gpu"), "code_sha": (r.get("code_sha") or "")[:8], "utc": r.get("utc")}
              for i, r in enumerate(rows) if r.get("kind") == "event" and r.get("what") in ("start", "resume", "preempt_exit", "complete")]
        rcount = Counter(r["round"] for r in rows if r.get("kind") == "round")
        dup = {k: v for k, v in sorted(rcount.items()) if v > 1}
        evals = [{"row": i, "panel": r["panel"], "round": r["round"], "pass1": r["pass1"], "n": r["n"], "max_tokens": r.get("max_tokens")}
                 for i, r in enumerate(rows) if r.get("kind") == "eval"]
        out_d[run] = {"rows": len(rows), "events": ev, "round_rows_duplicated": dup, "distinct_rounds": len(rcount), "evals": evals}
        files = sorted(os.listdir(f"{P}/runs/{run}"))
        out_d[run]["dir"] = files
        for sub in files:
            if sub.startswith("records") and sub != "records" or sub.startswith("superseded"):
                out_d[run][f"listing_{sub}"] = sorted(os.listdir(f"{P}/runs/{run}/{sub}"))[:12]
    pre = f"{P}/runs/qis-main-H-s202-sym/metrics.pre_resume_a17.jsonl"
    rows = metrics(pre)
    out_d["H-s202 pre_resume_a17"] = {"rows": len(rows), "max_round": max((r["round"] for r in rows if r.get("kind") == "round"), default=None),
                                      "evals": [{"panel": r["panel"], "round": r["round"], "pass1": r["pass1"]} for r in rows if r.get("kind") == "eval"],
                                      "config": next((r["config"] for r in rows if r.get("kind") == "config"), None)}
    for f in ("resume_audit_a17.txt", "resume_audit_a17b.txt"):
        p = f"{P}/runs/qis-main-H-s202-sym/{f}"
        out_d[f] = open(p).read()[:1500] if os.path.exists(p) else "unavailable"
    for ck in ("round_0064", "round_0128"):
        st = f"{P}/ckpt/qis-main-H-s202-sym/{ck}/state.json"
        if os.path.exists(st):
            s = json.load(open(st))
            out_d[f"H-s202 {ck} state.json"] = {k: s.get(k) for k in ("round", "attempt_id", "saved_utc", "saved", "time", "counters", "file_sha256")}
            out_d[f"H-s202 {ck} state.json keys"] = sorted(s)
            out_d[f"H-s202 {ck} state.json mtime"] = os.path.getmtime(st)
    fb = f"{P}/ckpt/qis-main-H-s202-sym-fallback"
    out_d["fallback listing"] = sorted(os.listdir(fb)) if os.path.isdir(fb) else "unavailable"
    if os.path.exists(f"{fb}/ckpt/state.json"):
        s = json.load(open(f"{fb}/ckpt/state.json"))
        out_d["fallback state.json"] = {k: s.get(k) for k in ("round", "attempt_id", "counters", "file_sha256")}
    res["d_resumed"] = out_d

    # (e) all-correct group in training and (f) momentum-only step: U-s202 (first zero-gradient step) and R-s101 round 101
    out_e = {}
    run = "qis-main-R-s101-sym"
    rr = list(recs(f"{P}/runs/{run}/records/train_r0100.jsonl.gz"))
    by = defaultdict(list)
    for x in rr:
        by[x["slot"]].append(x)
    for slot in sorted(by):
        g = by[slot]
        if all(x["reward"] == 1 for x in g):
            out_e = {"run": run, "file": "records/train_r0100.jsonl.gz", "round_1based": 101, "slot": slot, "role": g[0]["role"],
                     "prompt_id": g[0]["prompt_id"], "attempts": len(g), "rewards": [x["reward"] for x in g], "adv": [x["adv"] for x in g],
                     "n_tokens": [x["n_tokens"] for x in g], "finish": [x["finish"] for x in g]}
            break
    M = metrics(f"{P}/runs/{run}/metrics.jsonl")
    rrow = [r for r in M if r.get("kind") == "round" and r["round"] == 101][-1]
    gc = Counter()
    for slot, g in by.items():
        c = sum(x["reward"] for x in g)
        gc["all0" if c == 0 else "all1" if c == 8 else "mixed"] += 1
    out_e["round_row_groups"] = {"all0": rrow["groups_all0"], "all1": rrow["groups_all1"], "mixed": rrow["groups_mixed"]}
    out_e["records_groups"] = dict(gc)
    out_e["zero_adv_attempts_in_round"] = sum(1 for x in rr if x["adv"] == 0.0)
    res["e_all_correct"] = out_e

    out_f = {}
    for run, seed in (("qis-main-U-s202-sym", 202), ("qis-main-H-s202-sym", 202)):
        M = metrics(f"{P}/runs/{run}/metrics.jsonl")
        # canonical step rows: last occurrence per (round, minibatch) (a17_train convention after resumes)
        steps = {}
        for i, r in enumerate(M):
            if r.get("kind") == "step":
                steps[(r["round"], r["minibatch"])] = (i, r)
        zero = sorted(k for k, (i, r) in steps.items() if r["grad_norm_preclip"] == 0.0)
        k0 = zero[0]
        rnd, mb = k0
        rr = list(recs(f"{P}/runs/{run}/records/train_r{rnd:04d}.jsonl.gz"))
        slots = defaultdict(list)
        for x in rr:
            slots[x["slot"]].append(x)
        roles = [slots[s][0]["role"] for s in range(16)]
        plan = minibatch_plan(roles, 4, rng(seed, f"minibatch/r{rnd}"))
        members = plan[mb]
        grp = [{"slot": s, "role": roles[s], "prompt_id": slots[s][0]["prompt_id"], "correct": sum(x["reward"] for x in slots[s]),
                "adv_nonzero": sum(1 for x in slots[s] if x["adv"] != 0.0)} for s in members]
        neigh = [dict(steps[(rnd, m)][1], row=steps[(rnd, m)][0]) for m in range(4)] + \
                ([dict(steps[(rnd + 1, 0)][1], row=steps[(rnd + 1, 0)][0])] if (rnd + 1, 0) in steps else [])
        all_zero_ok = []
        for (zr, zm) in zero:
            rz = list(recs(f"{P}/runs/{run}/records/train_r{zr:04d}.jsonl.gz"))
            sl = defaultdict(list)
            for x in rz:
                sl[x["slot"]].append(x)
            rl = [sl[s][0]["role"] for s in range(16)]
            pl = minibatch_plan(rl, 4, rng(seed, f"minibatch/r{zr}"))
            all_zero_ok.append(all(x["adv"] == 0.0 for s in pl[zm] for x in sl[s]))
        out_f[run] = {"zero_grad_steps": len(zero), "first_zero_step_round_index0": rnd, "first_zero_step_round_1based": rnd + 1,
                      "minibatch": mb, "step_row": dict(steps[k0][1], row=steps[k0][0]), "plan": plan, "members": grp,
                      "rows_this_round_and_next": [{k: v.get(k) for k in ("row", "opt_step", "round", "minibatch", "grad_norm_preclip", "loss", "grad_clipped", "valid_tokens")} for v in neigh],
                      "all_zero_grad_steps_have_all_32_adv_zero": f"{sum(all_zero_ok)}/{len(all_zero_ok)}",
                      "weight_change_logged": any("update_norm" in r or "param_delta" in r or "weight_change" in r for _, r in steps.values())}
    res["f_momentum_only"] = out_f

    # dev / probe values from canonical eval rows (last occurrence per (panel, round) after the last resume) for all runs
    out_dev = {}
    for run in ("H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202"):
        M = metrics(f"{P}/runs/qis-main-{run}-sym/metrics.jsonl")
        last_res = {}
        ev = {}
        for i, r in enumerate(M):
            if r.get("kind") == "event" and r.get("what") == "resume":
                s = r.get("resume_round", r.get("round"))
                for k in [k for k in ev if k[1] > s]:
                    del ev[k]
            if r.get("kind") == "eval":
                ev[(r["panel"], r["round"])] = {"pass1": r["pass1"], "n": r["n"], "problems": r["problems"], "max_tokens": r.get("max_tokens"), "row": i,
                                                "c": [p["c"] for p in r.get("per_problem", [])]}
        out_dev[run] = {f"{k[0]}@{k[1]}": v for k, v in sorted(ev.items())}
    res["dev_probe_canonical"] = out_dev
    json.dump(res, open(out, "w"), indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
