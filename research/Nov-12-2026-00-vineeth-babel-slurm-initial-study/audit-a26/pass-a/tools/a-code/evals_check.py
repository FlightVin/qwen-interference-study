#!/usr/bin/env python3
"""A26 pass A (a-code): check evaluation outputs against evaluate_ckpt / eval_merge / eval_extend behaviour.

Read-only on $P; numpy only (seed re-implementation from src/qis/seeds.py). No scores are computed or printed:
only invariants (counts of rows that satisfy each rule).
evaluate_ckpt outputs (records.jsonl.gz + summary.json):
  complete   (prompt_id, sample) unique; ids == panel ids; samples 0..n-1 per id
  seeds      record seed == int_seed(4242, "eval/{panel}/{id}/{seed_label}/n{n}")    (evaluate_ckpt.py:68)
  grading    finish=length -> n_tokens == max_tokens, reward 0, status truncated; stop -> last id EOS; reward in {0,1}
             or None only with status unknown; unknown count
  summary    per_problem (c, c_within_8192, n) == recomputed from records; "unknown" field == recount
eval_extend outputs (rows.jsonl.gz): extended <=> finish_1 == length; extended rows have n_tokens_1 == from_cap;
  n_tokens_2 <= to_cap; finish_2 == length -> n_tokens_2 == to_cap and reward_2 == 0; not extended -> *_2 == *_1;
  reward_1 == reward of the source record; --select all covers every source record exactly once.
Usage: python evals_check.py --P /data/user_data/vineethb/qis --cue-dir <worktree>/data/cue --out X.json
"""
import argparse, glob, gzip, hashlib, json, os, collections
import numpy as np

EOS = {151645, 151643}


def name_key(name):
    d = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))


def int_seed(master, name):
    return int(np.random.SeedSequence(entropy=master, spawn_key=name_key(name)).generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def check_eval(d, panel_ids):
    s = json.load(open(os.path.join(d, "summary.json")))
    n, key, lab, cap = s["n"], s["panel"], s["seed_label"], s["max_tokens"]
    t = collections.Counter()
    seen = collections.Counter()
    by = collections.defaultdict(list)
    for line in gzip.open(os.path.join(d, "records.jsonl.gz"), "rt"):
        r = json.loads(line)
        t["records"] += 1
        seen[(r["prompt_id"], r["sample"])] += 1
        by[r["prompt_id"]].append(r)
        t["seed_ok"] += r["seed"] == int_seed(4242, f"eval/{key}/{r['prompt_id']}/{lab}/n{n}")
        ids = r.get("ids") or []
        t["ntok_eq_len"] += r["n_tokens"] == len(ids)
        if r["finish"] == "length":
            t["capped"] += 1
            t["capped_ok"] += r["n_tokens"] == cap and r["reward"] == 0 and r["status"] == "truncated"
        elif r["finish"] == "stop":
            t["stop"] += 1
            t["stop_eos"] += bool(ids) and ids[-1] in EOS
        t["unknown"] += r["status"] == "unknown"
        t["status_reward_ok"] += ((r["status"] == "correct") == (r["reward"] == 1)) and (
            r["reward"] in (0, 1) or (r["reward"] is None and r["status"] == "unknown"))
    t["dup_keys"] = sum(v > 1 for v in seen.values())
    t["ids_eq_panel"] = set(by) == set(panel_ids) if panel_ids is not None else "panel unknown"
    t["samples_complete"] = all(sorted(x["sample"] for x in g) == list(range(n)) for g in by.values())
    per = {p["id"]: p for p in s["per_problem"]}
    ok = 0
    for pid, g in by.items():
        c = sum(x["reward"] == 1 for x in g)
        c8 = sum(x["reward"] == 1 and x["n_tokens"] <= 8192 for x in g)
        p = per.get(pid, {})
        ok += (p.get("c"), p.get("c_within_8192", c8), p.get("n")) == (c, c8, n)
    t["per_problem_eq_records"] = f"{ok}/{len(by)}"
    t["summary_unknown_field"] = s.get("unknown")
    meta = {k: s.get(k) for k in ("panel", "n", "seed_label", "max_tokens", "ckpt", "ckpt_model_sha256", "problems")}
    meta["shards"] = len(s["shards"]) if "shards" in s else None
    return {"dir": d, **meta, **dict(t)}


def check_extend(d, src_records, to_cap=16384, from_cap=12288):
    src = {}
    for line in gzip.open(src_records, "rt"):
        r = json.loads(line)
        src[(r["prompt_id"], r["sample"])] = r
    s = json.load(open(os.path.join(d, "summary.json")))
    t = collections.Counter()
    keys = collections.Counter()
    for line in gzip.open(os.path.join(d, "rows.jsonl.gz"), "rt"):
        r = json.loads(line)
        k = (r["prompt_id"], r["sample"])
        keys[k] += 1
        t["rows"] += 1
        o = src.get(k)
        t["src_found"] += o is not None
        if o is not None:
            t["reward1_eq_src"] += r["reward_1"] == o["reward"] and r["n_tokens_1"] == o["n_tokens"] and r["finish_1"] == o["finish"]
        t["extended"] += r["extended"]
        t["extended_iff_capped"] += r["extended"] == (r["finish_1"] == "length")
        if r["extended"]:
            t["ext_ntok1_eq_from_cap"] += r["n_tokens_1"] == from_cap
            t["ext_ntok2_le_to_cap"] += r["n_tokens_2"] <= to_cap and r["n_tokens_2"] == r["n_tokens_1"] + r["new_tokens"] == r["n_tokens_1"] + len(r.get("new_ids", []))
            if r["finish_2"] == "length":
                t["ext_still_capped"] += 1
                t["ext_still_capped_ok"] += r["n_tokens_2"] == to_cap and r["reward_2"] == 0
            else:
                t["ext_finished_eos"] += bool(r.get("new_ids")) and r["new_ids"][-1] in EOS
                t["ext_finished"] += 1
        else:
            t["kept_unchanged"] += (r["reward_2"], r["n_tokens_2"], r["finish_2"]) == (r["reward_1"], r["n_tokens_1"], r["finish_1"])
    t["dup_rows"] = sum(v > 1 for v in keys.values())
    t["covers_all_src"] = set(keys) == set(src)
    return {"dir": d, "select": s.get("select"), "seed_label": s.get("seed_label"), "from_cap": s.get("from_cap"),
            "to_cap": s.get("to_cap"), "src_records": src_records, "src_n": len(src), **dict(t)}


def panel_ids(path, key):
    return [it["id"] for it in json.load(open(path))["pools"][key]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", required=True)
    ap.add_argument("--cue-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = a.P
    test_ids = panel_ids(f"{P}/data/main-v1/target_test_sealed.json", "test")
    stress_ids = panel_ids(f"{P}/data/stress/stress_v1.json", "stress")
    cue_ids = panel_ids(os.path.join(a.cue_dir, "cue_v1_uncued.json"), "cue")
    out = {"eval": [], "extend": []}
    for sj in sorted(glob.glob(f"{P}/test-eval/*/summary.json") + glob.glob(f"{P}/test-eval/*/k*/summary.json")
                     + glob.glob(f"{P}/test-eval-cgpu/*/summary.json")):
        d = os.path.dirname(sj)
        ids = test_ids
        if "/k" in d[-3:]:                       # shard: its own interleaved panel file
            k = int(os.path.basename(d)[1:])
            pf = os.path.join(d, f"panel_{k}.json")
            ids = panel_ids(pf, "test") if os.path.exists(pf) else None
        out["eval"].append(check_eval(d, ids)); print(out["eval"][-1]["dir"], flush=True)
    for sj in sorted(glob.glob(f"{P}/stress-eval/*/summary.json") + glob.glob(f"{P}/stress-eval/*/k*/summary.json")):
        d = os.path.dirname(sj)
        ids = stress_ids
        if os.path.basename(d).startswith("k"):
            k = int(os.path.basename(d)[1:])
            pf = os.path.join(d, f"panel_{k}.json")
            ids = panel_ids(pf, "stress") if os.path.exists(pf) else None
        out["eval"].append(check_eval(d, ids)); print(out["eval"][-1]["dir"], flush=True)
    for sj in sorted(glob.glob(f"{P}/cue-eval/v1/*/summary.json")):
        out["eval"].append(check_eval(os.path.dirname(sj), cue_ids)); print(out["eval"][-1]["dir"], flush=True)
    # length extension on the sealed test (select all) and on stress (t1 / rest)
    lt_src = {"H-s101-r128": "qis-test-H-s101-sym", "R-s101-r128": "qis-test-R-s101-sym", "R-s202-r128": "qis-test-R-s202-sym",
              "U-s101-r128": "qis-test-U-s101-sym", "U-s202-r128": "qis-test-U-s202-sym"}
    for lab, t in lt_src.items():
        d = f"{P}/length-test/{lab}/all"
        if os.path.exists(os.path.join(d, "rows.jsonl.gz")):
            out["extend"].append(check_extend(d, f"{P}/test-eval/{t}/records.jsonl.gz")); print(d, flush=True)
    for d in sorted(glob.glob(f"{P}/length-diag/*/*")):
        lab = os.path.basename(os.path.dirname(d))
        if os.path.exists(os.path.join(d, "rows.jsonl.gz")):
            x = check_extend(d, f"{P}/stress-eval/{lab}/records.jsonl.gz")
            x["covers_all_src"] = "n/a (tranche)"
            out["extend"].append(x); print(d, flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
