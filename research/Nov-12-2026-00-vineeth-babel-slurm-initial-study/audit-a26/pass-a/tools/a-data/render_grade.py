"""a-data (A26 pass A): exact prompts, prompt lengths, round-1 grading examples, diagnostic draws, length-extension samples.

Runs ON COMPUTE with project code on PYTHONPATH (qis.prompts.render, qis.verify.Verifier, qis.seeds, qis.eval_extend.select).
Read-only on $P; writes one JSON (argv[2]).
Usage: PYTHONPATH=<W>/src python render_grade.py W OUT.json
"""
import collections
import gzip
import json
import os
import sys

from transformers import AutoTokenizer

from qis import prompts, seeds
from qis.eval_extend import select as ext_select
from qis.verify import Verifier

P = "/data/user_data/vineethb/qis"
MODEL = f"{P}/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def main(W, out_path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = {"render": {}, "prompt_tokens": {}, "grading": {}, "diag": {}, "length": {}}
    B = json.load(open(f"{P}/data/main-v1/bundle_main_v1.json"))["pools"]
    sets = {"H_train": B["hard"], "R_pool": B["related"], "U_pool": B["unrelated"], "dev": B["dev"], "J_R": B["probe_J_R"],
            "J_U": B["probe_J_U"], "test": json.load(open(f"{P}/data/main-v1/target_test_sealed.json"))["pools"]["test"],
            "calib_train": json.load(open(f"{P}/data/calib-panels-k8/bundle.json"))["pools"]["calib_train"],
            "calib_val": json.load(open(f"{P}/data/calib-panels-k8/bundle.json"))["pools"]["calib_val"],
            "stress": json.load(open(f"{P}/data/stress/stress_v1.json"))["pools"]["stress"],
            "cue_uncued": json.load(open(f"{W}/data/cue/cue_v1_uncued.json"))["pools"]["cue"],
            "cue_cued": json.load(open(f"{W}/data/cue/cue_v1_cued.json"))["pools"]["cue"]}
    for k, v in sets.items():
        lens = [len(tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"]) for it in v]
        out["prompt_tokens"][k] = {"n": len(lens), "min": min(lens), "max": max(lens), "over_1024": sum(x > prompts.PROMPT_CAP for x in lens)}
        it = v[0]
        out["render"][k] = {"id": it["id"], "gold": it["gold"], "rendered": prompts.render(tok, it["problem"])}
    # ---- grading examples from round 1 (train_r0000) of the seed-101 runs
    ver = Verifier(4)
    for run, role, label in (("qis-main-H-s101-sym", "hard_common", "hard"), ("qis-main-R-s101-sym", "aux", "related"),
                             ("qis-main-U-s101-sym", "aux", "unrelated")):
        pool = {"hard": B["hard"], "related": B["related"], "unrelated": B["unrelated"]}[label]
        gold = {x["id"]: x["gold"] for x in pool}
        acc, rej, stat = [], [], collections.Counter()
        want_rej = ["wrong", "no_answer", "truncated"]
        with gzip.open(f"{P}/runs/{run}/records/train_r0000.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["role"] != role:
                    continue
                stat[r["status"]] += 1
                ex = {"prompt_id": r["prompt_id"], "slot": r["slot"], "sample": r["sample"], "gold": gold[r["prompt_id"]],
                      "status": r["status"], "reward": r["reward"], "extracted": r["extracted"], "n_tokens": r["n_tokens"],
                      "finish": r["finish"], "ids": r["ids"]}
                if r["reward"] == 1 and len(acc) < 3 and all(a["prompt_id"] != r["prompt_id"] for a in acc):
                    acc.append(ex)
                elif r["reward"] == 0 and r["status"] in want_rej and all(x["status"] != r["status"] for x in rej):
                    rej.append(ex)
        # fill rejected to 3 with other wrong examples when a status is absent
        if len(rej) < 3:
            with gzip.open(f"{P}/runs/{run}/records/train_r0000.jsonl.gz", "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r["role"] == role and r["reward"] == 0 and len(rej) < 3 and all(
                            (x["prompt_id"], x["sample"]) != (r["prompt_id"], r["sample"]) for x in rej) and r["status"] == "wrong" and all(x["prompt_id"] != r["prompt_id"] for x in rej):
                        rej.append({"prompt_id": r["prompt_id"], "slot": r["slot"], "sample": r["sample"], "gold": gold[r["prompt_id"]],
                                    "status": r["status"], "reward": r["reward"], "extracted": r["extracted"], "n_tokens": r["n_tokens"],
                                    "finish": r["finish"], "ids": r["ids"]})
        exs = acc + rej
        texts = [tok.decode(e["ids"], skip_special_tokens=True) for e in exs]
        regr = ver.score([{"gold": e["gold"], "response": t, "truncated": e["finish"] == "length"} for e, t in zip(exs, texts)])
        for e, t, g in zip(exs, texts, regr):
            del e["ids"]
            e["response_tail"] = t[-350:]
            e["regrade"] = {k: g.get(k) for k in ("status", "reward", "extracted", "pred_parsed", "would_match")}
            e["regrade_agrees"] = (g["status"], g["reward"]) == (e["status"], e["reward"])
        out["grading"][label] = {"run": run, "role": role, "status_counts_round1": dict(stat), "accepted": acc, "rejected": rej}
    ver.close()
    # ---- local diagnostic draws (diag_local.branch_prompts logic: seeds.rng(mb, 'diag/hard'|'diag/pairs'))
    rec = {}
    for lab in ("r0", "r64"):
        for line in open(f"{W}/research/evidence/a15/diag_{lab}.jsonl"):
            r = json.loads(line)
            if r.get("branch") in ("hard", "related", "unrelated"):
                rec[(lab, r["branch"], r["seed"])] = r.get("prompt_ids") or r.get("prompts")
    agree, draws = 0, {}
    for mb in (4101, 4102, 4103, 4104):
        hi = sorted(seeds.rng(mb, "diag/hard").permutation(len(B["hard"]))[:4].tolist())
        pi = sorted(seeds.rng(mb, "diag/pairs").permutation(len(B["related"]))[:4].tolist())
        d = {"hard": [B["hard"][i]["id"] for i in hi], "related": [B["related"][i]["id"] for i in pi],
             "unrelated": [B["unrelated"][i]["id"] for i in pi], "pair_idx": pi, "hard_idx": hi}
        draws[mb] = d
        for lab in ("r0", "r64"):
            for br in ("hard", "related", "unrelated"):
                agree += rec.get((lab, br, mb)) == d[br]
    out["diag"] = {"draws": draws, "recorded_equal_recomputed": agree, "of": 24,
                   "distinct_hard": len({i for d in draws.values() for i in d["hard"]}),
                   "distinct_pairs": len({i for d in draws.values() for i in d["pair_idx"]})}
    # ---- length extension samples
    S = sets["stress"]
    t1 = set(map(tuple, ext_select(S)))
    out["length"]["tranche1_planned"] = {"attempts": len(t1), "items": len({i for i, _ in t1}),
                                         "by_tier": dict(collections.Counter(next(x["tier"] for x in S if x["id"] == i) for i, _ in t1 if _ == 0))}
    for base in ("length-diag", "length-test"):
        for pol in sorted(os.listdir(f"{P}/{base}")):
            for part in ("t1", "rest", "all"):
                p = f"{P}/{base}/{pol}/{part}/rows.jsonl.gz"
                if not os.path.exists(p):
                    continue
                keys, ext, n1 = set(), 0, collections.Counter()
                with gzip.open(p, "rt") as f:
                    for line in f:
                        r = json.loads(line)
                        keys.add((r["prompt_id"], r["sample"]))
                        ext += r["extended"]
                        if r["extended"]:
                            n1[r["n_tokens_1"]] += 1
                d = {"rows": len(keys), "items": len({i for i, _ in keys}), "extended": ext, "n_tokens_1_of_extended": dict(n1)}
                if base == "length-diag" and part == "t1":
                    d["equals_planned_tranche1"] = keys == t1
                if base == "length-diag" and part == "rest":
                    d["disjoint_from_tranche1"] = not (keys & t1)
                    d["t1_plus_rest_equals_all_221x8"] = len(keys | t1) == 221 * 8
                out["length"][f"{base}/{pol}/{part}"] = d
    json.dump(out, open(out_path, "w"), indent=1, sort_keys=True, default=str)
    print("written", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
