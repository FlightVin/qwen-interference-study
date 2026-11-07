"""Thinking- vs non-thinking-mode baseline diagnostic (project.md §20, amendment A6; decision D56).

One fixed, seeded panel of 80 distinct problems from the frozen main bundle (sealed test untouched):
32 H_train (proportional to source x primary-technique cells, largest remainder, seeded), 16 related
auxiliary and their 16 matched unrelated partners (seeded pair indices), 16 target_dev items.
Eight attempts per item per mode; only `enable_thinking` changes (same user prompt, sampling, verifier,
cap). Request seeds are shared across modes (common seeds, not identical samples).

Thinking-mode grading: reasoning and answer tokens both count against the cap. Only the text after the
first `</think>` token is graded. No `</think>` -> no final answer: a capped response gets the existing
truncation treatment (reward 0, status truncated); an uncapped one scores 0 with status no_final.
A boxed candidate inside the reasoning can never earn reward.

Usage: python -m qis.mode_diag --bundle B --model M --out DIR --modes off,on --max-tokens 8192
       [--panel-only] [--gpu-mem-util 0.85]
"""
import argparse
import gzip
import json
import os
import time

import numpy as np

from qis import prompts, seeds

PANEL_SEED = 2611          # A6 panel stream master (fixed before any result)
REQ_SEED = 2612            # A6 request-seed stream master (shared by both modes)
N_ATTEMPTS = 8
NO_FINAL = "no_final"
FAIL_SAMPLE_PER_MODE = 16


def select_panel(bundle):
    """Deterministic 80-item panel. Returns {pool: [items]} with pools hard/related/unrelated/dev."""
    P = bundle["pools"]
    r = seeds.rng(PANEL_SEED, "a6/panel")
    hard = sorted(P["hard"], key=lambda x: x["id"])
    cells = {}
    for it in hard:
        cells.setdefault((it["source"], it["family"]), []).append(it)
    keys = sorted(cells)
    quota = {k: 32 * len(cells[k]) / len(hard) for k in keys}
    alloc = {k: int(np.floor(q)) for k, q in quota.items()}
    rest = 32 - sum(alloc.values())
    tieb = r.permutation(len(keys))
    order = sorted(range(len(keys)), key=lambda i: (-(quota[keys[i]] - alloc[keys[i]]), tieb[i]))
    for i in order[:rest]:
        alloc[keys[i]] += 1
    h = []
    for k in keys:
        idx = r.permutation(len(cells[k]))[:alloc[k]]
        h += [cells[k][i] for i in sorted(idx)]
    pairs = sorted(r.permutation(len(P["related"]))[:16].tolist())
    dev = sorted(P["dev"], key=lambda x: x["id"])
    d = [dev[i] for i in sorted(r.permutation(len(dev))[:16].tolist())]
    panel = {"hard": h, "related": [P["related"][k] for k in pairs],
             "unrelated": [P["unrelated"][k] for k in pairs], "dev": d}
    ids = [x["id"] for v in panel.values() for x in v]
    assert len(ids) == 80 and len(set(ids)) == 80, "panel must hold 80 distinct problems"
    return panel, {"hard_alloc": {f"{s}|{f}": n for (s, f), n in alloc.items()}, "pair_indices": pairs}


def render(tok, problem, thinking):
    return tok.apply_chat_template([{"role": "user", "content": prompts.USER_PREFIX + problem}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def check_templates(tok):
    """Both renders differ only by the empty think stub; </think> is one token."""
    off, on = render(tok, "What is 1+1?", False), render(tok, "What is 1+1?", True)
    end_id = tok.convert_tokens_to_ids("</think>")
    ok = (off == on + prompts.THINK_STUB and on.endswith("<|im_start|>assistant\n")
          and isinstance(end_id, int) and tok.encode("</think>", add_special_tokens=False) == [end_id])
    return {"rendered_off": off, "rendered_on": on, "think_end_id": end_id, "ok": bool(ok)}


def split_final(ids, think_end_id):
    """(reasoning_ids, final_ids or None). None means the reasoning segment never closed."""
    if think_end_id not in ids:
        return ids, None
    j = ids.index(think_end_id)
    return ids[:j], ids[j + 1:]


def grade_inputs(gold, ids, finish, thinking, decode, think_end_id):
    """Verifier input for one attempt, or a fixed result when there is no final segment."""
    capped = finish == "length"
    if not thinking:
        return {"gold": gold, "response": decode(ids), "truncated": capped}, None, len(ids), 0
    reason, final = split_final(ids, think_end_id)
    if final is None:
        fixed = {"status": "truncated" if capped else NO_FINAL, "reward": 0, "extracted": None}
        return None, fixed, len(ids), len(ids)
    return {"gold": gold, "response": decode(final), "truncated": capped}, None, len(ids), len(reason)


def warm(ver, tries=6):
    """Import math_verify in every worker before real scoring (cold imports can exceed the 5 s/15 s
    limits on a cold node and would surface as 'unknown'). Warm-up results are discarded."""
    for _ in range(tries):
        r = ver.score([{"gold": "1", "response": "\\boxed{1}", "truncated": False}] * ver.n)
        if all(x["status"] == "correct" for x in r):
            return True
    raise RuntimeError("verifier warm-up failed")


def paired_ci(d, name, draws=10_000):
    d = np.asarray(d, dtype=np.float64)
    r = seeds.rng(seeds.BOOTSTRAP_SEED, f"a6/{name}")
    bs = d[r.integers(0, len(d), (draws, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)), "n": len(d)}


def run(a):
    bundle = json.load(open(a.bundle))
    panel, meta = select_panel(bundle)
    os.makedirs(a.out, exist_ok=True)
    pj = {"panel": {k: [x["id"] for x in v] for k, v in panel.items()}, **meta,
          "panel_seed": PANEL_SEED, "req_seed": REQ_SEED, "n_attempts": N_ATTEMPTS}
    json.dump(pj, open(os.path.join(a.out, "panel.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({k: len(v) for k, v in pj["panel"].items()}), meta["hard_alloc"])
    if a.panel_only:
        return
    from transformers import AutoTokenizer
    from qis.train.rollout import Rollout
    from qis.verify import Verifier
    tok = AutoTokenizer.from_pretrained(a.model)
    tpl = check_templates(tok)
    json.dump(tpl, open(os.path.join(a.out, "template_check.json"), "w"), indent=1)
    assert tpl["ok"], "template check failed"
    end_id = tpl["think_end_id"]
    roll = Rollout(a.model, a.max_tokens, a.gpu_mem_util)
    ver = Verifier(8)
    warm(ver)
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)
    per, cost, fails = [], [], []
    rec = gzip.open(os.path.join(a.out, f"records_cap{a.max_tokens}.jsonl.gz"), "wt")
    for mode in a.modes.split(","):
        thinking = mode == "on"
        for pool, items in panel.items():
            ids = [tok(render(tok, it["problem"], thinking), add_special_tokens=False)["input_ids"] for it in items]
            assert max(map(len, ids)) <= prompts.PROMPT_CAP
            req = [seeds.int_seed(REQ_SEED, f"a6/{it['id']}/n{N_ATTEMPTS}") for it in items]
            t0 = time.time()
            outs = roll.generate(ids, N_ATTEMPTS, req, logprobs=False)
            gen_s = time.time() - t0
            inputs, slots = [], []
            for it, g in zip(items, outs):
                for j, o in enumerate(g):
                    vin, fixed, n_tok, n_reason = grade_inputs(it["gold"], o["ids"], o["finish"], thinking, dec, end_id)
                    slots.append((it, j, o, fixed, n_tok, n_reason, len(inputs) if vin else None))
                    if vin:
                        inputs.append(vin)
            sc = ver.score(inputs) if inputs else []
            toks = 0
            byitem = {}
            for it, j, o, fixed, n_tok, n_reason, k in slots:
                res = fixed if fixed is not None else sc[k]
                toks += n_tok
                b = byitem.setdefault(it["id"], {"id": it["id"], "pool": pool, "mode": mode, "c": 0, "n": 0, "trunc": 0,
                                                  "unknown": 0, "no_final": 0, "len": [], "reason_len": [],
                                                  "family": it.get("family"), "source": it.get("source")})
                b["n"] += 1; b["c"] += res["reward"] == 1; b["trunc"] += o["finish"] == "length"
                b["unknown"] += res["status"] == "unknown"; b["no_final"] += res["status"] == NO_FINAL
                b["len"].append(n_tok); b["reason_len"].append(n_reason)
                rec.write(json.dumps({"id": it["id"], "pool": pool, "mode": mode, "cap": a.max_tokens, "sample": j,
                                      "finish": o["finish"], "n_tokens": n_tok, "reason_tokens": n_reason,
                                      "status": res["status"], "reward": res["reward"],
                                      "extracted": res.get("extracted"), "ids": o["ids"]}) + "\n")
                if pool == "hard" and res["reward"] == 0 and res["status"] != "unknown":
                    reason, final = split_final(o["ids"], end_id) if thinking else ([], o["ids"])
                    fails.append({"id": it["id"], "family": it.get("family"), "mode": mode, "sample": j,
                                  "status": res["status"], "extracted": res.get("extracted"), "gold": it["gold"],
                                  "final_text": dec(final)[-800:] if final is not None else None,
                                  "reasoning_tail": dec(reason)[-1500:] if thinking else None})
            per += [{**b, "len_mean": float(np.mean(b["len"])), "reason_len_mean": float(np.mean(b["reason_len"])),
                     "len": None, "reason_len": None} for b in byitem.values()]
            cost.append({"mode": mode, "pool": pool, "gen_s": gen_s, "tokens": toks})
            print(mode, pool, f"{gen_s:.0f}s", toks, flush=True)
    rec.close()
    ver.close()
    # Fixed failure sample: H items in panel order, first failed attempt per item, up to 16 per mode.
    hid = [x["id"] for x in panel["hard"]]
    sample = []
    for mode in a.modes.split(","):
        seen = set()
        for f in sorted((f for f in fails if f["mode"] == mode), key=lambda f: (hid.index(f["id"]), f["sample"])):
            if f["id"] not in seen and len(seen) < FAIL_SAMPLE_PER_MODE:
                seen.add(f["id"]); sample.append(f)
    json.dump(sample, open(os.path.join(a.out, f"failure_sample_cap{a.max_tokens}.json"), "w"), indent=1)
    json.dump(summarize(per, cost, panel, a.max_tokens), open(os.path.join(a.out, f"summary_cap{a.max_tokens}.json"), "w"), indent=1)
    json.dump(per, open(os.path.join(a.out, f"per_item_cap{a.max_tokens}.json"), "w"))


def summarize(per, cost, panel, cap):
    S = {"cap": cap, "pools": {}, "cost": cost}
    modes = sorted({p["mode"] for p in per})
    for pool in panel:
        S["pools"][pool] = {}
        for m in modes:
            rows = [p for p in per if p["pool"] == pool and p["mode"] == m]
            n = sum(p["n"] for p in rows)
            S["pools"][pool][m] = {"items": len(rows), "pass1": float(np.mean([p["c"] / p["n"] for p in rows])),
                                   "mixed_frac": float(np.mean([0 < p["c"] < p["n"] for p in rows])),
                                   "any_success_frac": float(np.mean([p["c"] > 0 for p in rows])),
                                   "trunc_frac": sum(p["trunc"] for p in rows) / n,
                                   "no_final_frac": sum(p["no_final"] for p in rows) / n,
                                   "unknown": sum(p["unknown"] for p in rows),
                                   "len_mean": float(np.mean([p["len_mean"] for p in rows])),
                                   "reason_len_mean": float(np.mean([p["reason_len_mean"] for p in rows]))}
        if {"off", "on"} <= set(modes):
            a = {p["id"]: p["c"] / p["n"] for p in per if p["pool"] == pool and p["mode"] == "off"}
            b = {p["id"]: p["c"] / p["n"] for p in per if p["pool"] == pool and p["mode"] == "on"}
            S["pools"][pool]["paired_pass1_on_minus_off"] = paired_ci([b[i] - a[i] for i in sorted(a)], f"{pool}/{cap}")
    for m in modes:   # R vs U difficulty match over matched pairs (same order)
        r = [next(p for p in per if p["id"] == x["id"] and p["mode"] == m) for x in panel["related"]]
        u = [next(p for p in per if p["id"] == x["id"] and p["mode"] == m) for x in panel["unrelated"]]
        S.setdefault("related_minus_unrelated", {})[m] = paired_ci([x["c"] / x["n"] - y["c"] / y["n"] for x, y in zip(r, u)], f"RU/{m}/{cap}")
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--panel-only", action="store_true")
    run(ap.parse_args())
