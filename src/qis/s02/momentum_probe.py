"""Measure weight and policy movement under zero-gradient AdamW steps (study s02 report, Section on momentum).

CPU-only diagnostic. It reproduces the optimizer behaviour of qis.train (AdamW lr 2e-6, betas (0.9, 0.999), eps 1e-8,
weight decay 0, grad clip 1.0, loss divided by a fixed 32 responses) on Qwen3-1.7B in FP32, using recorded GRPO
training groups (prompt, response token ids, advantage) from a finished run. At ratio 1 the clipped GRPO objective's
gradient equals the policy gradient sum_i A_i * grad mean_t log p(y_it), so one "real" step here equals the first
minibatch step of a training round on that data.

Phases (all from the base model):
  warm-up   W real steps, one recorded mixed group (n responses) per step -> builds Adam moments
  zero      Z steps whose gradients are exact zero tensors (what a round of all-equal groups produces)
  single    1 real step on a held-out mixed group, then Z2 zero steps
At every step: L2 norm of the weight change, cosine with the previous change, cumulative distance from the phase start.
At listed steps: mean per-token log-probability of a fixed probe set of held-out correct answers (two strata).

Output: one JSON lines file (--out). Usage (compute node, CPU):
  python -m qis.s02.momentum_probe --model M --bundle B --train-records DIR --probe-records F1,F2 --out OUT
"""
import argparse
import glob
import gzip
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qis import prompts


def flat_params(model):
    return [p for p in model.parameters()]


def snapshot(params):
    return [p.detach().clone() for p in params]


def diff_stats(params, prev, anchor):
    """(||theta - prev||, ||theta - anchor||, dot(delta, prev_delta) helper returns delta list)."""
    d2, a2 = 0.0, 0.0
    deltas = []
    for p, q, a in zip(params, prev, anchor):
        d = (p.detach() - q)
        deltas.append(d)
        d2 += float((d.double() ** 2).sum())
        a2 += float(((p.detach() - a).double() ** 2).sum())
    return math.sqrt(d2), math.sqrt(a2), deltas


def cosine(d1, d2):
    if d1 is None or d2 is None:
        return None
    num = sum(float((x.double() * y.double()).sum()) for x, y in zip(d1, d2))
    n1 = math.sqrt(sum(float((x.double() ** 2).sum()) for x in d1))
    n2 = math.sqrt(sum(float((y.double() ** 2).sum()) for y in d2))
    return num / (n1 * n2) if n1 > 0 and n2 > 0 else None


def seq_logprob(model, prompt_ids, resp_ids, grad):
    ids = torch.tensor([prompt_ids + resp_ids])
    with torch.set_grad_enabled(grad):
        logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(resp_ids)[:, None])[:, 0]
    return lp.mean()


def load_groups(train_dir, tok, items, rounds, strata, max_resp_tokens):
    """Mixed groups (some adv != 0) from train records: list of (prompt_ids, [(resp_ids, adv)], meta)."""
    groups = []
    for rnd in rounds:
        p = os.path.join(train_dir, f"train_r{rnd:04d}.jsonl.gz")
        by = {}
        for l in gzip.open(p, "rt"):
            r = json.loads(l)
            if r["role"] != "T" or (strata and r["stratum"] not in strata):
                continue
            by.setdefault((r["slot"], r["prompt_id"]), []).append(r)
        for (slot, pid), rs in sorted(by.items()):
            if any(abs(x["adv"]) > 0 for x in rs) and max(x["n_tokens"] for x in rs) <= max_resp_tokens:
                it = items[pid]
                pids = tok(prompts.render(tok, it["problem"], it.get("prefix")), add_special_tokens=False)["input_ids"]
                groups.append((pids, [(x["ids"], x["adv"]) for x in rs],
                               {"round": rnd, "prompt_id": pid, "stratum": rs[0]["stratum"],
                                "n_correct": sum(x["reward"] == 1 for x in rs)}))
    return groups


def load_probe(files, tok, items, per_file):
    out = []
    for f in files:
        seen = set()
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if r["status"] != "correct" or r["prompt_id"] in seen:
                continue
            seen.add(r["prompt_id"])
            it = items[r["prompt_id"]]
            pids = tok(prompts.render(tok, it["problem"], it.get("prefix")), add_special_tokens=False)["input_ids"]
            out.append((os.path.basename(f), pids, r["ids"]))
            if len(seen) >= per_file:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--train-records", required=True, help="records dir of a finished run")
    ap.add_argument("--probe-records", required=True, help="comma-separated eval record files (correct answers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--strata", default="d4")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--zero", type=int, default=20)
    ap.add_argument("--zero2", type=int, default=12)
    ap.add_argument("--probe-per-file", type=int, default=8)
    ap.add_argument("--probe-at", default="0,1,2,4,8,12,20")
    ap.add_argument("--max-resp-tokens", type=int, default=1200)
    ap.add_argument("--truncate-resp", type=int, default=0, help="canary only: cut responses to this many tokens")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    t0 = time.time()
    log = open(a.out, "w")

    def emit(**kv):
        kv["t"] = round(time.time() - t0, 1)
        log.write(json.dumps(kv) + "\n"); log.flush()
        print(json.dumps(kv), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    bundle = json.load(open(a.bundle))
    items = {it["id"]: it for pool in bundle["pools"].values() for it in pool}
    groups = load_groups(a.train_records, tok, items, [int(x) for x in a.rounds.split(",")],
                         set(a.strata.split(",")) if a.strata else None, a.max_resp_tokens)
    probe = load_probe(a.probe_records.split(","), tok, items, a.probe_per_file)
    if a.truncate_resp:
        groups = [(p, [(r[:a.truncate_resp], adv) for r, adv in g], m) for p, g, m in groups]
        probe = [(f, p, r[:a.truncate_resp]) for f, p, r in probe]
    need = a.warmup + 1
    assert len(groups) >= need, f"only {len(groups)} mixed groups found, need {need}"
    emit(event="setup", groups_found=len(groups), probe=len(probe), probe_files=sorted({f for f, _, _ in probe}),
         warmup_groups=[m for _, _, m in groups[:a.warmup]], single_group=groups[a.warmup][2])

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    params = flat_params(model)
    opt = torch.optim.AdamW(params, lr=2e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=True)

    def probe_lp():
        model.eval()
        res = {}
        with torch.no_grad():
            for f, p, r in probe:
                res.setdefault(f, []).append(float(seq_logprob(model, p, r, grad=False)))
        model.train()
        return {f: sum(v) / len(v) for f, v in res.items()}

    def real_step(group):
        pids, resp, meta = group
        gn_loss = 0.0
        for rids, adv in resp:
            if adv == 0:
                continue
            loss = -adv * seq_logprob(model, pids, rids, grad=True) / 32.0
            loss.backward()
            gn_loss += float(loss)
        gn = torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], 1.0, foreach=True)
        opt.step(); opt.zero_grad(set_to_none=True)
        return float(gn)

    def zero_step():
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step(); opt.zero_grad(set_to_none=True)

    def run_phase(name, n_steps, kind, groups_iter=None, probe_steps=()):
        anchor = snapshot(params)
        prev, prev_delta = anchor, None
        if 0 in probe_steps:
            emit(phase=name, step=0, probe=probe_lp())
        for s in range(1, n_steps + 1):
            gn = real_step(next(groups_iter)) if kind == "real" else (zero_step() or 0.0)
            step_norm, cum, delta = diff_stats(params, prev, anchor)
            rec = {"phase": name, "step": s, "kind": kind, "grad_norm": gn, "step_norm": step_norm,
                   "cum_from_phase_start": cum, "cos_prev": cosine(delta, prev_delta),
                   "adam_step_count": int(next(iter(opt.state.values()))["step"]) if opt.state else 0}
            if s in probe_steps:
                rec["probe"] = probe_lp()
            emit(**rec)
            prev, prev_delta = snapshot(params), delta
        return

    probe_at = {int(x) for x in a.probe_at.split(",")}
    it = iter(groups)
    run_phase("warmup", a.warmup, "real", groups_iter=it, probe_steps={0, a.warmup})
    run_phase("zero", a.zero, "zero", probe_steps=probe_at)
    run_phase("single", 1, "real", groups_iter=it, probe_steps={0, 1})
    run_phase("zero_after_single", a.zero2, "zero", probe_steps={x for x in probe_at if x <= a.zero2})
    emit(event="done")


if __name__ == "__main__":
    main()
