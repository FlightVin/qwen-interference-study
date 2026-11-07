"""Synchronous GRPO training run (one GPU). Usage: python -m qis.train.run --config CFG.json

Per round: schedule 16 prompts -> vLLM generates 8 responses each (BF16) -> verify -> group
advantages -> teacher-forced log probs of the round-start learner (pi_old, fixed for the round)
-> 4 optimizer steps of 4 complete groups (32 responses, micro-batch 1) -> sync weights to vLLM.
Checkpoints every `ckpt_every` committed rounds; scheduled evaluations at `eval_rounds`.

Exit codes: 0 done; 143 preemption notice (requeue); 3 round held (verifier unknown);
4 numerical failure (non-finite loss or gradient).
"""
import argparse
import gzip
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np
import torch

from qis import prompts, seeds
from qis.train import ckpt, grpo
from qis.train.schedule import round_plan
from qis.verify import UNKNOWN, Verifier

EVAL_SEED = 4242            # evaluation streams are independent of training/selection streams
N_PER_PROMPT = 8
N_MINIBATCH = 4


class Held(Exception):
    pass


def qtiles(x: torch.Tensor, ps=(0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)):
    if x.numel() == 0:
        return {}
    x = x.float().flatten()
    if x.numel() > 2_000_000:
        x = x[torch.randperm(x.numel(), device=x.device)[:2_000_000]]
    v = torch.quantile(x, torch.tensor(ps, device=x.device))
    return {f"q{p}": float(a) for p, a in zip(ps, v.tolist())}


class Log:
    """Append-only JSONL (source of truth) + optional W&B mirror."""

    def __init__(self, out_dir, cfg, resume_meta):
        self.f = open(os.path.join(out_dir, "metrics.jsonl"), "a")
        self.ids = {"run_id": cfg["run_id"], "attempt_id": cfg["attempt_id"]}
        self.wb = None
        if cfg.get("wandb", {}).get("enabled"):
            try:
                import wandb
                wid = (resume_meta or {}).get("wandb_id") or f"{cfg['run_id']}-{cfg['attempt_id']}"
                self.wb = wandb.init(project=cfg["wandb"]["project"], id=wid, resume="allow",
                                     group=cfg["run_id"], tags=[cfg["phase"], cfg["arm"]],
                                     config=cfg, mode=os.environ.get("WANDB_MODE", "offline"))
                self.wb.define_metric("opt_step"); self.wb.define_metric("round")
                self.wb.define_metric("step/*", step_metric="opt_step")
                self.wb.define_metric("round/*", step_metric="round")
                self.wb.define_metric("eval/*", step_metric="round")
                self.wandb_id = wid
            except Exception as e:  # noqa: BLE001 -- W&B must never block training
                self.event("wandb_unavailable", error=str(e)[:300])
                self.wb = None

    def write(self, kind, **kv):
        rec = {"kind": kind, "utc": time.time(), **self.ids, **kv}
        self.f.write(json.dumps(rec) + "\n"); self.f.flush()
        if self.wb is not None:
            try:
                flat = {f"{kind}/{k}": v for k, v in kv.items() if isinstance(v, (int, float))}
                if kind == "step":
                    flat["opt_step"] = kv["opt_step"]
                elif "round" in kv:
                    flat["round"] = kv["round"]
                self.wb.log(flat)
            except Exception as e:  # noqa: BLE001
                self.wb = None
                self.event("wandb_failed", error=str(e)[:300])

    def event(self, what, **kv):
        self.write("event", what=what, **kv)


def load_bundle(path):
    b = json.load(open(path))
    return b, hashlib.sha256(open(path, "rb").read()).hexdigest()


def evaluate(rollout, verifier, tok, panel_name, items, n, rnd, rec_dir):
    ids = [tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"] for it in items]
    req_seeds = [seeds.int_seed(EVAL_SEED, f"eval/{panel_name}/{it['id']}/r{rnd}/n{n}") for it in items]
    t0 = time.time()
    outs = rollout.generate(ids, n, req_seeds, logprobs=False)
    scored = verifier.score([{"gold": it["gold"], "response": tok.decode(o["ids"], skip_special_tokens=True),
                              "truncated": o["finish"] == "length"} for it, g in zip(items, outs) for o in g])
    per, k, recs, unknown = [], 0, [], 0
    for it, g, s in zip(items, outs, req_seeds):
        c = 0
        for j, o in enumerate(g):
            r = scored[k]; k += 1
            unknown += r["status"] == UNKNOWN
            c += r["reward"] == 1
            recs.append({"panel": panel_name, "round": rnd, "prompt_id": it["id"], "seed": s, "sample": j,
                         "n_tokens": len(o["ids"]), "finish": o["finish"], "extracted": r["extracted"],
                         "status": r["status"], "reward": r["reward"], "ids": o["ids"]})
        per.append({"id": it["id"], "c": c, "n": n})
    with gzip.open(os.path.join(rec_dir, f"eval_{panel_name}_r{rnd:04d}.jsonl.gz"), "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    toks = sum(x["n_tokens"] for x in recs)
    return {"panel": panel_name, "round": rnd, "n": n, "problems": len(items),
            "pass1": float(np.mean([p["c"] / n for p in per])),
            "pass8": float(np.mean([grpo.pass_at_k(n, p["c"], 8) for p in per])) if n >= 8 else None,
            "truncated": sum(x["finish"] == "length" for x in recs) / len(recs),
            "unknown": unknown, "tokens": toks, "seconds": time.time() - t0, "per_problem": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out, cdir = cfg["out_dir"], cfg["ckpt_dir"]
    rec_dir = os.path.join(out, "records")
    for d in (out, cdir, rec_dir):
        os.makedirs(d, exist_ok=True)
    bundle, bundle_sha = load_bundle(cfg["bundle"])
    pools = bundle["pools"]
    flag = os.environ.get("AE_PREEMPT_FLAG", "")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    from qis.train.learner import Learner
    from qis.train.rollout import Rollout
    rollout = Rollout(cfg["model"], cfg["max_tokens"], cfg["gpu_mem_util"], cfg.get("max_num_seqs", 256))
    learner = Learner(cfg["model"], cfg["lr"])
    verifier = Verifier(cfg.get("verify_workers", 6))

    # ---- resume
    meta, start = None, 0
    src = ckpt.latest(cdir)
    if src is None and cfg.get("init_ckpt"):
        src = {"round": None, "path": cfg["init_ckpt"]}
    counters = {"opt_step": 0, "gen_tokens": 0, "prompt_tokens": 0, "hard_responses": 0}
    if src is not None:
        meta = ckpt.load(src["path"], learner.model, learner.opt)
        rollout.load_weights(learner.model)
        start = meta["round"] if src["round"] is not None else 0
        counters.update(meta.get("counters", {}))
    log = Log(out, cfg, meta)
    log.event("start", host=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
              job=os.environ.get("SLURM_JOB_ID"), restart=os.environ.get("SLURM_RESTART_COUNT", "0"),
              resume_round=start, resume_from=src["path"] if src else None, bundle_sha=bundle_sha,
              code_sha=os.environ.get("AE_SHA"), torch=torch.__version__, argv=sys.argv)
    log.write("config", config=cfg, bundle_sha=bundle_sha)
    if start > 0:
        log.event("resume", round=start, opt_step=counters["opt_step"])

    def preempted():
        return bool(flag) and os.path.exists(flag)

    def run_evals(rnd):
        for panel, n in cfg.get("eval_panels", {}).items():
            done = os.path.join(rec_dir, f"eval_{panel}_r{rnd:04d}.jsonl.gz")
            if os.path.exists(done):
                continue
            r = evaluate(rollout, verifier, tok, panel, pools[panel], n, rnd, rec_dir)
            if r["unknown"]:
                log.event("eval_verifier_unknown", panel=panel, round=rnd, n=r["unknown"])
            log.write("eval", **{k: v for k, v in r.items() if k != "per_problem"},
                      per_problem=r["per_problem"],
                      ckpt_round=rnd)

    if start == 0 and 0 in cfg.get("eval_rounds", []):
        run_evals(0)

    n_hard = len(pools.get("hard", [])); n_pairs = len(pools.get("related", []))
    n_calib = len(pools.get("calib_train", []))
    eps_lo, eps_hi = cfg["eps_low"], cfg["eps_high"]
    ckpt_retain = set(cfg.get("retain_rounds", []))

    for rnd in range(start, cfg["rounds"]):
        if preempted():
            log.event("preempt_exit", round=rnd); sys.exit(143)
        t_round = time.time()
        plan = round_plan(cfg["arm"], cfg["seed"], rnd, n_hard, n_pairs, n_calib)
        items = [(role, pools[pool][i]) for role, pool, i in plan]
        p_ids = [tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"] for _, it in items]
        assert all(len(p) <= prompts.PROMPT_CAP for p in p_ids), "prompt over cap (bundle must pre-filter)"
        req_seeds = [seeds.int_seed(cfg["seed"], f"gen/r{rnd}/p{j}") for j in range(len(items))]
        t0 = time.time()
        groups = rollout.generate(p_ids, N_PER_PROMPT, req_seeds, logprobs=True)
        t_gen = time.time() - t0
        rollout.sleep()

        t0 = time.time()
        flat = [(gi, o) for gi, g in enumerate(groups) for o in g]
        scored = verifier.score([{"gold": items[gi][1]["gold"],
                                  "response": tok.decode(o["ids"], skip_special_tokens=True),
                                  "truncated": o["finish"] == "length"} for gi, o in flat])
        t_ver = time.time() - t0
        unk = [i for i, s in enumerate(scored) if s["status"] == UNKNOWN]
        if unk:
            log.event("round_held_verifier_unknown", round=rnd, n=len(unk),
                      diagnostics=[scored[i].get("diagnostic") for i in unk][:10],
                      predictions=[scored[i]["extracted"] for i in unk][:10])
            raise Held(f"round {rnd}: {len(unk)} verifier unknown")
        rewards = torch.tensor([s["reward"] for s in scored], dtype=torch.float32).view(len(groups), N_PER_PROMPT)
        adv = grpo.group_advantages(rewards)

        # pi_old: round-start learner, teacher forced, fixed for all 4 steps (decision D11)
        t0 = time.time()
        old, mism = {}, []
        with torch.no_grad():
            for gi, g in enumerate(groups):
                for j, o in enumerate(g):
                    lp = learner.token_logprobs(p_ids[gi], o["ids"], train=False)
                    old[(gi, j)] = lp
                    mism.append(lp - torch.tensor(o["lp"], device=lp.device))
        mism = torch.cat(mism)
        t_old = time.time() - t0

        # 4 optimizer steps; minibatches of 4 complete groups with balanced roles
        roles = [r for r, _ in items]
        mb_plan = grpo.minibatch_plan(roles, N_MINIBATCH, seeds.rng(cfg["seed"], f"minibatch/r{rnd}"))
        ratios, act_pos, act_neg, n_pos, n_neg = [], 0.0, 0.0, 0.0, 0.0
        t0 = time.time()
        for m, mb in enumerate(mb_plan):
            if preempted():
                log.event("preempt_exit_midround", round=rnd, minibatch=m); sys.exit(143)
            members = [(gi, j) for gi in mb for j in range(N_PER_PROMPT)]
            tot_loss, valid_tok = torch.zeros((), device="cuda"), 0
            for gi, j in members:
                o = groups[gi][j]
                lp = learner.token_logprobs(p_ids[gi], o["ids"], train=True)
                A = float(adv[gi, j])
                obj, st = grpo.response_objective(lp, old[(gi, j)], A, eps_lo, eps_hi)
                loss = -obj / len(members)
                loss.backward()
                tot_loss += loss.detach(); valid_tok += len(o["ids"])
                ratios.append(st["ratio"])
                if A > 0:
                    act_pos += st["clip_active"].sum(); n_pos += st["n_tokens"]
                elif A < 0:
                    act_neg += st["clip_active"].sum(); n_neg += st["n_tokens"]
            gn = learner.grad_norm_and_clip(cfg.get("grad_clip", 1.0))
            finite = bool(torch.isfinite(gn)) and bool(torch.isfinite(tot_loss))
            log.write("step", opt_step=counters["opt_step"] + 1, round=rnd, minibatch=m,
                      loss=float(tot_loss), policy_loss=float(tot_loss), grad_norm_preclip=float(gn),
                      grad_clipped=bool(gn > cfg.get("grad_clip", 1.0)), lr=cfg["lr"],
                      valid_tokens=valid_tok, responses=len(members), nonfinite=not finite,
                      kl_coef=0.0, entropy_coef=0.0)
            if not finite:
                log.event("nonfinite", round=rnd, minibatch=m); sys.exit(4)
            learner.step()
            counters["opt_step"] += 1
        t_train = time.time() - t0

        t0 = time.time()
        rollout.load_weights(learner.model)
        t_sync = time.time() - t0

        # ---- round metrics and attempt records
        ratios = torch.cat(ratios)
        lens = torch.tensor([len(o["ids"]) for _, o in flat], dtype=torch.float32)
        succ = {}
        for role in sorted(set(roles)):
            gis = [gi for gi, r in enumerate(roles) if r == role]
            succ[role] = float(rewards[gis].mean())
        ng = rewards.sum(1)
        counters["gen_tokens"] += int(lens.sum()); counters["prompt_tokens"] += sum(len(p) * N_PER_PROMPT for p in p_ids)
        counters["hard_responses"] += N_PER_PROMPT * sum(r.startswith("hard") for r in roles)
        rm = {"round": rnd + 1, "success": succ,
              "groups_all0": int((ng == 0).sum()), "groups_all1": int((ng == N_PER_PROMPT).sum()),
              "groups_mixed": int(((ng > 0) & (ng < N_PER_PROMPT)).sum()),
              "len_mean": float(lens.mean()), "len_q": qtiles(lens),
              "truncated": sum(o["finish"] == "length" for _, o in flat),
              "no_answer": sum(s["status"] == "no_answer" for s in scored),
              "ratio_q": qtiles(ratios), "clipfrac_pos": float(act_pos / max(n_pos, 1)),
              "clipfrac_neg": float(act_neg / max(n_neg, 1)),
              "logprob_mismatch_absq": qtiles(mism.abs()), "logprob_mismatch_mean": float(mism.mean()),
              "t_gen": t_gen, "t_verify": t_ver, "t_oldlp": t_old, "t_train": t_train, "t_sync": t_sync,
              "t_round": time.time() - t_round, **counters}
        if (rnd + 1) % 8 == 0:
            ent = [float(learner.token_entropy(p_ids[gi], o["ids"])) for gi, o in flat[:8]]
            rm["entropy_sample_mean"] = float(np.mean(ent))
            rm["mem_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        log.write("round", **rm)
        with gzip.open(os.path.join(rec_dir, f"train_r{rnd:04d}.jsonl.gz"), "wt") as f:
            for k, ((gi, o), s) in enumerate(zip(flat, scored)):
                j = k % N_PER_PROMPT
                f.write(json.dumps({"round": rnd, "prompt_id": items[gi][1]["id"], "role": roles[gi], "slot": gi,
                                    "seed": req_seeds[gi], "sample": j, "n_tokens": len(o["ids"]),
                                    "finish": o["finish"], "extracted": s["extracted"], "status": s["status"],
                                    "reward": s["reward"], "adv": float(adv[gi, j]), "ids": o["ids"]}) + "\n")

        done_rounds = rnd + 1
        if done_rounds in cfg.get("eval_rounds", []):
            run_evals(done_rounds)
        if done_rounds % cfg["ckpt_every"] == 0 or done_rounds == cfg["rounds"] or done_rounds in ckpt_retain:
            info = ckpt.save(cdir, done_rounds, learner.model, learner.opt,
                             {"counters": counters, "config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                              "bundle_sha": bundle_sha, "wandb_id": getattr(log, "wandb_id", None),
                              "attempt_id": cfg["attempt_id"]},
                             keep=2, retain=ckpt_retain)
            log.event("checkpoint", round=done_rounds, **info)

    log.event("complete", rounds=cfg["rounds"], **counters)
    with open(os.path.join(out, "RUN_COMPLETE"), "w") as f:
        f.write(json.dumps(counters))
    verifier.close()


if __name__ == "__main__":
    try:
        main()
    except Held as e:
        print(f"HELD: {e}", file=sys.stderr)
        sys.exit(3)
