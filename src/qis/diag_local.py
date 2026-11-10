"""Local update diagnostic (project.md §7 "Local update diagnostic"; PROTOCOL §8; A9.2).

From one source state (model AND optimizer), clone four branches: none, hard, related, unrelated. Each update branch
draws 4 prompts x 8 responses from the SOURCE policy and applies ONE GRPO optimizer step (same objective, clip,
advantages, reduction and grad clip as training; pi_old = source, teacher forced). Minibatch seeds 4101-4104.
Related and unrelated use the same matched pair indices (related[k] <-> unrelated[k]); hard uses its own draw.
Every branch, and the source itself ("none"), is evaluated on J_H (32 dev problems), J_R and J_U with 16
responses/problem and evaluation seeds shared across branches and sources (paired). Source checkpoints are only read.
Reported as finite changes after an actual AdamW/GRPO update, not as gradient cosines.
Usage: python -m qis.diag_local --bundle B --model M --source base|CKPT_DIR --label r0|r64 --out DIR
       [--lr 2e-6 --eps-low 0.2 --eps-high 0.2]
"""
import argparse
import copy
import json
import os
import time

import torch

from qis import prompts, seeds
from qis.train import grpo

MB_SEEDS = (4101, 4102, 4103, 4104)
BRANCHES = ("hard", "related", "unrelated")


def branch_order():
    """Seed-major order: a partial run holds complete (hard, related, unrelated) sets for its first seeds."""
    return [(s, b) for s in MB_SEEDS for b in BRANCHES]


def completed(path):
    """(branch, seed) pairs already written by an earlier attempt; they are skipped on resume."""
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                                                  # partial last line of a killed attempt
        out.add((r["branch"], r["seed"]))
    return out
PANELS = ("probe_J_H", "probe_J_R", "probe_J_U")
N_PROMPTS, N_RESP, N_EVAL = 4, 8, 16
EVAL_ROUND_LABEL = 7064          # constant label in the eval seed key -> identical seeds for every branch/source


def branch_prompts(pools, branch, mb_seed):
    """Deterministic 4-prompt draw. Related/unrelated share pair indices; hard draws from H_train."""
    if branch == "hard":
        idx = seeds.rng(mb_seed, "diag/hard").permutation(len(pools["hard"]))[:N_PROMPTS]
        return [pools["hard"][i] for i in sorted(idx.tolist())], sorted(idx.tolist())
    idx = sorted(seeds.rng(mb_seed, "diag/pairs").permutation(len(pools["related"]))[:N_PROMPTS].tolist())
    return [pools[branch][i] for i in idx], idx


def snapshot(model, opt):
    return ({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
            copy.deepcopy({"state": {k: {n: (t.detach().to("cpu", copy=True) if torch.is_tensor(t) else t)
                                         for n, t in v.items()} for k, v in opt.state_dict()["state"].items()},
                           "param_groups": opt.state_dict()["param_groups"]}))


def restore(model, opt, snap):
    msd, osd = snap
    model.load_state_dict(msd)
    opt.load_state_dict(copy.deepcopy(osd))
    opt.zero_grad(set_to_none=True)


def one_update(learner, groups, p_ids, rewards, eps_lo, eps_hi, grad_clip=1.0):
    """One optimizer step over all responses (one minibatch of 4 groups), exactly as in training."""
    adv = grpo.group_advantages(rewards)
    old = {}
    with torch.no_grad():
        for gi, g in enumerate(groups):
            for j, o in enumerate(g):
                old[(gi, j)] = learner.token_logprobs(p_ids[gi], o["ids"], train=False)
    members = [(gi, j) for gi in range(len(groups)) for j in range(len(groups[gi]))]
    for gi, j in members:
        lp = learner.token_logprobs(p_ids[gi], groups[gi][j]["ids"], train=True)
        obj, _ = grpo.response_objective(lp, old[(gi, j)], float(adv[gi, j]), eps_lo, eps_hi)
        (-obj / len(members)).backward()
    gn = learner.grad_norm_and_clip(grad_clip)
    if not bool(torch.isfinite(gn)):
        raise RuntimeError("non-finite gradient in diagnostic update")
    learner.step()
    return float(gn), int(((rewards.sum(1) > 0) & (rewards.sum(1) < rewards.shape[1])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--eps-low", type=float, default=0.2)
    ap.add_argument("--eps-high", type=float, default=0.2)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from qis.train import ckpt
    from qis.train.learner import Learner
    from qis.train.rollout import Rollout
    from qis.train.run import evaluate
    from qis.verify import UNKNOWN, Verifier
    os.makedirs(a.out, exist_ok=True)
    pools = json.load(open(a.bundle))["pools"]
    tok = AutoTokenizer.from_pretrained(a.model)
    learner = Learner(a.model, a.lr)
    src_meta = {"source": "base", "fresh_optimizer": True}
    if a.source != "base":
        meta = ckpt.load(a.source, learner.model, learner.opt)      # verifies file sha256 and tensor checksums
        src_meta = {"source": a.source, "round": meta.get("round"), "file_sha256": meta.get("file_sha256")}
    for g in learner.opt.param_groups:
        g["lr"] = a.lr
    snap = snapshot(learner.model, learner.opt)
    roll = Rollout(a.model, 8192, 0.30)
    ver = Verifier(8)
    res_path = os.path.join(a.out, f"diag_{a.label}.jsonl")
    done = completed(res_path)
    results = open(res_path, "a")

    def evaluate_all(tag):
        rec = os.path.join(a.out, "records", tag)
        os.makedirs(rec, exist_ok=True)
        out = {}
        for p in PANELS:
            r = evaluate(roll, ver, tok, p, pools[p], N_EVAL, EVAL_ROUND_LABEL, rec)
            if r["unknown"]:
                raise RuntimeError(f"{tag} {p}: {r['unknown']} verifier unknown")
            out[p] = {k: r[k] for k in ("pass1", "truncated", "tokens", "seconds", "per_problem")}
        return out

    t0 = time.time()
    if ("none", None) not in done:
        roll.load_weights(learner.model)
        results.write(json.dumps({"label": a.label, "branch": "none", "seed": None, **src_meta,
                                  "eval": evaluate_all(f"{a.label}-none")}) + "\n"); results.flush()
    for s, branch in branch_order():
        if (branch, s) in done:
            continue
        roll.sleep()                                                  # free vLLM memory before the optimizer restore
        restore(learner.model, learner.opt, snap)
        roll.load_weights(learner.model)                          # generate from the SOURCE policy
        items, idx = branch_prompts(pools, branch, s)
        p_ids = [tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"] for it in items]
        req = [seeds.int_seed(s, f"diag/{branch}/p{j}") for j in range(len(items))]
        groups = roll.generate(p_ids, N_RESP, req, logprobs=False)
        roll.sleep()
        sc = ver.score([{"gold": it["gold"], "response": tok.decode(o["ids"], skip_special_tokens=True),
                         "truncated": o["finish"] == "length"} for it, g in zip(items, groups) for o in g])
        if any(x["status"] == UNKNOWN for x in sc):
            raise RuntimeError(f"{branch} seed {s}: verifier unknown in update batch")
        rewards = torch.tensor([x["reward"] for x in sc], dtype=torch.float32).view(len(items), N_RESP)
        gn, mixed = one_update(learner, groups, p_ids, rewards, a.eps_low, a.eps_high)
        roll.load_weights(learner.model)                          # evaluate the UPDATED branch
        ev = evaluate_all(f"{a.label}-{branch}-{s}")
        results.write(json.dumps({"label": a.label, "branch": branch, "seed": s, "prompt_idx": idx,
                                  "prompt_ids": [it["id"] for it in items], "batch_reward": float(rewards.mean()),
                                  "mixed_groups": mixed, "grad_norm_preclip": gn,
                                  "capped": sum(o["finish"] == "length" for g in groups for o in g),
                                  **src_meta, "eval": ev}) + "\n"); results.flush()
        print(a.label, branch, s, {p: round(ev[p]["pass1"], 4) for p in PANELS}, f"{time.time() - t0:.0f}s", flush=True)
    ver.close()


if __name__ == "__main__":
    main()
