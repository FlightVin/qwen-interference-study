"""A10 long-response smoke test: one complete training round at a 16,384 (or given) cap on one 48 GB GPU.

Same components and batch shape as training: 16 hard prompts x 8 responses, rollout from vLLM (BF16, sleep mode),
teacher-forced pi_old from the FP32 learner, 4 optimizer steps of 4 complete groups, micro-batch 1 response,
grad clip 1.0, AdamW (FP32 moments). Two prompts get `min_tokens` = cap - 1,000 so that at least two groups contain
genuinely long responses (EOS is suppressed only until that length; the test is about memory, limits and clipping,
not about content). Checks: engine max_model_len >= 1,024 + cap; max response length > 8,192; every teacher-forced
log-prob vector has exactly one entry per response token (no silent clipping); finite gradients; peak device memory
and phase times. Does not touch any run directory or checkpoint.
Usage: python -m qis.smoke_long --bundle B --model M --cap 16384 --out OUT.json [--gpu-mem-util 0.30]
"""
import argparse
import json
import time

import torch

from qis import prompts, seeds
from qis.train import grpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from qis.train.learner import Learner
    from qis.train.rollout import Rollout
    from qis.verify import Verifier
    res = {"cap": a.cap, "gpu": torch.cuda.get_device_name(0)}
    tok = AutoTokenizer.from_pretrained(a.model)
    hard = json.load(open(a.bundle))["pools"]["hard"][:16]
    roll = Rollout(a.model, 8192, a.gpu_mem_util, max_model_tokens=a.cap)
    try:
        res["engine_max_model_len"] = roll.llm.llm_engine.model_config.max_model_len
    except AttributeError:
        res["engine_max_model_len"] = roll.llm.llm_engine.vllm_config.model_config.max_model_len
    learner = Learner(a.model, 2e-6)
    ver = Verifier(6)
    p_ids = [tok(prompts.render(tok, it["problem"]), add_special_tokens=False)["input_ids"] for it in hard]
    req = [seeds.int_seed(99, f"smoke/p{j}") for j in range(16)]
    mins = [a.cap - 1000 if j in (0, 1) else 0 for j in range(16)]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    groups = roll.generate(p_ids, 8, req, logprobs=True, max_tokens=a.cap, min_tokens=mins)
    res["t_gen"] = time.time() - t0
    roll.sleep()
    lens = [len(o["ids"]) for g in groups for o in g]
    res.update(max_len=max(lens), n_over_8192=sum(x > 8192 for x in lens), n_capped=sum(o["finish"] == "length" for g in groups for o in g),
               mean_len=sum(lens) / len(lens))
    sc = ver.score([{"gold": it["gold"], "response": tok.decode(o["ids"], skip_special_tokens=True),
                     "truncated": o["finish"] == "length"} for it, g in zip(hard, groups) for o in g])
    rewards = torch.tensor([float(x["reward"] or 0) for x in sc]).view(16, 8)
    adv = grpo.group_advantages(rewards)
    t0 = time.time()
    old, mism_len_ok = {}, True
    with torch.no_grad():
        for gi, g in enumerate(groups):
            for j, o in enumerate(g):
                lp = learner.token_logprobs(p_ids[gi], o["ids"], train=False)
                mism_len_ok &= lp.shape[0] == len(o["ids"]) == len(o["lp"])
                old[(gi, j)] = lp
    res["t_oldlp"] = time.time() - t0
    roles = ["hard_common"] * 8 + ["aux"] * 8
    plan = grpo.minibatch_plan(roles, 4, seeds.rng(99, "smoke/mb"))
    t0, gns, train_len_ok = time.time(), [], True
    for mb in plan:
        members = [(gi, j) for gi in mb for j in range(8)]
        for gi, j in members:
            lp = learner.token_logprobs(p_ids[gi], groups[gi][j]["ids"], train=True)
            train_len_ok &= lp.shape[0] == len(groups[gi][j]["ids"])
            obj, _ = grpo.response_objective(lp, old[(gi, j)], float(adv[gi, j]), 0.2, 0.2)
            (-obj / len(members)).backward()
        gn = learner.grad_norm_and_clip(1.0)
        gns.append(float(gn))
        learner.step()
    res["t_train"] = time.time() - t0
    t0 = time.time()
    roll.load_weights(learner.model)
    res["t_sync"] = time.time() - t0
    f, t = torch.cuda.mem_get_info()
    res.update(grad_norms=gns, finite=all(torch.isfinite(torch.tensor(gns)).tolist()),
               logprob_lengths_match_rollout=bool(mism_len_ok), logprob_lengths_match_train=bool(train_len_ok),
               torch_peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30, device_used_gib_end=(t - f) / 2**30)
    res["pass"] = (res["engine_max_model_len"] >= prompts.PROMPT_CAP + a.cap and res["max_len"] > 8192
                   and res["logprob_lengths_match_rollout"] and res["logprob_lengths_match_train"] and res["finite"])
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    ver.close()


if __name__ == "__main__":
    main()
