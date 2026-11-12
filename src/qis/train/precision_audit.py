"""One-time numerical-precision audit of the trainer configuration (project.md §15, A1.1).

Records, from the live objects (not config fields):
  checkpoint tensor dtypes (safetensors on disk), rollout dtype (vLLM model params),
  learner compute dtype (autocast output of a forward), optimizer-owned parameter dtypes,
  gradient dtypes after accumulation, AdamW moment dtypes after the first step.
Then, on a mixed-reward minibatch of real rollouts: nonzero gradient norm; change of a fixed
seeded sample of optimizer-owned weights; accumulation over repeated small updates; rollout sync
delivers bf16(learner weight) exactly; checkpoint restart restores FP32 weights and moments
bit-exactly (no BF16 round trip). Writes precision_audit.json and a markdown table.
"""
import argparse
import collections
import glob
import json
import os
import time

import torch

from qis import prompts, seeds
from qis.train import ckpt, grpo
from qis.train.learner import Learner
from qis.train.rollout import Rollout
from qis.verify import Verifier


def dtypes_of(tensors):
    return dict(collections.Counter(str(t.dtype) for t in tensors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {"code_sha": os.environ.get("AE_SHA"), "gpu": torch.cuda.get_device_name(0), "lr": a.lr}

    from safetensors import safe_open
    st_dtypes = collections.Counter()
    for f in glob.glob(os.path.join(a.model, "*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                st_dtypes[str(h.get_slice(k).get_dtype())] += 1
    R["checkpoint_dtypes"] = dict(st_dtypes)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    probs = [json.loads(line) for line in open(a.problems)]
    roll = Rollout(a.model, a.gpu_mem_util, max_tokens=a.max_tokens)

    def vllm_model():
        return roll.llm.collective_rpc(lambda w: w.model_runner.model)[0]
    R["rollout_param_dtypes"] = dtypes_of(list(vllm_model().parameters()))
    vnames = {n for n, _ in vllm_model().named_parameters()}

    # mixed-reward minibatch: generate 8 responses for each problem, keep 4 groups with mixed rewards
    ids = [tok(prompts.render(tok, p["problem"]), add_special_tokens=False)["input_ids"] for p in probs]
    outs = roll.generate(ids, 8, [seeds.int_seed(7, f"audit/{i}") for i in range(len(ids))])
    ver = Verifier(4)
    groups = []
    for p, pid, g in zip(probs, ids, outs):
        sc = ver.score([{"gold": p["gold"], "response": tok.decode(o["ids"], skip_special_tokens=True),
                         "truncated": o["finish"] == "length"} for o in g])
        r = [s["reward"] for s in sc]
        if None not in r and 0 < sum(r) < 8:
            groups.append((pid, g, torch.tensor(r, dtype=torch.float32)))
    R["mixed_groups_found"] = len(groups)
    groups = groups[:4]
    roll.sleep()

    L = Learner(a.model, a.lr)
    params = dict(L.model.named_parameters())
    R["optimizer_param_dtypes"] = dtypes_of([p for g in L.opt.param_groups for p in g["params"]])
    R["n_param_groups"] = len(L.opt.param_groups)
    R["n_params"] = sum(p.numel() for p in params.values())
    R["all_params_require_grad"] = all(p.requires_grad for p in params.values())

    # compute dtype measured: autocast output of the body
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = L.model.model(input_ids=torch.tensor([ids[0][:16]], device="cuda")).last_hidden_state
    R["learner_hidden_dtype_under_autocast"] = str(h.dtype)
    R["lm_head_logprob_dtype"] = str(L.token_logprobs(ids[0][:8], ids[0][8:16], False).dtype)

    # fixed seeded sample of weights: 4 tensors x 64 elements
    g = seeds.rng(7, "audit/sample")
    names = sorted(n for n in params if n in vnames)     # 1:1 tensors so sync can be compared exactly
    R["n_param_tensors_1to1_with_vllm"] = len(names)
    pick = [names[i] for i in g.choice(len(names), 4, replace=False)]
    idx = {n: torch.as_tensor(g.choice(params[n].numel(), 64, replace=False), device="cuda") for n in pick}
    snap = lambda: {n: params[n].detach().flatten()[idx[n]].clone() for n in pick}  # noqa: E731
    w0 = snap()

    def one_step():
        olds = {}
        with torch.no_grad():
            for gi, (pid, g_, r) in enumerate(groups):
                for j, o in enumerate(g_):
                    olds[(gi, j)] = L.token_logprobs(pid, o["ids"], False)
        n = 8 * len(groups)
        for gi, (pid, g_, r) in enumerate(groups):
            adv = grpo.group_advantages(r[None])[0]
            for j, o in enumerate(g_):
                lp = L.token_logprobs(pid, o["ids"], True)
                obj, _ = grpo.response_objective(lp, olds[(gi, j)], float(adv[j]), 0.2, 0.28)
                (-obj / n).backward()
        grad_dt = dtypes_of([p.grad for p in params.values() if p.grad is not None])
        none_grads = sum(p.grad is None for p in params.values())
        gn = L.grad_norm_and_clip(1.0)
        L.step()
        return float(gn), grad_dt, none_grads

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gn, grad_dt, none_grads = one_step()
    R["step_seconds"] = time.time() - t0
    R["grad_norm_first_step"] = gn
    R["grad_dtypes"] = grad_dt
    R["params_without_grad"] = none_grads
    st = L.opt.state
    R["moment_dtypes"] = {"exp_avg": dtypes_of([s["exp_avg"] for s in st.values()]),
                          "exp_avg_sq": dtypes_of([s["exp_avg_sq"] for s in st.values()]),
                          "step": dtypes_of([s["step"] for s in st.values()])}
    w1 = snap()
    d1 = {n: (w1[n] - w0[n]).abs() for n in pick}
    R["sample_weight_change_step1"] = {n: {"mean_abs": float(d.mean()), "frac_changed": float((d > 0).float().mean())}
                                       for n, d in d1.items()}
    # bf16 visibility after one step: does the serving precision see the change?
    R["frac_bf16_changed_step1"] = float(torch.cat([(w1[n].bfloat16() != w0[n].bfloat16()).float() for n in pick]).mean())
    for _ in range(3):
        one_step()
    w4 = snap()
    d4 = {n: (w4[n] - w0[n]).abs() for n in pick}
    R["sample_weight_change_step4_mean_abs"] = float(torch.cat(list(d4.values())).mean())
    R["accumulates"] = R["sample_weight_change_step4_mean_abs"] > float(torch.cat(list(d1.values())).mean())
    R["frac_bf16_changed_step4"] = float(torch.cat([(w4[n].bfloat16() != w0[n].bfloat16()).float() for n in pick]).mean())
    R["peak_gpu_gib_training"] = torch.cuda.max_memory_allocated() / 2**30

    # sync: vLLM must hold exactly bf16(learner fp32 weight)
    roll.load_weights(L.model)
    vp = dict(vllm_model().named_parameters())
    sync = {}
    for n in pick:
        vn = n if n in vp else None
        if vn is None:
            sync[n] = "not a 1:1 vLLM parameter (fused/stacked); checked via prompt logprob change"
            continue
        sync[n] = bool(torch.equal(vp[vn].detach().flatten()[idx[n]], params[n].detach().flatten()[idx[n]].bfloat16()))
    R["sync_bf16_exact"] = sync
    R["rollout_param_dtypes_after_sync"] = dtypes_of(list(vllm_model().parameters()))
    roll.sleep()

    # restart: save, reload into a fresh learner, compare weights and moments bitwise
    R["gpu_used_gib_device_after_steps"] = (lambda f, t: (t - f) / 2**30)(*torch.cuda.mem_get_info())
    sv = ckpt.Saver(a.ckpt_dir, a.ckpt_dir)
    R["ckpt_snapshot_s"] = sv.save(4, L.model, L.opt, {"audit": True}, blocking=True)
    R["ckpt"] = sv.last
    assert sv.last["ok"], sv.last
    ref_w = {n: p.detach().to("cpu", copy=True) for n, p in params.items()}
    ref_m = [(s_["exp_avg"].to("cpu", copy=True), s_["exp_avg_sq"].to("cpu", copy=True)) for s_ in L.opt.state.values()]
    del L, params, st, sv, vp
    import gc
    gc.collect(); torch.cuda.empty_cache()
    L2 = Learner(a.model, a.lr)
    ckpt.load(os.path.join(a.ckpt_dir, "round_0004"), L2.model, L2.opt)
    p2 = dict(L2.model.named_parameters())
    R["restart_weights_bitexact"] = all(torch.equal(p2[n].detach().cpu(), ref_w[n]) for n in ref_w)
    R["restart_weight_dtypes"] = dtypes_of(list(p2.values()))
    m2 = list(L2.opt.state.values())
    R["restart_moments_bitexact"] = len(m2) == len(ref_m) and all(
        torch.equal(m2[i]["exp_avg"].cpu(), ref_m[i][0]) and torch.equal(m2[i]["exp_avg_sq"].cpu(), ref_m[i][1]) for i in range(len(ref_m)))
    R["restart_moment_dtypes"] = dtypes_of([s_["exp_avg"] for s_ in m2] + [s_["exp_avg_sq"] for s_ in m2])
    R["restart_moment_device"] = sorted({str(s_["exp_avg"].device) for s_ in m2})
    R["peak_gpu_gib_overall"] = torch.cuda.max_memory_allocated() / 2**30
    R["note_peak"] = "torch allocator stats include vLLM sleep-mode pool; device-level reading is gpu_used_gib_device_after_steps"
    R["documented_not_measured"] = [
        "BF16 autocast matmuls accumulate in FP32 inside cuBLAS (documented torch/cuBLAS behaviour; not inspectable)",
        "SDPA attention kernel internal precision under autocast (documented: BF16 inputs, FP32 softmax accumulation)",
        "vLLM sampling log-softmax precision (documented FP32 in vLLM sampler)"]
    json.dump(R, open(os.path.join(a.out, "precision_audit.json"), "w"), indent=1)
    rows = [("checkpoint tensors (disk)", R["checkpoint_dtypes"]), ("rollout (vLLM params)", R["rollout_param_dtypes"]),
            ("learner compute (body output under autocast)", R["learner_hidden_dtype_under_autocast"]),
            ("selected-token log probs", R["lm_head_logprob_dtype"]),
            ("optimizer-owned params", R["optimizer_param_dtypes"]), ("gradients (accumulated)", R["grad_dtypes"]),
            ("AdamW exp_avg / exp_avg_sq", R["moment_dtypes"]), ("restart weights / moments", (R["restart_weight_dtypes"], R["restart_moment_dtypes"])),
            ("first-step grad norm", R["grad_norm_first_step"]), ("updates accumulate (4 vs 1 step)", R["accumulates"]),
            ("sync == bf16(learner)", R["sync_bf16_exact"]), ("restart bit-exact (w, m)", (R["restart_weights_bitexact"], R["restart_moments_bitexact"])),
            ("peak GPU GiB", round(R["peak_gpu_gib_overall"], 2))]
    with open(os.path.join(a.out, "precision_audit.md"), "w") as f:
        f.write(f"| item | measured |\n|---|---|\n")
        for k, v in rows:
            f.write(f"| {k} | `{json.dumps(v)}` |\n")
    print(json.dumps(R, indent=1))
    ver.close()


if __name__ == "__main__":
    main()
