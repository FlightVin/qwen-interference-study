"""GPU probe (development only; project.md Section 1 "First session").

Checks, on one GPU:
 1. the non-thinking chat template (fixture saved);
 2. vLLM generation with the study decoding (T=1, top_p=1, no top_k, cap 4096), lengths, truncation;
 3. rollout (vLLM) vs teacher-forced (HF) log-prob agreement on the same tokens;
 4. one full-parameter AdamW step over 32 responses (micro-batch 1), peak memory, time;
 5. weight sync HF -> vLLM and proof that the rollout model changed;
 6. checkpoint save / reload equality and write time.
Writes JSON results to --out. Not a training run; problems come from MATH construction items.
"""
import argparse
import hashlib
import json
import os
import time

import torch

from qis import prompts, seeds


def q(xs, ps=(0.0, 0.5, 0.9, 0.99, 0.999, 1.0)):
    t = torch.tensor(xs, dtype=torch.float64)
    return {str(p): float(torch.quantile(t, p)) for p in ps} if len(xs) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True, help="jsonl with problem, gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0),
           "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    probs = [json.loads(line) for line in open(a.problems)][: a.n_prompts]
    rendered = [prompts.render(tok, p["problem"]) for p in probs]
    fx = prompts.template_fixture(tok)
    res["template_fixture"] = fx
    json.dump(fx, open(os.path.join(a.out, "template_fixture.json"), "w"), indent=1)
    assert fx["ok"], fx

    # ---- 2. generation
    import vllm
    from vllm import LLM, SamplingParams
    res["vllm"] = vllm.__version__
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
              enable_sleep_mode=True, max_model_len=1024 + a.max_tokens, seed=0,
              enable_prefix_caching=False)
    res["vllm_init_s"] = time.time() - t0
    sps = [SamplingParams(n=a.n, temperature=1.0, top_p=1.0, top_k=prompts.vllm_top_k_disabled(),
                          repetition_penalty=1.0, max_tokens=a.max_tokens, logprobs=0,
                          seed=seeds.int_seed(11, f"probe/gen/{i}")) for i in range(len(rendered))]
    prompt_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in rendered]
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sps)
    gen_s = time.time() - t0
    samples = []
    for i, o in enumerate(outs):
        for c in o.outputs:
            lp = [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]
            samples.append({"prompt": i, "ids": list(c.token_ids), "lp": lp,
                            "finish": c.finish_reason, "text": c.text})
    ntok = sum(len(s["ids"]) for s in samples)
    res["generation"] = {"seconds": gen_s, "responses": len(samples), "tokens": ntok,
                         "tok_per_s": ntok / gen_s, "lengths": q([len(s["ids"]) for s in samples]),
                         "truncated": sum(s["finish"] == "length" for s in samples),
                         "ends_with_eos": sum(s["ids"][-1] == tok.eos_token_id for s in samples
                                              if s["ids"]),
                         "eos_token": tok.eos_token, "eos_id": tok.eos_token_id}
    with open(os.path.join(a.out, "samples.jsonl"), "w") as f:
        for s in samples[:8]:
            f.write(json.dumps({k: s[k] for k in ("prompt", "finish", "text")}) + "\n")
    s0 = samples[0]
    probe_ids = prompt_ids[s0["prompt"]] + s0["ids"][:64]
    P0 = len(prompt_ids[s0["prompt"]])

    def prompt_lp():
        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        o = llm.generate([{"prompt_token_ids": probe_ids}], sp)[0]
        return torch.tensor([d[t].logprob for d, t in zip(o.prompt_logprobs[P0:], probe_ids[P0:])])
    pre_sync = prompt_lp()
    used = lambda: (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30  # noqa: E731
    res["gpu_used_after_gen_gib"] = used()
    llm.sleep(level=1)
    torch.cuda.empty_cache()
    res["gpu_used_after_sleep_gib"] = used()

    # ---- 3. teacher-forced log probs
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    res["dropout"] = {k: getattr(model.config, k) for k in dir(model.config) if "dropout" in k}
    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["n_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def seq_logprobs(s, train):
        ids = torch.tensor([prompt_ids[s["prompt"]] + s["ids"]], device="cuda")
        model.train(train)
        for m in model.modules():          # dropout stays off even in train mode
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = model.model(input_ids=ids).last_hidden_state
        P = len(prompt_ids[s["prompt"]])
        h = hid[0, P - 1:-1]
        lps = []
        for c in range(0, h.shape[0], 1024):   # chunked: never materialise full-vocab fp32 logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.lm_head(h[c:c + 1024]).float()
            tgt = ids[0, P + c:P + c + logits.shape[0]]
            lps.append(torch.log_softmax(logits, -1).gather(-1, tgt[:, None])[:, 0])
        return torch.cat(lps)

    diffs = []
    with torch.no_grad():
        for s in samples:
            tf = seq_logprobs(s, False)
            diffs += (tf.cpu() - torch.tensor(s["lp"])).abs().tolist()
    res["logprob_absdiff"] = q(diffs)

    # ---- 4. one optimizer step over 32 responses (micro-batch 1), toy advantages
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    batch = samples[:32]
    adv = torch.linspace(-1, 1, len(batch))
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for s, A in zip(batch, adv):
        lp = seq_logprobs(s, True)
        old = torch.tensor(s["lp"], device="cuda")
        ratio = torch.exp(lp - old)
        obj = torch.minimum(ratio * A, ratio.clamp(0.8, 1.28) * A)
        loss = -(obj.mean()) / len(batch)
        loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    before = {n: p.detach().clone() for n, p in list(model.named_parameters())[:3]}
    opt.step()
    torch.cuda.synchronize()
    res["opt_step"] = {"seconds": time.time() - t0, "grad_norm": float(gn),
                       "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "changed": {n: bool((p.detach() != before[n]).any())
                                   for n, p in list(model.named_parameters())[:3]}}

    # ---- 6. checkpoint save / reload (before sync so the state is the trained one)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    t0 = time.time()
    path = os.path.join(a.ckpt_dir, "probe.pt.tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "rng": torch.cuda.get_rng_state()}, path)
    os.replace(path, path[:-4])
    res["ckpt"] = {"write_s": time.time() - t0, "bytes": os.path.getsize(path[:-4])}
    t0 = time.time()
    st = torch.load(path[:-4], map_location="cpu", weights_only=False)
    same = all(torch.equal(st["model"][k], v.detach().cpu()) for k, v in list(model.state_dict().items())[:20])
    res["ckpt"].update(read_s=time.time() - t0, reload_equal_first20=same,
                       opt_state_tensors=len(st["opt"]["state"]))
    del st

    # ---- 5. sync weights to vLLM; generation must change
    opt.state.clear(); del opt
    torch.cuda.empty_cache()
    llm.wake_up()
    t0 = time.time()
    res["sync_method"] = prompts.sync_weights(llm, model)
    res["sync_s"] = time.time() - t0
    post = prompt_lp()
    res["post_sync_lp_change_mean_abs"] = float((post - pre_sync).abs().mean())
    res["top_k_value_used"] = prompts.vllm_top_k_disabled()
    json.dump(res, open(os.path.join(a.out, "probe.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
