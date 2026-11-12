"""A26 pass A (agent a-ckpt): restore-and-step checks of retained checkpoints through the PRODUCTION code path.

Audit code. It imports qis.* on purpose: the point is to test the loaders and the update exactly as training ran them
(`qis.train.ckpt.load` into `qis.train.learner.Learner`, `grpo`, `schedule`, `seeds`, as in `run.py:174-192,243-326`).
It never constructs `ckpt.Saver` and never writes under $P/ckpt or $P/runs. Outputs: one JSON file (--out).

Checks (each optional, chosen by flags):
  tied       lm_head.weight vs model.embed_tokens.weight in model.pt (bitwise), and Parameter identity after load.
  restore    ckpt.load (file sha256 + per-tensor bit checksums vs state.json) into a fresh Learner; AdamW step values.
  weights    weights-only load (model.pt, fresh sha256 while reading, checksums vs state.json) into the Learner model.
  fp         entropy fingerprint: run.py:350-352 logs, every 8th round, the mean token entropy of the round-end
             learner on the first 8 attempts of that round (`flat[:8]` = slot 0). Recomputed here from the
             checkpoint on the same records and compared with the logged `entropy_sample_mean`.
  replay     replay 0-based round R from the canonical training records (response ids, rewards; advantages
             recomputed and compared with the stored ones): pi_old = learner at round start (no grad), minibatch
             plan from seeds.rng(seed, f"minibatch/r{R}"), loss -obj/32 per attempt, clip at grad_clip (1.0),
             AdamW step. Compared with the logged `step` rows (last occurrence per minibatch) and the round row R+1.
  control    after the replay (state displaced by one round), minibatch 0 gradient norm and the fingerprint
             again: shows how large a mismatch a different state produces (discrimination of the test).
  diagrestore  qis.diag_local.snapshot before and restore after the replay; bit checksums must equal state.json.
  vllm       evaluate_ckpt.py load path: FP32 model.pt -> BF16 -> vLLM load_weights; read back EVERY vLLM
             parameter (fused qkv / gate_up rebuilt from the checkpoint) and compare bitwise.
Usage: see the recipes slurm-agentic-engineering/exp/qis-a26-replay-*.sh. --dry skips everything that needs a GPU.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch


def now():
    return time.strftime("%FT%TZ", time.gmtime())


def read_jsonl(path, limit=None):
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt") as f:
        for line in f:
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def metric_rows(path, kind):
    return [r for r in read_jsonl(path) if r.get("kind") == kind]


def pick(rows, occurrence):
    if not rows:
        return None
    if occurrence == "last":
        return rows[-1]
    if occurrence == "first":
        return rows[0]
    return rows[int(occurrence)]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


# ---------------------------------------------------------------- round inputs (pure CPU, production functions)
class Round:
    def __init__(self, cfg, pools, tok, rnd, records_path, limit=None):
        from qis import prompts, seeds
        from qis.train.schedule import round_plan
        from qis.train.run import N_PER_PROMPT
        self.rnd = rnd
        plan = round_plan(cfg["arm"], cfg["seed"], rnd, len(pools.get("hard", [])), len(pools.get("related", [])),
                          len(pools.get("calib_train", [])))
        self.items = [(role, pools[pool][i]) for role, pool, i in plan]
        self.roles = [r for r, _ in self.items]
        recs = read_jsonl(records_path, limit)
        self.groups = [[] for _ in self.items]
        bad = []
        for k, r in enumerate(recs):
            gi = r["slot"]
            exp_seed = seeds.int_seed(cfg["seed"], f"gen/r{rnd}/p{gi}")
            ok = (r["round"] == rnd and r["prompt_id"] == self.items[gi][1]["id"] and r["role"] == self.items[gi][0]
                  and r["sample"] == len(self.groups[gi]) and r["seed"] == exp_seed and k // N_PER_PROMPT == gi)
            if not ok:
                bad.append(k)
            self.groups[gi].append(r)
        self.check = {"records": len(recs), "plan_seed_order_mismatch": len(bad), "first_bad": bad[:5],
                      "records_path": records_path}
        need = [gi for gi, g in enumerate(self.groups) if g]
        self.p_ids = {gi: tok(prompts.render(tok, self.items[gi][1]["problem"]), add_special_tokens=False)["input_ids"]
                      for gi in need}
        if limit is None:
            from qis.train import grpo
            assert all(len(g) == N_PER_PROMPT for g in self.groups), "incomplete round records"
            self.rewards = torch.tensor([[r["reward"] for r in g] for g in self.groups], dtype=torch.float32)
            self.adv = grpo.group_advantages(self.rewards)
            stored = torch.tensor([[r["adv"] for r in g] for g in self.groups], dtype=torch.float32)
            self.check["adv_max_abs_diff_vs_stored"] = float((self.adv - stored).abs().max())
            self.check["groups_mixed"] = int(((self.rewards.sum(1) > 0) & (self.rewards.sum(1) < N_PER_PROMPT)).sum())
            self.mb_plan = grpo.minibatch_plan(self.roles, 4, seeds.rng(cfg["seed"], f"minibatch/r{rnd}"))
            self.check["minibatch_plan"] = self.mb_plan
            self.check["valid_tokens_plan"] = [sum(len(self.groups[gi][j]["ids"]) for gi in mb for j in range(8))
                                               for mb in self.mb_plan]


# ---------------------------------------------------------------- GPU checks
def entropy_fp(learner, rd):
    """run.py:351: mean over flat[:8] (slot 0, samples 0..7) of learner.token_entropy."""
    ent = [float(learner.token_entropy(rd.p_ids[0], o["ids"])) for o in rd.groups[0][:8]]
    return float(np.mean(ent)), ent


def adam_steps(opt):
    s = set()
    for st in opt.state.values():
        v = st.get("step")
        if v is not None:
            s.add(float(v.item()) if torch.is_tensor(v) else float(v))
    return sorted(s)


def all_finite(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def replay_round(learner, cfg, rd, n_mb, do_step=True):
    """run.py:275-321 for the first n_mb minibatches of rd.mb_plan. Returns step stats and round stats."""
    from qis.train import grpo
    from qis.train.run import N_PER_PROMPT, qtiles
    eps_lo, eps_hi, clip = cfg["eps_low"], cfg["eps_high"], cfg.get("grad_clip", 1.0)
    mbs = rd.mb_plan[:n_mb]
    need = sorted({gi for mb in mbs for gi in mb})
    t0 = time.time()
    old = {}
    with torch.no_grad():
        for gi in need:
            for j, o in enumerate(rd.groups[gi]):
                old[(gi, j)] = learner.token_logprobs(rd.p_ids_full[gi], o["ids"], train=False)
    t_old = time.time() - t0
    steps, ratios, act_pos, act_neg, n_pos, n_neg = [], [], 0.0, 0.0, 0.0, 0.0
    t0 = time.time()
    for m, mb in enumerate(mbs):
        members = [(gi, j) for gi in mb for j in range(N_PER_PROMPT)]
        tot_loss, valid_tok = torch.zeros((), device="cuda"), 0
        for gi, j in members:
            o = rd.groups[gi][j]
            lp = learner.token_logprobs(rd.p_ids_full[gi], o["ids"], train=True)
            A = float(rd.adv[gi, j])
            obj, st = grpo.response_objective(lp, old[(gi, j)], A, eps_lo, eps_hi)
            loss = -obj / len(members)
            loss.backward()
            tot_loss += loss.detach(); valid_tok += len(o["ids"])
            ratios.append(st["ratio"])
            if A > 0:
                act_pos += st["clip_active"].sum(); n_pos += st["n_tokens"]
            elif A < 0:
                act_neg += st["clip_active"].sum(); n_neg += st["n_tokens"]
        gn = learner.grad_norm_and_clip(clip)
        finite = bool(torch.isfinite(gn)) and bool(torch.isfinite(tot_loss))
        steps.append({"minibatch": m, "loss": float(tot_loss), "grad_norm_preclip": float(gn),
                      "grad_clipped": bool(gn > clip), "valid_tokens": valid_tok, "responses": len(members),
                      "finite": finite})
        if do_step and finite:
            learner.step()
            steps[-1]["adam_step_after"] = adam_steps(learner.opt)
        else:
            learner.opt.zero_grad(set_to_none=True)
        if not finite:
            break
    out = {"steps": steps, "t_oldlp": t_old, "t_train": time.time() - t0}
    if n_mb == 4:
        r = torch.cat(ratios)
        out["round"] = {"clipfrac_pos": float(act_pos / max(n_pos, 1)), "clipfrac_neg": float(act_neg / max(n_neg, 1)),
                        "ratio_q": qtiles(r)}
    return out


def compare_steps(got, ref_steps, ref_round=None):
    rows = []
    for s in got["steps"]:
        r = ref_steps.get(s["minibatch"])
        if r is None:
            rows.append({"minibatch": s["minibatch"], "ref": None}); continue
        rows.append({"minibatch": s["minibatch"], "opt_step_ref": r["opt_step"],
                     "grad_norm_replay": s["grad_norm_preclip"], "grad_norm_logged": r["grad_norm_preclip"],
                     "grad_norm_rel_diff": rel(s["grad_norm_preclip"], r["grad_norm_preclip"]) if r["grad_norm_preclip"] else
                     (0.0 if s["grad_norm_preclip"] == 0 else float("inf")),
                     "loss_replay": s["loss"], "loss_logged": r["loss"], "loss_abs_diff": abs(s["loss"] - r["loss"]),
                     "valid_tokens_replay": s["valid_tokens"], "valid_tokens_logged": r["valid_tokens"],
                     "grad_clipped_replay": s["grad_clipped"], "grad_clipped_logged": r["grad_clipped"],
                     "job_utc_logged": r["utc"]})
    out = {"steps": rows}
    if ref_round is not None and "round" in got:
        g, r = got["round"], ref_round
        out["round_row"] = {"round": r["round"], "clipfrac_pos": [g["clipfrac_pos"], r["clipfrac_pos"]],
                            "clipfrac_neg": [g["clipfrac_neg"], r["clipfrac_neg"]],
                            "ratio_q": {k: [g["ratio_q"].get(k), r["ratio_q"].get(k)] for k in r["ratio_q"]}}
    return out


def tied_file(path):
    t0 = time.time()
    if path.endswith(".safetensors") or os.path.isdir(path):
        from safetensors import safe_open
        import glob
        files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) if os.path.isdir(path) else [path]
        got = {}
        for f in files:
            with safe_open(f, "pt") as h:
                for k in ("lm_head.weight", "model.embed_tokens.weight"):
                    if k in h.keys():
                        got[k] = h.get_tensor(k)
        cfgp = os.path.join(path, "config.json") if os.path.isdir(path) else None
        tie_cfg = json.load(open(cfgp)).get("tie_word_embeddings") if cfgp and os.path.exists(cfgp) else None
    else:
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception:  # noqa: BLE001 -- fall back to a full load
            sd = torch.load(path, map_location="cpu", weights_only=True)
        got = {k: sd[k] for k in ("lm_head.weight", "model.embed_tokens.weight") if k in sd}
        tie_cfg = None
    res = {"path": path, "keys_present": sorted(got), "config_tie_word_embeddings": tie_cfg}
    if len(got) == 2:
        a, b = got["lm_head.weight"], got["model.embed_tokens.weight"]
        res.update({"shape": list(a.shape), "dtype": str(a.dtype), "bitwise_equal": bool(torch.equal(a, b)),
                    "max_abs_diff": float((a.float() - b.float()).abs().max()),
                    "n_diff": int((a != b).sum())})
    res["seconds"] = round(time.time() - t0, 1)
    return res


def vllm_load_check(model_dir, ckpt_dir):
    """evaluate_ckpt.py:44-63 load path, then a bitwise read-back of every vLLM parameter."""
    from qis.train.rollout import Rollout
    roll = Rollout(model_dir, 12288, 0.5)
    mpath = os.path.join(ckpt_dir, "model.pt")
    sd = torch.load(mpath, map_location="cpu", weights_only=True)
    params = [(n, t.to(torch.bfloat16)) for n, t in sd.items()]
    roll.llm.collective_rpc(lambda w: w.model_runner.model.load_weights(iter(params)) and True)
    want_bf16 = dict(params)
    names4 = [n for n in ("model.embed_tokens.weight", "model.norm.weight", "model.layers.0.input_layernorm.weight",
                          "model.layers.27.mlp.down_proj.weight") if n in sd]
    del sd, params

    def expected(name):
        if ".qkv_proj." in name:
            b = name.replace(".qkv_proj.", ".{}.")
            return torch.cat([want_bf16[b.format(x)] for x in ("q_proj", "k_proj", "v_proj")], 0)
        if ".gate_up_proj." in name:
            b = name.replace(".gate_up_proj.", ".{}.")
            return torch.cat([want_bf16[b.format(x)] for x in ("gate_proj", "up_proj")], 0)
        return want_bf16.get(name)

    def _readback(w):
        out = {}
        for n, p in w.model_runner.model.named_parameters():
            e = expected(n)
            if e is None:
                out[n] = {"expected": None, "shape": list(p.shape)}
                continue
            q = p.detach()
            same_shape = list(q.shape) == list(e.shape)
            eq = same_shape and bool(torch.equal(q.to(torch.bfloat16), e.to(q.device)))
            out[n] = {"shape": list(q.shape), "dtype": str(q.dtype), "equal": eq,
                      "max_abs_diff": float((q.float() - e.to(q.device).float()).abs().max()) if same_shape else None}
        return out
    got = roll.llm.collective_rpc(_readback)[0]
    eq = sum(1 for v in got.values() if v.get("equal"))
    covered = set()
    for n in got:
        if ".qkv_proj." in n:
            covered |= {n.replace(".qkv_proj.", f".{x}.") for x in ("q_proj", "k_proj", "v_proj")}
        elif ".gate_up_proj." in n:
            covered |= {n.replace(".gate_up_proj.", f".{x}.") for x in ("gate_proj", "up_proj")}
        else:
            covered.add(n)
    return {"vllm_params": len(got), "vllm_params_bitwise_equal": eq,
            "not_equal": {n: v for n, v in got.items() if not v.get("equal")},
            "ckpt_tensors": len(want_bf16), "ckpt_tensors_not_read_back": sorted(set(want_bf16) - covered),
            "evaluate_ckpt_4_tensor_check_names": names4}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True, help="$P/runs/<run>/config.json (model, bundle, arm, seed, eps)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["restore", "weights", "vllm", "tied"], required=True)
    ap.add_argument("--fp", action="append", default=[],
                    help="label|records.jsonl.gz|metrics.jsonl|round(1-based)|occurrence  (entropy fingerprint)")
    ap.add_argument("--replay-round", type=int, help="0-based round to replay")
    ap.add_argument("--replay-records")
    ap.add_argument("--ref-metrics", help="metrics file with the logged step/round rows of the replayed round")
    ap.add_argument("--ref-segment-job", help="only step/round rows written after this job's start event")
    ap.add_argument("--n-mb", type=int, default=4)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--no-step", action="store_true", help="gradient only (weights-only checkpoints: no optimizer state)")
    ap.add_argument("--diag-restore", action="store_true")
    ap.add_argument("--tied-extra", action="append", default=[], help="extra files/dirs for the tied check only")
    ap.add_argument("--dry", action="store_true", help="CPU only: inputs, plans and references; no model")
    a = ap.parse_args()
    t_start = time.time()
    cfg = json.load(open(a.run_config))
    res = {"script": "research/audit-a26/tools/ckpt_replay.py", "utc_start": now(), "args": vars(a),
           "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "code_sha": os.environ.get("AE_SHA"),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "run_config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
    state_path = os.path.join(a.ckpt, "state.json")
    meta = json.load(open(state_path)) if os.path.exists(state_path) else None
    if meta:
        res["state"] = {k: meta.get(k) for k in ("round", "counters", "attempt_id", "config_sha", "bundle_sha",
                                                  "file_sha256", "saved_utc")}
        res["config_sha_match"] = meta.get("config_sha") == res["run_config_sha"]

    def save():
        res["seconds"] = round(time.time() - t_start, 1)
        tmp = a.out + ".part"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1, default=str)
        os.replace(tmp, a.out)

    # tied embeddings (file level)
    res["tied"] = [tied_file(os.path.join(a.ckpt, "model.pt"))] if os.path.exists(os.path.join(a.ckpt, "model.pt")) else []
    for p in a.tied_extra:
        res["tied"].append(tied_file(p))
    save()
    if a.mode == "tied":
        res["utc_end"] = now(); save(); return
    if a.mode == "vllm":
        res["vllm"] = vllm_load_check(cfg["model"], a.ckpt); res["utc_end"] = now(); save(); return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    bundle_bytes = open(cfg["bundle"], "rb").read()
    res["bundle_sha_now"] = hashlib.sha256(bundle_bytes).hexdigest()
    res["bundle_sha_match"] = bool(meta) and meta.get("bundle_sha") == res["bundle_sha_now"]
    pools = json.loads(bundle_bytes)["pools"]

    fps = []
    for spec in a.fp:
        label, rec, met, rnd1, occ = spec.split("|")
        rows = [r for r in metric_rows(met, "round") if r["round"] == int(rnd1)]
        ref = pick(rows, occ)
        rd = Round(cfg, pools, tok, int(rnd1) - 1, rec, limit=8)
        fps.append((label, rd, ref, len(rows)))
    res["fingerprint_inputs"] = [{"label": l, "check": rd.check, "ref_entropy": (ref or {}).get("entropy_sample_mean"),
                                  "ref_utc": (ref or {}).get("utc"), "rows_for_round": n} for l, rd, ref, n in fps]
    rp = None
    if a.replay_round is not None:
        rp = Round(cfg, pools, tok, a.replay_round, a.replay_records)
        rp.p_ids_full = {gi: rp.p_ids[gi] for gi in rp.p_ids}
        # no --ref-metrics: mechanics only (e.g. one step after round 128: no logged reference exists)
        steps = [r for r in metric_rows(a.ref_metrics, "step") if r["round"] == a.replay_round] if a.ref_metrics else []
        rounds = [r for r in metric_rows(a.ref_metrics, "round") if r["round"] == a.replay_round + 1] if a.ref_metrics else []
        if a.ref_segment_job:
            starts = [r["utc"] for r in metric_rows(a.ref_metrics, "event")
                      if r.get("what") == "start" and str(r.get("job")) == a.ref_segment_job]
            t_seg = starts[-1] if starts else float("inf")
            steps = [r for r in steps if r["utc"] >= t_seg]
            rounds = [r for r in rounds if r["utc"] >= t_seg]
        ref_steps = {}
        for r in steps:                      # last occurrence per minibatch = canonical (resume rule)
            ref_steps[r["minibatch"]] = r
        rp.ref_steps, rp.ref_round = ref_steps, (rounds[-1] if rounds else None)
        res["replay_inputs"] = {"check": rp.check, "ref_step_rows_found": len(steps),
                                "ref_opt_steps": [ref_steps[m]["opt_step"] for m in sorted(ref_steps)],
                                "ref_valid_tokens": [ref_steps[m]["valid_tokens"] for m in sorted(ref_steps)],
                                "plan_valid_tokens_equal_logged": rp.check["valid_tokens_plan"] ==
                                [ref_steps[m]["valid_tokens"] for m in sorted(ref_steps)][:4]}
    save()
    if a.dry:
        res["dry"] = True; res["utc_end"] = now(); save(); return

    # ---- GPU part
    from qis.train import ckpt
    from qis.train.learner import Learner
    learner = Learner(cfg["model"], cfg["lr"])
    m = learner.model
    res["tied_params_after_init"] = m.lm_head.weight is m.model.embed_tokens.weight
    t0 = time.time()
    if a.mode == "restore":
        meta2 = ckpt.load(a.ckpt, learner.model, learner.opt)        # production loader (raises on any mismatch)
        res["load"] = {"loader": "qis.train.ckpt.load", "ok": True, "seconds": round(time.time() - t0, 1),
                       "round": meta2["round"], "adam_steps": adam_steps(learner.opt),
                       "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in learner.opt.param_groups]}
    else:
        h = hashlib.sha256(); buf = io.BytesIO()
        with open(os.path.join(a.ckpt, "model.pt"), "rb") as f:
            while True:
                b = f.read(1 << 26)
                if not b:
                    break
                h.update(b); buf.write(b)
        buf.seek(0)
        sd = torch.load(buf, map_location="cpu", weights_only=True)
        del buf
        sums = ckpt.tensor_checksums(sd)
        mism = [k for k in sums if not meta or meta["checksums"]["model"].get(k) != sums[k]]
        m.load_state_dict(sd, strict=True)
        del sd
        res["load"] = {"loader": "torch.load(weights_only) + load_state_dict(strict)", "ok": True,
                       "fresh_sha256": h.hexdigest(),
                       "sha_match_state": bool(meta) and h.hexdigest() == meta["file_sha256"]["model.pt"],
                       "n_checksum_mismatch_vs_state": len(mism), "seconds": round(time.time() - t0, 1)}
    res["tied_params_after_load"] = m.lm_head.weight is m.model.embed_tokens.weight
    live = ckpt.tensor_checksums(m.state_dict())
    res["live_model_checksums_equal_state"] = bool(meta) and live == meta["checksums"]["model"]
    res["live_finite"] = all_finite(m)
    save()

    snap = None
    if a.diag_restore:
        from qis.diag_local import snapshot
        snap = snapshot(learner.model, learner.opt)

    res["fingerprints"] = []
    for label, rd, ref, _ in fps:
        v, ent = entropy_fp(learner, rd)
        r = {"label": label, "entropy_ckpt": v, "per_attempt": ent, "ref_entropy": (ref or {}).get("entropy_sample_mean")}
        if r["ref_entropy"] is not None:
            r["abs_diff"] = abs(v - r["ref_entropy"]); r["rel_diff"] = rel(v, r["ref_entropy"])
        res["fingerprints"].append(r); save()

    if rp is not None:
        got = replay_round(learner, cfg, rp, a.n_mb, do_step=not a.no_step)
        res["replay"] = {"raw": got, "compare": compare_steps(got, rp.ref_steps, rp.ref_round),
                         "adam_steps_after": adam_steps(learner.opt), "finite_after": all_finite(learner.model)}
        save()
        if a.control:
            # state displaced by the replayed update(s): same records, recomputed pi_old, minibatch 0, no step
            ctl = replay_round(learner, cfg, rp, 1, do_step=False)
            r0 = rp.ref_steps.get(0)
            res["control_displaced"] = {"grad_norm": ctl["steps"][0]["grad_norm_preclip"],
                                        "grad_norm_logged_step1": r0["grad_norm_preclip"] if r0 else None,
                                        "rel_diff": rel(ctl["steps"][0]["grad_norm_preclip"], r0["grad_norm_preclip"]) if r0 else None,
                                        "fingerprints": []}
            for label, rd, ref, _ in fps:
                v, _ = entropy_fp(learner, rd)
                rv = (ref or {}).get("entropy_sample_mean")
                res["control_displaced"]["fingerprints"].append({"label": label, "entropy": v, "ref": rv,
                                                                 "rel_diff": rel(v, rv) if rv is not None else None})
            save()
    if snap is not None:
        from qis.diag_local import restore
        restore(learner.model, learner.opt, snap)
        del snap
        osum = ckpt.tensor_checksums(ckpt._opt_flat(learner.opt.state_dict()))
        res["diag_restore"] = {"loader": "qis.diag_local.snapshot/restore",
                               "model_checksums_equal_state": ckpt.tensor_checksums(learner.model.state_dict()) == meta["checksums"]["model"],
                               "optim_checksums_equal_state": osum == meta["checksums"]["optim"],
                               "adam_steps": adam_steps(learner.opt)}
    res["utc_end"] = now()
    save()


if __name__ == "__main__":
    sys.exit(main())
