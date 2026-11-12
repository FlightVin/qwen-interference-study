"""A26 pass A (agent a-evals): assemble the evaluation map (pass-a/data/eval_map.json).

Inputs:
  SCAN_DIR/scan.json                 scan_evals.py output (record counts, summary fields, panel hashes, shard checks)
  SCAN_DIR/sacct_evals_D.txt         `sacct -D -j <eval jobs> -o JobID,JobName%28,State,Start,End,NodeList -P` (times EST)
  SCAN_DIR/node_gres.txt             `sinfo -N -h -o "%N %G"` (node -> GPU type)
  SCAN_DIR/remote-results/<expid>/<sha-expsha>/gpu.txt, ckpt_identity.txt   (copied from ~/ae-qis/results)
  REPO/slurm-agentic-engineering/state/LEDGER.jsonl
Usage: python3 build_eval_map.py SCAN_DIR REPO OUT_JSON
"""
import glob
import json
import os
import sys
from collections import defaultdict

SD, REPO, OUT = sys.argv[1:4]
scan = json.load(open(f"{SD}/scan.json"))
gres = dict(line.split()[:2] for line in open(f"{SD}/node_gres.txt") if line.strip())
alloc = defaultdict(list)
for line in open(f"{SD}/sacct_evals_D.txt"):
    f = line.strip().split("|")
    if len(f) < 6 or f[0] == "JobID":
        continue
    alloc[f[0]].append({"state": f[2], "start_est": f[3], "end_est": f[4], "node": f[5], "gpu": gres.get(f[5], "unknown").split(":")[1] if f[5] in gres else None})
ledger = defaultdict(list)
for line in open(f"{REPO}/slurm-agentic-engineering/state/LEDGER.jsonl"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    ledger[r["expid"]].append({"jobid": str(r["jobid"]), "sha": r.get("sha", "")[:8], "expsha": r.get("expsha", "")[:12], "submitted_utc": r.get("submitted")})


def results(expid):
    out = []
    for d in sorted(glob.glob(f"{SD}/remote-results/{expid}/*/")):
        g = open(d + "gpu.txt").read().splitlines()[1].split(",")[0] if os.path.exists(d + "gpu.txt") else None
        ci = open(d + "ckpt_identity.txt").read().strip() if os.path.exists(d + "ckpt_identity.txt") else None
        out.append({"run_dir": os.path.basename(d.rstrip("/")), "gpu_txt": g, "ckpt_identity": ci})
    return out


def jobs(expid):
    out = []
    for j in ledger.get(expid, []):
        out.append({**j, "allocations": alloc.get(j["jobid"], [])})
    return out


P = "/data/user_data/vineethb/qis"
TEST = {"file": f"{P}/data/main-v1/target_test_sealed.json", "sha256": scan["panels"]["test"]["sha256"], "problems": 128}
STRESS = {"file": f"{P}/data/stress/stress_v1.json", "sha256": scan["panels"]["stress"]["sha256"], "problems": 221}
CUE = {"files": [scan["panels"]["cue_uncued"]["path"], scan["panels"]["cue_cued"]["path"]],
       "sha256": [scan["panels"]["cue_uncued"]["sha256"], scan["panels"]["cue_cued"]["sha256"]], "problems": 24}
CK = {"base": ("base", "Qwen/Qwen3-1.7B snapshot 70d244cc (HF revision; no model.pt)"),
      "H-s101": (f"{P}/ckpt/qis-main-H-s101-sym/round_0128", None), "H-s202": (f"{P}/ckpt/qis-main-H-s202-sym/round_0128", None),
      "H-s202-fb": (f"{P}/ckpt/qis-main-H-s202-sym-fallback/ckpt", "copy of a2-r60 round 116 (ROUND.txt = 116)"),
      "R-s101": (f"{P}/ckpt/qis-main-R-s101-sym/round_0128", None), "R-s202": (f"{P}/ckpt/qis-main-R-s202-sym/round_0128", None),
      "U-s101": (f"{P}/ckpt/qis-main-U-s101-sym/round_0128", None), "U-s202": (f"{P}/ckpt/qis-main-U-s202-sym/round_0128", None),
      "R-s101-r64": (f"{P}/ckpt/qis-main-R-s101-sym/round_0064", "a2, saved 2026-11-08 12:15Z (DEFINITIONS §3.B; not re-read here)"),
      "H-s202-r64": (f"{P}/ckpt/qis-main-H-s202-sym/round_0064", "trajectory a2-r60; state.json saved_utc 2026-11-11T03:26:38Z (read by traces.py)")}
ANALYSIS = {
    "test": "src/qis/analysis/a17_test.py with --cgpu-root; output research/evidence/a17/analysis/test-cgpu/*.json committed at 934a1ea "
            "(code revision that ran is not recorded in the output; a17_test.py last changed at dde81a5 before 934a1ea)",
    "stress": "src/qis/analysis/a17_stress.py; output research/evidence/a17/analysis/stress/stress.json committed at 1a9b49b (a17_stress.py last changed d3fa9d5)",
    "length_test": "src/qis/analysis/a19_length.py --panel-key test --sets all=all; output research/evidence/a19/length/test/length.json committed at 86bbe77",
    "length_stress": "src/qis/analysis/a19_length.py; outputs research/evidence/a19/length/{t1,full}/length.json committed at d3dbd22 (t1) and 6360e1f (full)",
    "cue": "src/qis/analysis/cue.py at 6d4e33a (inside job qis-cue-eval); output research/evidence/a15/cue_analysis.json committed at 65d53a6",
    "diag": "diag_local.py writes diag_r*.jsonl; table by src/qis/analysis/a17_train.py (--diag); outputs research/evidence/a15/diag_r*.jsonl (08ed26e) and "
            "research/evidence/a17/analysis/train/{branches,delta_j_table}* committed at fd94374",
    "audit": "research/evidence/a17/audit/build.py packets; reads in ann_*.json merged to audit_merged.json (4029ee2)",
}
E = []


def ev(label, kind, ck, expids, panel, n, label_seed, cap, used_in, analysis, shards=None, note=""):
    s = scan["evals"].get(label, {})
    sf = s.get("summary_fields", {})
    rec = s.get("records", {})
    E.append({"evaluation": label, "kind": kind, "checkpoint": CK[ck][0], "checkpoint_note": CK[ck][1],
              "model_pt_sha256_recorded": sf.get("ckpt_model_sha256"),
              "hash_source": "summary.json ckpt_model_sha256 (written by evaluate_ckpt from state.json file_sha256); job ckpt_identity.txt compares it with an on-disk sha256 at job start; not re-hashed by a-evals",
              "panel": panel, "n": n, "cap": cap, "seed_label": label_seed, "seed_key": f"eval/{'cue' if kind == 'cue' else kind}/<id>/{label_seed}/n{n}",
              "records": s.get("path"), "records_sha256": s.get("records_sha256"), "summary_sha256": s.get("summary_sha256"),
              "record_lines": s.get("lines"), "unknown": rec.get("unknown_n"), "truncated": rec.get("truncated_n"),
              "attempts_per_problem_ok": not s.get("wrong_n") and not s.get("dup_or_gap_samples"),
              "seeds_match_formula": s.get("n_seed_mismatch") == 0 and not s.get("seed_not_single"),
              "summary_matches_records": s.get("summary_vs_records") == [],
              "pass1": rec.get("pass1"), "jobs": [{"expid": x, "runs": jobs(x), "results": results(x)} for x in expids],
              "shards": shards, "analysis": analysis, "report_sections": used_in, "note": note})


ev("test/base", "test", "base", ["qis-test-base"], TEST, 32, "test-final", 12288, ["opening 1", "Q1", "2", "4.1 bands (attempts 0–15)", "4.2 (attempts 16–31)"], ANALYSIS["test"],
   note="Earlier ledger rows 11079315 and 11079548 were cancelled before start; job 11089196 (779c54d) produced the records.")
for k, ex in (("H-s101", "qis-test-H-s101-sym"), ("R-s101", "qis-test-R-s101-sym"), ("R-s202", "qis-test-R-s202-sym"),
              ("U-s101", "qis-test-U-s101-sym"), ("U-s202", "qis-test-U-s202-sym")):
    ev(f"test/{k}", "test", k, [ex], TEST, 32, "test-final", 12288, ["opening 1–3", "Q1", "Q2", "2", "4", "5.2 (original scores and capped attempts)"], ANALYSIS["test"])
sh = scan["shards"]["test/H-s202"]
ev("test/H-s202", "test", "H-s202", [f"qis-test-H-s202-sym-k{i}" for i in range(6)], TEST, 32, "test-final", 12288, ["opening 1", "Q1", "2", "4.2", "4.3"], ANALYSIS["test"],
   shards={"split": "eval_merge --split 6: shard k gets panel items[k::6] (22,22,21,21,21,21 problems)", "merge": "eval_merge by the last shard (k5 job 11120595 wrote merge.log)",
           "check": {k: v for k, v in sh.items() if k != "shards"},
           "per_shard": {k: {"records": v["lines"], "problems": v["problems_seen"], "panel_ids_equal_split": v["panel_ids_equal_split"],
                             "seed_mismatch": v["n_seed_mismatch"], "summary_vs_records": v["summary_vs_records"], "pass1": v["pass1"]} for k, v in sh["shards"].items()}},
   note="k0 11119210 and k1 11119212 (f9b294b) ran; ledger rows 11119213/5/6/7 for k2–k5 were cancelled before start and resubmitted as 11120634 (k2, 815a9a6) and 11120590/1/5 (k3–k5, b77f066).")
fb = scan["shards"]["test/H-s202-fb"]
ev("test/H-s202-fb", "test", "H-s202-fb", [f"qis-test-H-s202-fb-k{i}" for i in range(6)], TEST, 32, "test-final", 12288, ["2 (round-116 fallback paragraph)"], ANALYSIS["test"],
   shards={"split": "items[k::6]", "merge": "eval_merge by shard k5 (job 11118851)", "check": {k: v for k, v in fb.items() if k != "shards"}})
for k in ("R-s101", "R-s202", "U-s101", "U-s202"):
    ev(f"cgpu/{k}", "test", k, [f"qis-cgpu-{k}"], TEST, 32, "test-final", 12288, ["opening 1", "2 (common-GPU tables)"], ANALYSIS["test"],
       note="Preempted and requeued; the final COMPLETED allocation only ran the done-predicate (seconds). gpu.txt reflects the last allocation; every allocation was on an L40S node.")
for k, ck, ex in (("base", "base", "qis-stress-base"), ("H-s101-r128", "H-s101", "qis-stress-H-s101-r128"), ("R-s101-r128", "R-s101", "qis-stress-R-s101-r128"),
                  ("U-s101-r128", "U-s101", "qis-stress-U-s101-r128"), ("R-s202-r128", "R-s202", "qis-stress-R-s202-r128"), ("U-s202-r128", "U-s202", "qis-stress-U-s202-r128"),
                  ("R-s101-r64", "R-s101-r64", "qis-stress-R-s101-r64"), ("H-s202-r64", "H-s202-r64", "qis-stress-H-s202-r64")):
    ev(f"stress/{k}", "stress", ck, [ex], STRESS, 8, "stress-v1", 12288, ["opening 5", "Q2", "Q3", "5.3/5.4 (original scores)", "7.2"], ANALYSIS["stress"])
ss = scan["shards"]["stress/H-s202-r128"]
ev("stress/H-s202-r128", "stress", "H-s202", [f"qis-stress-H-s202-r128-k{i}" for i in range(3)], STRESS, 8, "stress-v1", 12288, ["opening 5", "7.2"], ANALYSIS["stress"],
   shards={"split": "items[k::3] (74,74,73 problems)", "merge": "eval_merge by shard k2 (job 11121110)", "check": {k: v for k, v in ss.items() if k != "shards"}},
   note="Ledger rows 11119925/6/7 cancelled before start; ran as 11120596 (k0), 11120598 (k1), 11121110 (k2).")
for pol in ("base", "R-s101", "R-s202", "U-s101", "U-s202"):
    for form in ("uncued", "cued"):
        ev(f"cue/{pol}-{form}", "cue", pol, ["qis-cue-eval"], CUE, 8, "cue-v1", 12288, ["8"], ANALYSIS["cue"],
           note="One job (11104738, 6d4e33a, babel-p9-20 L40S) ran all ten cells; the earlier row 11102440 was cancelled before start.")
# length extension
LX = []
for kind, ex_fmt, root in (("test", "qis-lentest-{}", f"{P}/length-test/{{}}-r128/all"), ("stress", "qis-len-{}", P + "/length-diag/{}-r128/(t1|rest)")):
    for k in ("H-s101", "R-s101", "R-s202", "U-s101", "U-s202"):
        L = scan["length"][f"{kind}/{k}"]
        LX.append({"evaluation": f"length-{kind}/{k}", "checkpoint": CK[k][0], "model_pt_sha256_recorded": {p: s.get("ckpt_model_sha256") for p, s in L["summaries"].items()},
                   "panel": TEST if kind == "test" else STRESS, "from_cap": 12288, "to_cap": 16384,
                   "seed_key": f"extend/{kind}/<id>/{'extend-test-v1' if kind == 'test' else 'extend-v1'}/s<sample>",
                   "rows": root.format(k), "original_records": "test-eval/qis-test-<run>-sym" if kind == "test" else f"stress-eval/{k}-r128",
                   "checks": {x: L[x] for x in ("rows", "orig_attempts", "rows_cover_all_original", "duplicates", "capped_original",
                                                "capped_original_with_ntokens_ne_12288", "n_capped_not_extended", "n_bad")},
                   "gpu_from_summary": {p: s.get("gpu") for p, s in L["summaries"].items()},
                   "jobs": [{"expid": ex_fmt.format(k), "runs": jobs(ex_fmt.format(k)), "results": results(ex_fmt.format(k))}],
                   "analysis": ANALYSIS["length_test" if kind == "test" else "length_stress"]})
# diagnostic
DG = {"evaluation": "local one-step diagnostic", "panels": "J_H (32), J_R (28), J_U (32) from bundle panels; n 16; cap 8,192 (diag_local.py:50,126)",
      "seed_label": "r7064 (EVAL_ROUND_LABEL)", "records": f"{P}/diag-local/{{r0,r64}}/records/<branch>/eval_probe_J_*_r7064.jsonl.gz",
      "record_checks": scan["diag"],
      "jobs": [{"expid": x, "runs": jobs(x), "results": results(x)} for x in ("qis-diag-local-r0", "qis-diag-local-r64", "qis-diag-ref")],
      "checkpoints": {"r0": "base (fresh AdamW)", "r64": f"{P}/ckpt/qis-main-R-s101-sym/round_0064 (saved optimizer); sha 6fe89540… recorded in manifest"},
      "analysis": ANALYSIS["diag"], "report_sections": ["opening 4", "6"],
      "note": "r0 final job 11102499 (594d408, babel-w5-24 L40); r64 final job 11101908 (4b47386, babel-x5-20 RTX 6000 Ada); no-update reruns job 11105174 (042cebd, babel-o5-28 L40S)."}
out = {"generated_by": "research/audit-a26/pass-a/tools/a-evals/build_eval_map.py", "panels": {"test": TEST, "stress": STRESS, "cue": CUE},
       "evaluations": E, "length_extension": LX, "diagnostic": DG,
       "hash_note": "a-evals did not hash or load checkpoints. Hashes are the recorded values (summary.json) and each job's own ckpt_identity.txt "
                    "comparison made at job start; the manifest research/evidence/a17/manifest/manifest.json lists the same values."}
json.dump(out, open(OUT, "w"), indent=1)
print("evaluations", len(E), "length", len(LX))
for e in E:
    gp = sorted({a["gpu"] for j in e["jobs"] for r in j["runs"] for a in r["allocations"] if a["state"] in ("COMPLETED", "PREEMPTED") and a["node"] != "None assigned"})
    print(e["evaluation"], e["record_lines"], e["unknown"], e["attempts_per_problem_ok"], e["seeds_match_formula"], e["summary_matches_records"], gp,
          [r["gpu_txt"] for j in e["jobs"] for r in j["results"]])
