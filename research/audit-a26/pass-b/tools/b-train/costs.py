"""A26 pass B, b-train: GPU-hours and CPU-hours over the ledger job IDs (stdlib only).

Input 1: sacct output produced by
  bin/ae slurm sacct -j <all ledger jobids> -D -X -n -P -o JobID,JobName,State,ElapsedRaw,AllocTRES,Partition,Start,End,NodeList
Input 2: slurm-agentic-engineering/state/LEDGER.jsonl
Input 3: research/audit-a26/pass-b/data/train_segments.json and train_superseded_time.json (b-train's own outputs)
Usage: python3 costs.py SACCT_TXT LEDGER OUT_JSON DATA_DIR
Rule: each sacct allocation row (-D keeps every requeued instance) is charged ElapsedRaw x allocated GPUs (gres/gpu=N)
and x allocated CPUs (cpu=N). Category from the expid (first matching prefix below). Failed/superseded is an overlay:
(a) whole main-run jobs whose rows are all superseded (a1 attempts; jobs with no canonical metrics row),
(b) superseded rounds/evaluations inside otherwise canonical main-run jobs, estimated from the trainer's t_round and eval
seconds of the non-canonical metrics rows (lower bound: partial rounds and checkpoint time are not counted),
(c) allocation rows of other categories whose state is FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY or NODE_FAIL.
"""
import collections
import json
import re
import sys

sacct_fn, ledger_fn, out_fn, data_dir = sys.argv[1:5]
CATS = [("training_main", r"^qis-main-"),
        ("training_calibration", r"^qis-cal(8k)?-lr"),
        ("evaluation", r"^qis-(test-|cgpu-|stress-|cue-eval|len-|lentest-|validate-endpoints|a11-dev16k-)"),
        ("diagnostics", r"^qis-(diag-|a6-|a11-capcheck|a11-truncstudy|lenprobe|trunc-inspect|probe-gpu|smoke-k16|trainer-smoke)"),
        ("data_screening", r"^qis-(data-v1|screen|select-hard|match-k|rescreen|bundle-main|targets-v1|authored|export-cand|calib-panels|stage3)"),
        ("audits", r"^qis-(a26-|r32-provenance|a8-cpu|ckpt-inspect)"),
        ("analysis_reporting", r"^qis-(report-|cost|interference|train-summary|probe-table|calib-select|peek|monitor)"),
        ("infrastructure", r"^qis-(smoke|env-v1|tests-|prune-|storage-|ckpt-watch|boundary-watch|h202-fallback-copy|r101-recover|r202-recover|u202-recover|a11-supersede|a11-scan)")]
BAD = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

expid = {}
for line in open(ledger_fn):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    expid[r["jobid"]] = r["expid"]


def cat(e):
    for c, rx in CATS:
        if re.match(rx, e):
            return c
    return "uncategorised"


segs = json.load(open(f"{data_dir}/train_segments.json"))["segments"]
canon_jobs, all_seg_jobs = set(), set()
for run, ss in segs.items():
    for s in ss:
        all_seg_jobs.add(s["job"])
        if s["canonical_round_rows"] or s["canonical_eval_rows"]:
            canon_jobs.add(s["job"])
A1_JOBS = {"11078180", "11078086", "11078164"}   # start events in $P/runs/superseded/*-a1/metrics.jsonl (attempt a1)
sup_time = json.load(open(f"{data_dir}/train_superseded_time.json"))

rows = []
for line in open(sacct_fn):
    f = line.rstrip("\n").split("|")
    if len(f) < 9:
        continue
    jid, name, state, el, tres, part, start, end, nodes = f[:9]
    state = state.split()[0]
    gpus = int(re.search(r"gres/gpu=(\d+)", tres).group(1)) if "gres/gpu=" in tres else 0
    cpus = int(re.search(r"cpu=(\d+)", tres).group(1)) if "cpu=" in tres else 0
    e = expid.get(jid, "?")
    rows.append({"jobid": jid, "expid": e, "name": name, "state": state, "elapsed_s": int(el or 0), "gpus": gpus, "cpus": cpus,
                 "partition": part, "start": start, "end": end, "category": cat(e)})

tot = collections.defaultdict(lambda: {"gpu_h": 0.0, "cpu_h": 0.0, "alloc_rows": 0, "jobs": set(), "by_state_gpu_h": collections.Counter(),
                                        "by_state_cpu_h": collections.Counter()})
overlay = {"a1_jobs_gpu_h": 0.0, "main_jobs_without_canonical_rows_gpu_h": 0.0, "main_jobs_without_canonical_rows": [],
           "superseded_rounds_in_canonical_jobs_gpu_h_lower_bound": 0.0, "failed_allocations_other_categories_gpu_h": 0.0,
           "failed_allocations_other_categories_cpu_h": 0.0, "failed_allocations_other_categories": []}
for r in rows:
    g, c = r["elapsed_s"] * r["gpus"] / 3600, r["elapsed_s"] * r["cpus"] / 3600
    t = tot[r["category"]]
    t["gpu_h"] += g; t["cpu_h"] += c; t["alloc_rows"] += 1; t["jobs"].add(r["jobid"])
    t["by_state_gpu_h"][r["state"]] += g; t["by_state_cpu_h"][r["state"]] += c
    if r["category"] == "training_main":
        if r["jobid"] in A1_JOBS:
            overlay["a1_jobs_gpu_h"] += g
        elif r["jobid"] not in canon_jobs:
            overlay["main_jobs_without_canonical_rows_gpu_h"] += g
            overlay["main_jobs_without_canonical_rows"].append({k: r[k] for k in ("jobid", "expid", "state", "elapsed_s", "gpus")}
                                                               | {"has_metrics_segment": r["jobid"] in all_seg_jobs})
    elif r["state"] in BAD and r["elapsed_s"] > 0:
        overlay["failed_allocations_other_categories_gpu_h"] += g
        overlay["failed_allocations_other_categories_cpu_h"] += c
        overlay["failed_allocations_other_categories"].append({k: r[k] for k in ("jobid", "expid", "state", "elapsed_s", "gpus", "cpus", "category")})
overlay["superseded_rounds_in_canonical_jobs_detail"] = {}
for run, v in sup_time.items():
    for job, pj in v["per_job"].items():
        if job in canon_jobs:          # jobs without canonical rows are already charged in full above
            overlay["superseded_rounds_in_canonical_jobs_gpu_h_lower_bound"] += (pj["round_s"] + pj["eval_s"]) / 3600
            overlay["superseded_rounds_in_canonical_jobs_detail"][f"{run}|{job}"] = pj
# per-run canonical training GPU-h from sacct (all jobs of the run's expid, minus overlays)
per_run = collections.defaultdict(lambda: {"gpu_h_all_allocations": 0.0, "jobs": []})
for r in rows:
    if r["category"] == "training_main":
        pr = per_run[r["expid"]]
        pr["gpu_h_all_allocations"] += r["elapsed_s"] * r["gpus"] / 3600
        pr["jobs"].append({k: r[k] for k in ("jobid", "state", "elapsed_s", "gpus", "start", "end")} | {"a1": r["jobid"] in A1_JOBS, "canonical_rows": r["jobid"] in canon_jobs})
per_run_summary = {}
for e, v in per_run.items():
    run = e[len("qis-main-"):-len("-sym")]
    a1 = sum(j["elapsed_s"] / 3600 for j in v["jobs"] if j["a1"])
    nocanon = sum(j["elapsed_s"] * j["gpus"] / 3600 for j in v["jobs"] if not j["a1"] and not j["canonical_rows"])
    canon_alloc = sum(j["elapsed_s"] * j["gpus"] / 3600 for j in v["jobs"] if j["canonical_rows"])
    inside = sum((pj["round_s"] + pj["eval_s"]) / 3600 for job, pj in sup_time.get(run, {}).get("per_job", {}).items() if job in canon_jobs)
    per_run_summary[e] = {"all_allocations_gpu_h": v["gpu_h_all_allocations"], "a1_gpu_h": a1, "jobs_without_canonical_rows_gpu_h": nocanon,
                          "canonical_jobs_gpu_h": canon_alloc, "superseded_inside_canonical_jobs_gpu_h_lower_bound": inside,
                          "canonical_training_gpu_h_upper_bound": canon_alloc - inside}
out = {"rule": __doc__, "per_run_summary": per_run_summary, "categories": {k: {**{kk: vv for kk, vv in v.items() if kk not in ("jobs", "by_state_gpu_h", "by_state_cpu_h")},
                                           "n_jobs": len(v["jobs"]), "by_state_gpu_h": dict(v["by_state_gpu_h"]),
                                           "by_state_cpu_h": dict(v["by_state_cpu_h"])} for k, v in sorted(tot.items())},
       "total_gpu_h": sum(v["gpu_h"] for v in tot.values()), "total_cpu_h": sum(v["cpu_h"] for v in tot.values()),
       "alloc_rows": len(rows), "distinct_jobs": len({r["jobid"] for r in rows}), "ledger_jobids": len(expid),
       "ledger_jobids_missing_in_sacct": sorted(set(expid) - {r["jobid"] for r in rows}),
       "running_rows": [{k: r[k] for k in ("jobid", "expid", "elapsed_s")} for r in rows if r["state"] == "RUNNING"],
       "failed_superseded_overlay": overlay, "training_main_per_expid": per_run, "rows": rows}
json.dump(out, open(out_fn, "w"), indent=1, sort_keys=True, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
print(json.dumps({k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("gpu_h", "cpu_h", "n_jobs", "alloc_rows")}
                  for k, v in out["categories"].items()}, indent=0))
print("total", round(out["total_gpu_h"], 2), round(out["total_cpu_h"], 2), "rows", len(rows), "jobs", out["distinct_jobs"], "ledger", len(expid),
      "missing", out["ledger_jobids_missing_in_sacct"])
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in overlay.items() if not isinstance(v, list)})
print("no-canonical main jobs", overlay["main_jobs_without_canonical_rows"])
print("failed other", [(x["expid"], x["state"], x["elapsed_s"], x["gpus"]) for x in overlay["failed_allocations_other_categories"]])
for e, v in sorted(per_run_summary.items()):
    print(e, {k: round(x, 3) for k, x in v.items()})
for e, v in sorted(per_run.items()):
    print(e, round(v["gpu_h_all_allocations"], 2), [(j["jobid"], j["state"], round(j["elapsed_s"] / 3600, 2), j["a1"], j["canonical_rows"]) for j in v["jobs"]])
