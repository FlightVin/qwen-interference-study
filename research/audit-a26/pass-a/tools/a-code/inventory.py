#!/usr/bin/env python3
"""A26 pass A (a-code): job inventory from the ledger, recipes, generated job scripts and sacct.

Stdlib only; reads local files. Inputs (all copied, never modified):
  --ledger   slurm-agentic-engineering/state/LEDGER.jsonl
  --exp      recipe dirs (exp/, exp-deferred/)
  --jobs     copy of babel:~/ae-qis/jobs/ (generated job scripts, one per <expid>-<sha7>-<expsha>)
  --sacct    `sacct -j <all ids> -D -X -P -o JobID,JobName%40,State,ExitCode,NodeList,Partition,QOS,AllocTRES%80,Submit,Start,End,Elapsed,Restarts`
  --nodes    `sinfo -N -h -o "%N|%G|%f"`
Output: JSON list (one object per ledger row) + a family summary on stdout.

expsha = first 12 hex chars of sha256(exp/<expid>.sh) at submit time (bin/ae:123 exp_sha).
The generated job script holds the recipe minus '#EXP ' lines after the marker line (bin/ae:436-438).
"""
import argparse, hashlib, json, os, re, sys, collections

FAMILIES = [  # (family, regex on expid); first match wins
    ("selftest/env", r"^qis-(smoke(-dep|-gen)?|env-v1b?|probe-gpu-v1|trainer-smoke-v1|smoke-k16b?|cost)$"),
    ("tests", r"^qis-tests-"),
    ("calibration", r"^qis-(cal-|cal8k-|calib-select)"),
    ("data build/screening", r"^qis-(data-v1|screen|rescreen|select-hard|authored|export-cand|targets-v1|calib-panels|match-k8|bundle-main|stage3|lenprobe|a6-|a8-cpu|diag-trunc)"),
    ("training", r"^qis-main-"),
    ("in-run monitoring/inspection", r"^qis-(monitor|peek|boundary-watch|a11-|probe-table|train-summary|trunc-inspect|r32-provenance)"),
    ("ops/storage/recovery", r"^qis-(prune-|storage-|r101-recover|r202-recover|u202-recover|ckpt-watch|ckpt-inspect|validate-endpoints|h202-fallback-copy)"),
    ("sealed test (fallback r116)", r"^qis-test-H-s202-fb-k"),
    ("sealed test", r"^qis-test-"),
    ("cgpu reruns", r"^qis-cgpu-"),
    ("stress", r"^qis-stress-"),
    ("length", r"^qis-(len|lentest)-"),
    ("cue", r"^qis-cue-"),
    ("diag", r"^qis-(diag-local|diag-ref|interference)"),
    ("report", r"^qis-report-"),
    ("audits (A26)", r"^qis-a26-"),
]

def family(expid):
    for f, rx in FAMILIES:
        if re.search(rx, expid):
            return f
    return "other"

def sha256(b):
    return hashlib.sha256(b).hexdigest()

def body_of_local(text):
    lines = [l for l in text.splitlines() if not l.startswith("#EXP ")]
    return "\n".join(lines) + "\n"

def body_of_job(text, expid):
    marker = "# ---------------- experiment definition: %s ----------------" % expid
    i = text.find(marker)
    if i < 0:
        return None
    rest = text[i + len(marker):]
    rest = rest[1:] if rest.startswith("\n") else rest
    return rest if rest.endswith("\n") else rest + "\n"

def sbatch_meta(text):
    m = {}
    for l in text.splitlines():
        if l.startswith("#SBATCH "):
            a = l[8:].strip()
            for k, rx in (("partition", r"^-p (\S+)"), ("qos", r"^-q (\S+)"), ("gres", r"^--gres=(\S+)"),
                          ("constraint", r"^--constraint=(\S+)"), ("exclude", r"^--exclude=(\S+)"),
                          ("time", r"^-t (\S+)"), ("mem", r"^--mem=(\S+)"), ("cpus", r"^-c (\S+)")):
                mm = re.search(rx, a)
                if mm:
                    m[k] = mm.group(1)
    return m

def recipe_facts(body):
    f = {}
    exports = dict(re.findall(r'^export (\w+)="?([^"\n]*)"?', body, re.M))
    f["venv"] = exports.get("VENV") or (re.search(r"(\S*venv[-\w]*)", body).group(1) if re.search(r"venv", body) else None)
    f["model"] = exports.get("MODEL")
    f["pythonpath"] = exports.get("PYTHONPATH")
    keys = ("BUNDLE", "TEST", "PANEL", "PANEL_SHA", "CKPT", "SRC", "REC", "CUEDIR", "OUT", "OUTROOT", "FINAL", "ROOT", "W", "CK", "RUN")
    f["exports"] = {k: exports[k] for k in keys if k in exports}
    f["modules"] = sorted(set(re.findall(r"-m (qis\.[\w.]+)", body)))
    f["scripts"] = sorted(set(re.findall(r"(?:python3?\"?\s+)((?:\$PWD/|\./)?[\w./-]+\.py)", body)))
    f["ckpt_args"] = sorted(set(re.findall(r"--(?:ckpt|source|init-ckpt|init_ckpt)\s+\"?([^\s\"]+)", body)))
    f["panel_args"] = sorted(set(re.findall(r"--panel\s+\"?([^\s\"]+)", body)))
    f["seed_labels"] = sorted(set(re.findall(r"--seed-label\s+(\S+)", body)))
    f["max_tokens_args"] = sorted(set(re.findall(r"--max-tokens\s+(\d+)", body)))
    f["sha256_pins"] = sorted(set(h[:12] for h in re.findall(r"\b([0-9a-f]{64})\b", body)))
    f["attempt_id"] = sorted(set(re.findall(r'"attempt_id": "(\w+)"', body)))
    f["runs_pytest"] = "pytest" in body
    return f

def gpu_type(tres):
    m = re.search(r"gres/gpu:([^=,]+)=", tres or "")
    return m.group(1) if m else ("none" if "gres/gpu" not in (tres or "") else "untyped")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--sacct", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.ledger) if l.strip()]
    sacct = collections.defaultdict(list)
    with open(a.sacct) as fh:
        hdr = fh.readline().rstrip("\n").split("|")
        for l in fh:
            d = dict(zip(hdr, l.rstrip("\n").split("|")))
            sacct[d["JobID"].split("_")[0]].append(d)
    nodegpu = {}
    for l in open(a.nodes):
        p = l.rstrip("\n").split("|")
        if len(p) >= 2:
            nodegpu[p[0]] = p[1]

    out = []
    for r in rows:
        e, sha7, esha = r["expid"], r["sha"][:7], r["expsha"]
        o = dict(expid=e, jobid=r["jobid"], sha=r["sha"], sha7=sha7, expsha=esha, submitted=r["submitted"],
                 deps=r.get("deps", ""), family=family(e))
        local = None
        for d in a.exp:
            p = os.path.join(d, e + ".sh")
            if os.path.isfile(p):
                local = p
                break
        o["recipe_local"] = os.path.relpath(local, os.path.dirname(os.path.dirname(a.ledger))) if local else None
        ltext = open(local).read() if local else None
        o["recipe_local_hash"] = sha256(open(local, "rb").read())[:12] if local else None
        o["expsha_match"] = (o["recipe_local_hash"] == esha) if local else None
        jp = os.path.join(a.jobs, "%s-%s-%s.sh" % (e, sha7, esha))
        o["jobscript"] = os.path.basename(jp) if os.path.isfile(jp) else None
        jbody = None
        if o["jobscript"]:
            jtext = open(jp).read()
            jbody = body_of_job(jtext, e)
            o["sbatch"] = sbatch_meta(jtext)
            o["jobscript_sha_line"] = (re.search(r"sha=([0-9a-f]{40})", jtext) or [None, None])[1] if re.search(r"sha=([0-9a-f]{40})", jtext) else None
        o["jobscript_body_sha"] = sha256(jbody.encode())[:12] if jbody else None
        o["local_body_sha"] = sha256(body_of_local(ltext).encode())[:12] if ltext else None
        o["body_match_local"] = (o["jobscript_body_sha"] == o["local_body_sha"]) if (jbody and ltext) else None
        facts = recipe_facts(jbody if jbody else (body_of_local(ltext) if ltext else ""))
        facts["source"] = "jobscript" if jbody else ("local recipe" if ltext else "none")
        o["recipe"] = facts
        inst = []
        for d in sacct.get(r["jobid"], []):
            inst.append(dict(state=d["State"], exit=d["ExitCode"], node=d["NodeList"], partition=d["Partition"],
                             qos=d["QOS"], gpu=gpu_type(d["AllocTRES"]), submit=d["Submit"], start=d["Start"],
                             end=d["End"], elapsed=d["Elapsed"]))
        o["sacct_instances"] = inst
        o["final_state"] = inst[-1]["state"] if inst else "unknown (no sacct row)"
        o["gpus_used"] = sorted(set(i["gpu"] for i in inst if i["gpu"] not in ("none",) and i["node"] not in ("None assigned", "")))
        o["nodes_used"] = sorted(set(i["node"] for i in inst if i["node"] not in ("None assigned", "")))
        out.append(o)

    json.dump(out, open(a.out, "w"), indent=1)

    fam = collections.OrderedDict()
    for f, _ in FAMILIES + [("other", "")]:
        fam[f] = [o for o in out if o["family"] == f]
    print("family|rows|expids|commits|final states|gpu types|recipe local missing|expsha mismatch|jobscript missing|body!=local")
    for f, L in fam.items():
        if not L:
            continue
        st = collections.Counter(o["final_state"].split(" by ")[0] for o in L)
        g = collections.Counter(x for o in L for x in o["gpus_used"])
        print("|".join(map(str, [f, len(L), len(set(o["expid"] for o in L)), len(set(o["sha7"] for o in L)),
                                  dict(st), dict(g), sum(o["recipe_local"] is None for o in L),
                                  sum(o["expsha_match"] is False for o in L), sum(o["jobscript"] is None for o in L),
                                  sum(o["body_match_local"] is False for o in L)])))
    print("TOTAL rows", len(out), "no-sacct", sum(not o["sacct_instances"] for o in out))

if __name__ == "__main__":
    main()
