#!/usr/bin/env python3
"""A26 pass A (a-code): distinct commits per job family and which code files changed between them.

Reads data/jobs.json (from inventory.py) and runs read-only `git diff --name-only` in the repo.
For each family: commits ordered by commit time; for each consecutive pair and for (last, HEAD), the changed
files restricted to the family's code paths. Output JSON + a text summary.
Usage: python3 revisions.py --jobs research/audit-a26/pass-a/data/jobs.json --repo . --out OUT.json
"""
import argparse, json, subprocess, collections

EVAL = ["src/qis/evaluate_ckpt.py", "src/qis/eval_merge.py", "src/qis/train/rollout.py", "src/qis/prompts.py",
        "src/qis/verify.py", "src/qis/seeds.py", "src/qis/data/math_source.py", "src/qis/train/run.py",
        "src/qis/train/grpo.py"]
TRAIN = ["src/qis/train", "src/qis/prompts.py", "src/qis/verify.py", "src/qis/seeds.py", "src/qis/data/math_source.py"]
PATHS = {
    "training": TRAIN,
    "sealed test": EVAL, "sealed test (fallback r116)": EVAL, "cgpu reruns": EVAL, "stress": EVAL,
    "cue": EVAL + ["src/qis/analysis/cue.py"],
    "length": EVAL + ["src/qis/eval_extend.py"],
    "diag": TRAIN + ["src/qis/diag_local.py", "src/qis/analysis/interference.py", "src/qis/evaluate_ckpt.py"],
    "data build/screening": ["src/qis/data", "src/qis/screen.py", "src/qis/verify.py", "src/qis/prompts.py",
                             "src/qis/train/rollout.py", "src/qis/seeds.py", "src/qis/mode_diag.py", "src/qis/diag_trunc.py"],
    "calibration": TRAIN + ["src/qis/analysis/calib_select.py"],
    "tests": ["src", "tests"],
    "report": ["src/qis/analysis"],
    "audits (A26)": ["research/audit-a26/tools"],
}


def git(repo, *a):
    return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(a.jobs))
    fam = collections.defaultdict(set)
    for o in J:
        fam[o["family"]].add(o["sha"])
    res = {}
    for f, shas in fam.items():
        paths = PATHS.get(f, ["src", "tests"])
        order = sorted(shas, key=lambda s: int(git(a.repo, "show", "-s", "--format=%ct", s).strip()))
        steps = []
        for x, y in list(zip(order, order[1:])) + [(order[-1], "HEAD")]:
            ch = [l for l in git(a.repo, "diff", "--name-only", x, y, "--", *paths).splitlines() if l]
            steps.append({"from": x[:7], "to": y[:7] if y != "HEAD" else "HEAD", "changed": ch})
        res[f] = {"paths": paths, "commits": [s[:7] for s in order], "steps": steps}
        print(f"## {f}: {len(order)} commits; paths {paths}")
        for s in steps:
            if s["changed"]:
                print(f"   {s['from']}..{s['to']}: {', '.join(c.replace('src/qis/', '') for c in s['changed'])}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
