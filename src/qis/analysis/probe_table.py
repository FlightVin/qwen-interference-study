"""Probe-panel table per run and evaluation round: J_H (derived from dev), J_R, J_U pass@1/pass@8, dev truncation.
Read-only; run in a CPU job. Usage: python -m qis.analysis.probe_table RUNS_DIR BUNDLE"""
import json
import os
import sys

from qis.analysis.wandb_companion import run_rows


def main(runs, bundle):
    jh = {it["id"] for it in json.load(open(bundle))["pools"]["probe_J_H"]}
    print("run\tround\tJ_H_p1\tJ_H_p8\tJ_R_p1\tJ_R_p8\tJ_U_p1\tJ_U_p8\tdev_trunc")
    for n in sorted(x for x in os.listdir(runs) if x.startswith("qis-main-")):
        rows = run_rows(os.path.join(runs, n), jh)
        for r in sorted(rows):
            x = rows[r]
            if "eval/dev/pass1" in x:
                vals = [x.get("eval/%s/%s" % (p, m), float("nan")) for p in ("probe_J_H", "probe_J_R", "probe_J_U") for m in ("pass1", "pass8")]
                print(n, r, *["%.3f" % v for v in vals], "%.3f" % x.get("eval/dev/truncated", float("nan")), sep="\t")


if __name__ == "__main__":
    main(*sys.argv[1:3])
