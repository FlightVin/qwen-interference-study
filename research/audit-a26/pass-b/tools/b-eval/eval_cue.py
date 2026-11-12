"""A26 pass B task 6a: cue study, cued vs uncued per policy (paired by problem)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import P, boot_diff, correct, dump, integrity, load_records, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
N = 8
ORDER = [it["id"] for it in json.load(open(HERE + "/cue_v1_uncued.json"))["pools"]["cue"]]
ORDER_C = [it["id"] for it in json.load(open(HERE + "/cue_v1_cued.json"))["pools"]["cue"]]
POLS = ["base", "R-s101", "U-s101", "R-s202", "U-s202"]
out = {"problems": len(ORDER), "same_ids_both_forms": ORDER == ORDER_C, "cells": {}, "integrity": {}, "contrasts": {}}
X = {}
for pol in POLS:
    for form in ("uncued", "cued"):
        recs, dups = load_records(f"{P}/cue-eval/v1/{pol}-{form}/records.jsonl.gz")
        out["integrity"][f"{pol}-{form}"] = integrity(recs, dups, ORDER, N)
        c = np.array([sum(correct(recs[(p, j)]) for j in range(N)) for p in ORDER], float)
        t = np.array([sum(recs[(p, j)]["finish"] == "length" for j in range(N)) for p in ORDER], float)
        out["cells"][f"{pol}-{form}"] = metrics(c, t, N)
        X[(pol, form)] = (c / N, recs)
    su = X[(pol, "uncued")][1]; sc = X[(pol, "cued")][1]
    out["integrity"][f"{pol} seeds_equal_across_forms"] = sum(su[k]["seed"] == sc[k]["seed"] for k in su)
    out["contrasts"][f"{pol} cued - uncued"] = boot_diff(X[(pol, "cued")][0], X[(pol, "uncued")][0])
for s in ("s101", "s202"):
    for form in ("uncued", "cued"):
        out["contrasts"][f"R-U {s} {form}"] = boot_diff(X[(f"R-{s}", form)][0], X[(f"U-{s}", form)][0])
    gR = X[(f"R-{s}", "cued")][0] - X[(f"R-{s}", "uncued")][0]
    gU = X[(f"U-{s}", "cued")][0] - X[(f"U-{s}", "uncued")][0]
    out["contrasts"][f"cue gain R - U {s}"] = boot_diff(gR, gU)
dump("eval_cue.json", out)
