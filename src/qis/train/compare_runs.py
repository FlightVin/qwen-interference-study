"""Compare an uninterrupted run (A) with an interrupted+resumed run (B): committed rounds,
per-round rewards, optimizer-step counters, and final checkpoint weights."""
import json
import sys

import torch


def rounds(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["kind"] == "round":
            out.setdefault(r["round"], []).append(r)
    return out


def main(a_dir, b_dir, ckpt_a, ckpt_b, out):
    A, B = rounds(f"{a_dir}/metrics.jsonl"), rounds(f"{b_dir}/metrics.jsonl")
    res = {"rounds_A": sorted(A), "rounds_B": sorted(B),
           "B_duplicated_rounds": [r for r, v in B.items() if len(v) > 1],
           "B_missing_rounds": [r for r in A if r not in B]}
    res["success_equal"] = {r: A[r][-1]["success"] == B[r][-1]["success"] for r in A if r in B}
    res["opt_step_equal"] = {r: A[r][-1]["opt_step"] == B[r][-1]["opt_step"] for r in A if r in B}
    res["len_mean"] = {r: (A[r][-1]["len_mean"], B[r][-1]["len_mean"]) for r in A if r in B}
    wa = torch.load(f"{ckpt_a}/model.pt", map_location="cpu", weights_only=True)
    wb = torch.load(f"{ckpt_b}/model.pt", map_location="cpu", weights_only=True)
    d = [(wa[k] - wb[k]).abs().max().item() for k in wa]
    res["final_weight_maxabs_diff"] = max(d)
    res["final_weights_bitexact"] = all(torch.equal(wa[k], wb[k]) for k in wa)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
