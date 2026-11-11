"""EXPLORATORY (A19 §3, D154): when did R and U diverge within a seed? No same-seed R/U checkpoint pair survives below
round 128 (only R-s101 and H-s202 keep round 64), so this uses the fixed measurements taken during training.

Inputs (a17_train outputs, TRAIN_DIR): curves.json (dev pass@1, 53 problems × 8 attempts, at rounds 0/32/64/96/128;
shared-hard success per 8-round bin) and signal.json (per 32-round window and role: generated tokens, all-correct /
all-wrong group shares). Role hard_common = the shared hard prompts, the same schedule in H, R and U of one seed, so
its length and success are comparable across arms within a seed.
Output (OUT_DIR/divergence.{json,md}): per seed, R, U (and H) side by side with R − U for: dev pass@1 per eval round;
mean response tokens on hard_common per window; hard_common success per 8-round bin. Dev intervals are not shown:
53 problems × 8 attempts give sd ≈ 0.02 per value (D50); differences under ≈ 0.04 are within that noise.
Usage: python -m qis.analysis.a19_divergence TRAIN_DIR OUT_DIR
"""
import json
import os
import sys

WINDOWS = ["r1-32", "r33-64", "r65-96", "r97-128"]
H_RUN = {"s101": "H-s101", "s202": "H-s202-r60"}


def main(train_dir, out):
    os.makedirs(out, exist_ok=True)
    cur = json.load(open(os.path.join(train_dir, "curves.json")))["runs"]
    sig = json.load(open(os.path.join(train_dir, "signal.json")))["runs"]
    res, md = {}, ["# When did R and U diverge? (A19 §3; EXPLORATORY)", ""]
    for seed in ("s101", "s202"):
        runs = {"R": f"R-{seed}", "U": f"U-{seed}", "H": H_RUN[seed]}
        dev = {arm: {e["round"]: e["pass1"] for e in cur[k]["evals"]["dev"]} if k in cur else {} for arm, k in runs.items()}
        rounds = sorted(set(dev["R"]) & set(dev["U"]))
        win = {}
        for arm, k in runs.items():
            for w in WINDOWS:
                W = sig.get(k, {}).get("windows", {}).get(w)
                g = (W or {}).get("by_role", {}).get("hard_common")
                if W and g and W.get("complete_window"):
                    win.setdefault(w, {})[arm] = {"tokens_per_response": W["gen_tokens"]["hard_common"] / (g["groups"] * 8),
                                                  "all_correct": g["frac_all_correct"], "all_wrong": g["frac_all_wrong"],
                                                  "responses": g["groups"] * 8}
        bins = {arm: cur[k]["practised_shared_hard_bins"] for arm, k in runs.items() if arm != "H" and k in cur}
        res[seed] = {"dev": {r: {arm: dev[arm].get(r) for arm in runs} for r in rounds}, "windows": win,
                     "hard_common_bins": [{"bin": a["bin"], "R": a["success"], "U": b["success"], "responses": a["responses"]}
                                          for a, b in zip(bins["R"], bins["U"])]}
        md += [f"## Seed {seed[1:]}", "", "| Dev round | R | U | R − U | H |", "|---|---|---|---|---|"]
        for r in rounds:
            h = dev["H"].get(r)
            md.append(f"| {r} | {dev['R'][r]:.3f} | {dev['U'][r]:.3f} | {dev['R'][r] - dev['U'][r]:+.3f} | {f'{h:.3f}' if h is not None else '—'} |")
        md += ["", "| Window (shared hard prompts) | Responses per arm | Tokens R | Tokens U | R − U | Tokens H | All-correct groups R / U |",
               "|---|---|---|---|---|---|---|"]
        for w in WINDOWS:
            x = win.get(w, {})
            if "R" not in x or "U" not in x:
                continue
            tr, tu = x["R"]["tokens_per_response"], x["U"]["tokens_per_response"]
            th = f"{x['H']['tokens_per_response']:,.0f}" if "H" in x else "—"
            md.append(f"| {w} | {x['R']['responses']} | {tr:,.0f} | {tu:,.0f} | {tr - tu:+,.0f} | {th} | "
                      f"{x['R']['all_correct']:.2f} / {x['U']['all_correct']:.2f} |")
        md += ["", "| Rounds | Shared-hard success R | U | R − U |", "|---|---|---|---|"]
        for b in res[seed]["hard_common_bins"]:
            md.append(f"| {b['bin']} | {b['R']:.3f} | {b['U']:.3f} | {b['R'] - b['U']:+.3f} |")
        md.append("")
    json.dump(res, open(os.path.join(out, "divergence.json"), "w"), indent=1)
    open(os.path.join(out, "divergence.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:3])
