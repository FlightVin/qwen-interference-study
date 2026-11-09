"""A1.2 report figures (run in a CPU job; matplotlib Agg). Missing inputs skip their figure and are listed.

Inputs: report_data JSON (qis.analysis.report_data), optional A9 table JSON (qis.analysis.report_tables), optional
local-update diagnostic files diag_r0.jsonl / diag_r64.jsonl. Outputs PNG files in OUT_DIR and figures.json (which
figures were drawn and from what).
- main_result.png: target_test pass@1 for H, R, U per seed (+ base model line); paired R-U and R-H per seed and mean
  with 95% problem-resampling intervals (conditional on the trained policies).
- learning_rounds.png / learning_exposure.png: dev pass@1 vs round and vs cumulative hard-prompt responses, each seed
  drawn separately; the cap change after round 32 is marked.
- pope_R.png / pope_U.png: (J_R, J_H) and (J_U, J_H) per run on common axes; start marked; arrows between successive
  scheduled evaluations.
- local_update.png: change in J_H (and J_R, J_U) after one AdamW/GRPO step per branch, per minibatch seed, per source.
Usage: python -m qis.analysis.figures REPORT_DATA.json OUT_DIR [--table T.json] [--diag D1.jsonl D2.jsonl ...]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

ARM_COLOR = {"H": "#4c72b0", "R": "#dd8452", "U": "#55a868"}
SEED_STYLE = {101: "-", 202: "--"}
CAP_ROUND = 32


def run_key(run):
    parts = run.split("-")                                              # qis-main-R-s101-sym
    return parts[2], int(parts[3][1:])


def ev(run, panel):
    return [(e["round"], e["pass1"]) for e in run["evals"] if e["panel"] == panel]


def learning(data, out, drawn):
    for name, xkey, xlabel in (("learning_rounds", "round", "training round"),
                               ("learning_exposure", "hard", "cumulative hard-prompt training responses")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for run, v in sorted(data["runs"].items()):
            arm, seed = run_key(run)
            pts = ev(v, "dev")
            if not pts:
                continue
            if xkey == "round":
                xs = [r for r, _ in pts]
            else:
                xs = [v["exposure"].get(str(r), v["exposure"].get(r, {})).get("cum_hard_responses", 0) if r else 0
                      for r, _ in pts]
            ax.plot(xs, [p for _, p in pts], SEED_STYLE.get(seed, ":"), marker="o", color=ARM_COLOR[arm],
                    label=f"{arm} seed {seed}")
        if xkey == "round":
            ax.axvline(CAP_ROUND, color="grey", lw=0.8, ls=":")
            ax.text(CAP_ROUND + 1, ax.get_ylim()[0], " cap 8,192 → 12,288", fontsize=7, color="grey", va="bottom")
        ax.set_xlabel(xlabel); ax.set_ylabel("target_dev pass@1 (53 problems × 8 samples)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, name + ".png"), dpi=150); plt.close(fig)
        drawn.append(name)


def pope(data, out, drawn):
    lims = []
    for v in data["runs"].values():
        for p in ("probe_J_H", "probe_J_R", "probe_J_U"):
            lims += [x for _, x in ev(v, p)]
    if not lims:
        return
    lo, hi = min(lims) - 0.03, max(lims) + 0.03
    for aux in ("R", "U"):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for run, v in sorted(data["runs"].items()):
            arm, seed = run_key(run)
            jx, jh = dict(ev(v, f"probe_J_{aux}")), dict(ev(v, "probe_J_H"))
            rs = sorted(set(jx) & set(jh))
            if not rs:
                continue
            xs, ys = [jx[r] for r in rs], [jh[r] for r in rs]
            ax.plot(xs, ys, SEED_STYLE.get(seed, ":"), color=ARM_COLOR[arm], lw=1, label=f"{arm} seed {seed}")
            ax.plot(xs[0], ys[0], "s", color=ARM_COLOR[arm], ms=5)
            for i in range(len(rs) - 1):
                ax.annotate("", (xs[i + 1], ys[i + 1]), (xs[i], ys[i]),
                            arrowprops={"arrowstyle": "->", "color": ARM_COLOR[arm], "lw": 1})
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"J_{aux}: pass@1 on the fixed {'related' if aux == 'R' else 'unrelated'} probe panel")
        ax.set_ylabel("J_H: pass@1 on the fixed 32-problem hard-dev panel")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"pope_{aux}.png"), dpi=150); plt.close(fig)
        drawn.append(f"pope_{aux}")


def main_result(table, out, drawn):
    T = json.load(open(table))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for i, row in enumerate(T["rows"]):
        for j, arm in enumerate("HRU"):
            if row[arm] != "missing":
                a1.bar(i * 4 + j, row[arm], color=ARM_COLOR[arm], label=arm if i == 0 else None)
    if T.get("base_pass1") not in (None, "missing"):
        a1.axhline(T["base_pass1"], color="black", lw=0.8, ls="--", label="base model")
    a1.set_xticks([1, 5]); a1.set_xticklabels([f"seed {r['seed']}" for r in T["rows"]])
    a1.set_ylabel("target_test pass@1 (128 problems × 32 samples)"); a1.legend(fontsize=7)
    labels, k = [], 0
    for name in ("R-U", "R-H"):
        for row in T["rows"] + [dict(T["mean"], seed="mean")]:
            x = row.get(name)
            if isinstance(x, dict):
                a2.errorbar(k, x["est"], yerr=[[x["est"] - x["lo"]], [x["hi"] - x["est"]]], fmt="o", color="black", capsize=3)
            labels.append(f"{name}\n{row['seed']}"); k += 1
    a2.axhline(0, color="grey", lw=0.8)
    a2.set_xticks(range(k)); a2.set_xticklabels(labels, fontsize=7)
    a2.set_ylabel("paired difference in pass@1 (95% problem bootstrap)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "main_result.png"), dpi=150); plt.close(fig)
    drawn.append("main_result")


def local_update(paths, out, drawn):
    rows = []
    for p in paths:
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p) if l.strip().startswith("{")]
    labels = sorted({r["label"] for r in rows})
    if not labels:
        return
    fig, axes = plt.subplots(1, len(labels), figsize=(4.8 * len(labels), 4.0), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        R = [r for r in rows if r["label"] == lab]
        none = next((r for r in R if r["branch"] == "none"), None)
        if none is None:
            continue
        for k, br in enumerate(("hard", "related", "unrelated")):
            for panel, dx, mk in (("J_H", -0.2, "o"), ("J_R", 0, "^"), ("J_U", 0.2, "s")):
                key = next((p for p in none["eval"] if p.endswith(panel)), None)
                if key is None:
                    continue
                ds = [r["eval"][key]["pass1"] - none["eval"][key]["pass1"] for r in R if r["branch"] == br]
                ax.plot([k + dx] * len(ds), ds, mk, color="black" if panel == "J_H" else "grey", ms=4,
                        label=panel if k == 0 else None)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["hard", "related", "unrelated"])
        ax.set_title(f"source {lab}", fontsize=9); ax.set_ylabel("change in pass@1 after one update (vs no update)")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "local_update.png"), dpi=150); plt.close(fig)
    drawn.append("local_update")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--table"); ap.add_argument("--diag", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, drawn = json.load(open(a.data)), []
    learning(data, a.out, drawn)
    pope(data, a.out, drawn)
    if a.table and os.path.exists(a.table):
        main_result(a.table, a.out, drawn)
    local_update(a.diag, a.out, drawn)
    json.dump({"drawn": drawn, "inputs": {"data": a.data, "table": a.table, "diag": a.diag}},
              open(os.path.join(a.out, "figures.json"), "w"), indent=1)
    print("drawn:", drawn)


if __name__ == "__main__":
    main()
