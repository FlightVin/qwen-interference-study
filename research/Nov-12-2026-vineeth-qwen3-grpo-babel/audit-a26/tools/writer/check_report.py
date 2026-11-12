#!/usr/bin/env python3
"""A26 writer self-check for reports/REPORT.md (stdlib only).

Checks:
  1. numbers   every number with a decimal point or a % sign in REPORT.md must appear in, or round from, a value in
               the named source files (v3 tables, audit files). Signs are ignored (a contrast may be printed with a
               sign its source omits). Rounding: half-even and half-up, 0-3 decimals, on the value and on 100 x value.
  2. equals    no '=' between two numbers.
  3. figures   every figure link exists; every PNG in reports/figures/v3 is embedded exactly once.
  4. links     every relative Markdown link target exists (anchors stripped).
  5. phrases   none of the report-v2 error phrases remain.
  6. lines     line count per main section and for the appendices.
Usage: python3 check_report.py REPO_ROOT OUT_MD
"""
import glob
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = sys.argv[2] if len(sys.argv) > 2 else None
REPORT = os.path.join(ROOT, "reports", "REPORT.md")
A26 = os.path.join(ROOT, "research", "audit-a26")

SOURCE_GLOBS = [
    "reports/tables/v3/*.md", "reports/tables/v3/t0*.json",
    "reports/figures/v3/captions.md", "reports/figures/v3/inspection.md",
    "research/DEFINITIONS.md", "research/PROTOCOL.md",
    "research/audit-a26/discrepancies.md", "research/audit-a26/comparison.md",
    "research/audit-a26/pass-a/*.md", "research/audit-a26/pass-a/data/numbers.json",
    "research/audit-a26/pass-a/data/checkpoints.json",
    "research/audit-a26/pass-b/*.md", "research/audit-a26/pass-b/data/eval_*.json",
    "research/audit-a26/pass-b/data/train_*.json",
    "research/audit-a26/transfer/*.md", "research/audit-a26/transfer/control.json",
    "research/audit-a26/tools/reconcile/*.json",
    "slurm-agentic-engineering/results/qis-a26-ckpt-k*/ckpt_k*.jsonl",
    "data/control/README.md",
]

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")
# report tokens: numbers with a decimal point or a percent sign, not part of an identifier, path or section number
REP_RE = re.compile(r"(?<![\w.§/\-])[+−-]?\d[\d,]*(?:\.\d+)+%?|(?<![\w.§/\-])[+−-]?\d[\d,]*%")


def norm(s):
    return s.replace("−", "-").replace(",", "").replace("+", "").strip()


def rounded_forms(x):
    out = set()
    if x != x or abs(x) > 1e12:          # NaN or too large to be a displayed statistic
        return out
    for v in (x, x * 100):
        d = Decimal(repr(v))
        for k in range(0, 4):
            q = Decimal(1).scaleb(-k)
            for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                r = d.quantize(q, rounding=mode)
                out.add(str(abs(r)))
            out.add(f"{abs(v):.{k}f}")
    return out


def walk_json(o, acc):
    if isinstance(o, bool) or o is None:
        return
    if isinstance(o, (int, float)):
        acc.add(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            walk_json(v, acc)
    elif isinstance(o, list):
        for v in o:
            walk_json(v, acc)
    elif isinstance(o, str):
        for m in NUM_RE.findall(o.replace("−", "-")):
            try:
                acc.add(float(m.replace(",", "")))
            except ValueError:
                pass


def load_sources():
    files, literal, values = [], set(), set()
    for g in SOURCE_GLOBS:
        for p in sorted(glob.glob(os.path.join(ROOT, g))):
            files.append(os.path.relpath(p, ROOT))
            text = open(p, encoding="utf-8").read()
            literal.update(norm(m).lstrip("-") for m in re.findall(r"[\d][\d,]*(?:\.\d+)+", text))
            if p.endswith(".json"):
                walk_json(json.loads(text), values)
            elif p.endswith(".jsonl"):
                for line in text.splitlines():
                    if line.strip():
                        walk_json(json.loads(line), values)
            for m in NUM_RE.findall(text.replace("−", "-")):
                try:
                    values.add(float(m.replace(",", "")))
                except ValueError:
                    pass
    forms = set(literal)
    for v in values:
        forms |= rounded_forms(v)
    return files, forms


def main():
    text = open(REPORT, encoding="utf-8").read()
    lines = text.splitlines()
    files, forms = load_sources()
    res = {}

    # 1. numbers
    # section numbers of the report's own headings (e.g. 5.4.4) are references, not data
    sections = set(re.findall(r"^#+ (\d+(?:\.\d+)+)", text, flags=re.M))
    unmatched, n_checked, n_sections = [], 0, 0
    for i, line in enumerate(lines, 1):
        for m in REP_RE.finditer(line):
            tok = m.group(0)
            core = norm(tok.rstrip("%")).lstrip("-")
            if core in sections and not tok.endswith("%"):
                n_sections += 1
                continue
            n_checked += 1
            if core not in forms:            # multi-dot tokens (versions) can only match literally
                unmatched.append((i, tok))
    res["numbers"] = (n_checked, unmatched)
    res["section_refs"] = n_sections

    # 2. '=' between numbers
    eq = [(i, l.strip()[:120]) for i, l in enumerate(lines, 1)
          if re.search(r"\d[ \t]*=[ \t]*[+−-]?\d", l)]
    res["equals"] = eq

    # 3. figures
    figs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    fig_missing = [f for f in figs if not os.path.exists(os.path.join(ROOT, "reports", f))]
    pngs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "reports/figures/v3/*.png")))
    used = [os.path.basename(f) for f in figs]
    fig_count = {p: used.count(p) for p in pngs}
    res["figures"] = (figs, fig_missing, fig_count)

    # 4. links
    bad_links = []
    for i, line in enumerate(lines, 1):
        for tgt in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", line):
            if tgt.startswith(("http://", "https://", "#")):
                continue
            path = tgt.split("#")[0]
            if path and not os.path.exists(os.path.normpath(os.path.join(ROOT, "reports", path))):
                bad_links.append((i, tgt))
    res["links"] = bad_links

    # 5. phrases
    phrases = ["five finished trained policies", "between rounds 32 and 64", "does not rescue", "learning signal",
               "without updating the model", "What we learned", "Four practical answers"]
    hits = [(p, i) for p in phrases for i, l in enumerate(lines, 1) if p.lower() in l.lower()]
    hits += [("3 points", i) for i, l in enumerate(lines, 1) if re.search(r"(?<![\d.,])3 points", l)]
    res["phrases"] = hits

    # 6. line counts per section
    heads = [(i, l) for i, l in enumerate(lines, 1) if re.match(r"^## ", l)]
    counts = []
    for k, (i, l) in enumerate(heads):
        end = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(lines)
        counts.append((l[3:], end - i + 1))
    app_start = next(i for i, l in heads if l.startswith("## Appendix A"))
    sec1 = next(i for i, l in heads if l.startswith("## 1."))
    res["lines"] = (counts, app_start - sec1, len(lines) - app_start + 1, len(lines))

    out = []
    w = out.append
    w("# Writer self-check for reports/REPORT.md (report v3)")
    w("")
    w("Purpose: record the result of the writer's automatic checks on `reports/REPORT.md` (A26 writer task). Script:")
    w("`research/audit-a26/tools/writer/check_report.py` (stdlib only). Run: `python3 research/audit-a26/tools/writer/"
      "check_report.py . research/audit-a26/writer-selfcheck.md` from the repository root.")
    w("")
    w("## 1. Summary")
    w("")
    w("| Check | Result |")
    w("|---|---|")
    n, um = res["numbers"]
    w(f"| Numbers with a decimal point or % traced to a source | {n - len(um)} of {n} matched; {len(um)} unmatched "
      f"({res['section_refs']} section references such as 5.4.4 skipped) |")
    w(f"| `=` between two numbers | {len(res['equals'])} found |")
    figs, fm, fc = res["figures"]
    w(f"| Figure links that exist | {len(figs) - len(fm)} of {len(figs)} |")
    w(f"| PNG files in `reports/figures/v3/` embedded exactly once | {sum(1 for v in fc.values() if v == 1)} of {len(fc)} |")
    w(f"| Relative links whose target is missing | {len(res['links'])} |")
    w(f"| Report-v2 error phrases still present | {len(res['phrases'])} |")
    c, main_lines, app_lines, total = res["lines"]
    w(f"| Main text (sections 1–6) lines | {main_lines} (target about 450–600) |")
    w(f"| Appendix lines / total lines | {app_lines} / {total} |")
    w("")
    w("Limit of check 1: a match means the number occurs in, or rounds from, some value in the named sources; it does not")
    w("prove that the number belongs to the quantity the sentence names. Short numbers such as 4.2 match by coincidence")
    w("easily. The writer took every number from the source cited beside it; this check catches typos and invented")
    w("values, not misattribution.")
    w("")
    w("## 2. Unmatched numbers")
    w("")
    if um:
        w("| Line | Token |")
        w("|---|---|")
        for i, t in um:
            w(f"| {i} | `{t}` |")
    else:
        w("None.")
    w("")
    w("## 3. Equations, figures, links, phrases")
    w("")
    w("- `=` between numbers: " + ("none." if not res["equals"] else "; ".join(f"line {i}: `{s}`" for i, s in res["equals"])))
    w("- Missing figure files: " + ("none." if not fm else ", ".join(fm)))
    w("- Figure use count: " + ", ".join(f"`{k}` {v}" for k, v in fc.items()) + ".")
    w("- Missing link targets: " + ("none." if not res["links"] else "; ".join(f"line {i}: `{t}`" for i, t in res["links"])))
    w("- Phrases checked: \"five finished trained policies\", \"between rounds 32 and 64\", \"does not rescue\", "
      "\"learning signal\", \"without updating the model\", \"What we learned\", \"Four practical answers\", "
      "\"3 points\" (not preceded by a digit or decimal point). Hits: "
      + ("none." if not res["phrases"] else "; ".join(f"\"{p}\" line {i}" for p, i in res["phrases"])))
    w("")
    w("## 4. Lines per section")
    w("")
    w("| Section | Lines |")
    w("|---|---|")
    for h, k in c:
        w(f"| {h} | {k} |")
    w("")
    w("## 5. Sources searched for numbers")
    w("")
    w(f"{len(files)} files: " + ", ".join(f"`{f}`" for f in files) + ".")
    w("")
    report = "\n".join(out) + "\n"
    if OUT:
        open(OUT, "w", encoding="utf-8").write(report)
    print(report)
    bad = len(um) + len(res["equals"]) + len(fm) + len(res["links"]) + len(res["phrases"]) + \
        sum(1 for v in fc.values() if v != 1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
