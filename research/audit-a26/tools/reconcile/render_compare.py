"""Render compare.json as a markdown table (appendix of comparison.md). Usage: python3 render_compare.py compare.json"""
import json, sys
rows = json.load(open(sys.argv[1]))
def f(v):
    if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer() and abs(v) > 1.5):
        return f"{int(v):,}"
    return f"{v:+.4f}" if v < 0 or (v > 0 and v < 1 and False) else f"{v:.4f}"
def fmt(vals, kind):
    if kind == "ci":
        e, lo, hi = vals
        return f"{e:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return ", ".join(f(x) for x in vals)
print("| # | Group | Quantity | Pass A | Pass B | Agree? | Max abs diff |")
print("|---|---|---|---|---|---|---|")
for i, r in enumerate(rows, 1):
    print(f"| {i} | {r['group']} | {r['quantity']} | {fmt(r['A'], r['kind'])} | {fmt(r['B'], r['kind'])} | "
          f"{'yes' if r['agree'] else '**no**'} | {r['max_abs_diff']:.4g} |")
