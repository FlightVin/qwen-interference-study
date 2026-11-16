"""Procedural task families of study Nov-15-2026-02 (interference dynamics).

Every problem has an exact digit-string answer graded by `qis.verify` with match="digits" (exact comparison after
removing separators and a base subscript; see `qis.verify.normalize_digits`). Items carry:
  id        "<family>/<stratum>/<split>/<index>"   (unique; the split is part of the id)
  problem   the user text (the chat template and the instruction prefix are added at render time)
  gold      the answer as a digit string
  task      family name, for example "add9"
  stratum   difficulty label inside the family, for example "d6" (6 digits)
  match     "digits"
  prefix    (optional) instruction that replaces the default USER_PREFIX of `qis.prompts`
  len_limit (optional) response-token limit: a correct answer longer than this earns reward 0 ("over_len")

Families (x and y are random digit strings with digits 0-8 and a nonzero first digit, so the SAME strings are
valid numerals in base 9 and in base 10):
  add9  d   x + y read in base 9, answer in base 9               (target family: a stated non-decimal rule)
  add7  d   x + y with digits 0-6, read in base 7, answer in base 7 (shares the non-decimal carry procedure)
  add10 d   x + y read in base 10                                (same surface as add9, the decimal procedure)
  mul10 k   product of two k-digit base-10 integers               (decimal arithmetic, a different operation)
  terse10 k product of two k-digit integers, answered with no working (prefix), reward needs len <= len_limit
            (a competing behavioural demand: short answers)
  countdown n  n numbers in 1..25 and a reachable target; answer an expression (graded by exact rational value and
            use of every number once)                                          (logic puzzle / search)
  caesar L  Caesar shift of a random L-letter string by 1..25 (exact string)  (symbolic manipulation)
  mbpp all  MBPP sanitized (427 problems), graded by running its asserts      (code generation; public, needs network)
  arc all   ARC-Challenge, 4-option questions, graded by the option letter    (factual / science knowledge; public)
Splits use disjoint named seeds: train, sel (selection / screening evaluation), conf (confirmation evaluation).
A problem whose operand pair (in either order) already occurs in the same dedup group (mul10 and terse10 form one
group) is redrawn, so no problem, commuted copy or re-prefixed copy is shared between or within splits.

Usage: python -m qis.s02.tasks --out BUNDLE.json --spec SPEC [--n-train 4000 --n-eval 64]
SPEC is a comma list of family:stratum[:n_train], for example add9:d6,add9:d8,add10:d10,terse10:k2:1500.
"""
import argparse
import hashlib
import json
import random

from qis import seeds

TERSE_PREFIX = ("Answer the following problem immediately. Do not show any working: reply with only the final "
                "answer in \\boxed{}.\n\n")
TERSE_LEN = 24          # tokens; a bare "\boxed{1234}" answer is about 8 tokens


def to_base(x: int, b: int) -> str:
    s = ""
    while x:
        s = str(x % b) + s
        x //= b
    return s or "0"


def _digits(rng, d, top):
    return str(rng.randint(1, top)) + "".join(str(rng.randint(0, top)) for _ in range(d - 1))


# Families whose problems must not share an operand pair across splits (commuted or re-prefixed copies).
DEDUP_GROUP = {"add9": "add9", "add10": "add10", "add7": "add7", "mul10": "mul", "terse10": "mul",
               "countdown": "countdown", "caesar": "caesar", "mbpp": "mbpp", "arc": "arc"}
PUBLIC = ("mbpp", "arc")        # fixed public datasets; strata "all"; split assignment by a seeded shuffle

CODE_PREFIX = ("Write a Python function for the following task. Explain briefly, then give the complete code in one "
               "```python code block at the end of your answer.\n\n")


def _countdown(rng, n):
    """n numbers in 1..25 and a target reachable with + - * / using every number once (integer intermediate values,
    target 10..999). Returns (numbers, target, one witness expression)."""
    while True:
        nums = [rng.randint(1, 25) for _ in range(n)]
        vals = [(float(x), str(x)) for x in nums]
        ok = True
        while len(vals) > 1:
            i, j = rng.sample(range(len(vals)), 2)
            (a, ea), (b, eb) = vals[i], vals[j]
            op = rng.choice("+-*/")
            if op == "+":
                v = a + b
            elif op == "-":
                v = a - b
            elif op == "*":
                v = a * b
            else:
                if b == 0 or a % b:
                    ok = False
                    break
                v = a / b
            if v <= 0 or v != int(v):
                ok = False
                break
            vals = [x for k, x in enumerate(vals) if k not in (i, j)] + [(v, f"({ea} {op} {eb})")]
        if ok and 10 <= vals[0][0] <= 999:
            return nums, int(vals[0][0]), vals[0][1]


def make(family: str, stratum: str, rng) -> dict:
    """One problem. "operands" (sorted) is the split-disjointness key together with DEDUP_GROUP."""
    k = int(stratum[1:])
    if family in ("add9", "add10"):
        x, y = _digits(rng, k, 8), _digits(rng, k, 8)
        if family == "add9":
            it = {"problem": f"The numbers {x} and {y} are written in base 9. Compute {x} + {y}. "
                             f"Give the answer in base 9.", "gold": to_base(int(x, 9) + int(y, 9), 9), "base": 9}
        else:
            it = {"problem": f"Compute {x} + {y}.", "gold": str(int(x) + int(y)), "base": 10}
    elif family == "add7":
        x, y = _digits(rng, k, 6), _digits(rng, k, 6)
        it = {"problem": f"The numbers {x} and {y} are written in base 7. Compute {x} + {y}. "
                         f"Give the answer in base 7.", "gold": to_base(int(x, 7) + int(y, 7), 7), "base": 7}
    elif family in ("mul10", "terse10"):
        x, y = rng.randint(10 ** (k - 1), 10 ** k - 1), rng.randint(10 ** (k - 1), 10 ** k - 1)
        it = {"problem": f"Compute {x} \\times {y}.", "gold": str(x * y), "base": 10}
        if family == "terse10":
            it.update(prefix=TERSE_PREFIX, len_limit=TERSE_LEN)
    elif family == "countdown":
        nums, target, wit = _countdown(rng, k)
        shown = ", ".join(map(str, nums))
        return {"problem": f"Using the numbers {shown}, each exactly once, and the operations +, -, \\times, / and "
                           f"parentheses, write an expression whose value is {target}. Put only the expression in "
                           f"\\boxed{{}}.", "gold": wit, "numbers": nums, "target": target,
                "operands": [str(target)] + sorted(map(str, nums))}
    elif family == "caesar":
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(k))
        sh = rng.randint(1, 25)
        g = "".join(chr((ord(c) - 97 + sh) % 26 + 97) for c in w)
        return {"problem": f"Apply a Caesar shift of {sh} to the string \"{w}\": replace each letter by the letter "
                           f"{sh} positions later in the alphabet, wrapping around from z to a. Give the resulting "
                           f"string.", "gold": g, "operands": [w, str(sh)]}
    else:
        raise ValueError(f"unknown family {family}")
    it["operands"] = sorted([str(x), str(y)])
    return it


MATCH = {"countdown": "countdown", "caesar": "string", "mbpp": "code", "arc": "choice"}


def public_items(family: str, master: int, n_eval: int):
    """{split: [items]} for a public dataset (needs network or a filled HF cache). Deterministic split assignment."""
    from datasets import load_dataset
    rows = []
    if family == "mbpp":
        for sp in ("train", "test", "validation", "prompt"):
            for r in load_dataset("google-research-datasets/mbpp", "sanitized", split=sp):
                tests = list(r["test_list"])
                rows.append({"src": f"mbpp/{r['task_id']}", "prefix": CODE_PREFIX,
                             "problem": r["prompt"].strip() + "\nYour code must pass this test:\n" + tests[0],
                             "gold": "", "tests": tests, "test_setup": "\n".join(r["test_imports"])})
        rows.sort(key=lambda x: int(x["src"].split("/")[1]))
        rng = random.Random(seeds.int_seed(master, "s02/mbpp/split"))
        rng.shuffle(rows)
        parts = {"sel": rows[:n_eval], "conf": rows[n_eval:2 * n_eval], "train": rows[2 * n_eval:]}
    elif family == "arc":
        ds = {sp: load_dataset("allenai/ai2_arc", "ARC-Challenge", split=sp) for sp in ("train", "test")}

        def conv(r, sp, i):
            labs = list(r["choices"]["label"])
            if labs != ["A", "B", "C", "D"]:
                return None
            opts = "\n".join(f"({l}) {t}" for l, t in zip(labs, r["choices"]["text"]))
            return {"src": f"arc/{sp}/{i}", "problem": f"{r['question']}\n{opts}\n\nThe final answer is the letter "
                                                     f"of the correct option.", "gold": r["answerKey"]}
        tr = [x for i, r in enumerate(ds["train"]) if (x := conv(r, "train", i))]
        te = [x for i, r in enumerate(ds["test"]) if (x := conv(r, "test", i))]
        rng = random.Random(seeds.int_seed(master, "s02/arc/split"))
        rng.shuffle(te)
        parts = {"train": tr, "sel": te[:n_eval], "conf": te[n_eval:2 * n_eval]}
    else:
        raise ValueError(family)
    for sp, items in parts.items():
        for i, it in enumerate(items):
            it.update(id=f"{family}/all/{sp}/{i:05d}", task=family, stratum="all", match=MATCH[family])
    return parts


SPLIT_SEED = {"train": 0, "sel": 1, "conf": 2}


def build(spec: list, n_train: int, n_eval: int, master: int = 20261115) -> dict:
    pools, seen, dropped = {}, set(), {}
    for sp in spec:
        if sp[0] in PUBLIC:
            for split, items in public_items(sp[0], master, n_eval).items():
                pools[f"{sp[0]}_all_{split}"] = items
    for split in ("train", "sel", "conf"):           # train first: later splits drop any text already used
        for sp in spec:
            fam, strat = sp[0], sp[1]
            if fam in PUBLIC:
                continue
            n = (int(sp[2]) if len(sp) > 2 else n_train) if split == "train" else n_eval
            rng = random.Random(seeds.int_seed(master, f"s02/{fam}/{strat}/{split}"))
            items, tries = [], 0
            while len(items) < n:
                tries += 1
                if tries > 50 * n:
                    raise RuntimeError(f"cannot draw {n} distinct problems for {fam}:{strat}:{split}")
                it = make(fam, strat, rng)
                key = (DEDUP_GROUP[fam], tuple(it["operands"]))
                if key in seen:
                    dropped[f"{fam}:{strat}:{split}"] = dropped.get(f"{fam}:{strat}:{split}", 0) + 1
                    continue
                seen.add(key)
                it.update(id=f"{fam}/{strat}/{split}/{len(items):05d}", task=fam, stratum=strat,
                      match=MATCH.get(fam, "digits"))
                items.append(it)
            pools[f"{fam}_{strat}_{split}"] = items
    return {"version": "s02-tasks-v1", "master_seed": master, "spec": [":".join(map(str, sp)) for sp in spec],
            "dropped_duplicates": dropped, "pools": pools}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--master", type=int, default=20261115)
    a = ap.parse_args()
    spec = [tuple(x.split(":")) for x in a.spec.split(",")]
    b = build(spec, a.n_train, a.n_eval, a.master)
    raw = json.dumps(b, sort_keys=True).encode()
    with open(a.out, "wb") as f:
        f.write(raw)
    print(json.dumps({"out": a.out, "sha256": hashlib.sha256(raw).hexdigest(),
                      "pools": {k: len(v) for k, v in b["pools"].items()},
                      "dropped_duplicates": b["dropped_duplicates"]}, indent=1))


if __name__ == "__main__":
    main()
