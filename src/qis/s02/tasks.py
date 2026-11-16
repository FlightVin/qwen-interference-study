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
Splits use disjoint named seeds: train, sel (selection / screening evaluation), conf (confirmation evaluation).
A problem text that occurs in two splits is removed from the later split (checked; counts are recorded).

Usage: python -m qis.s02.tasks --out BUNDLE.json --spec SPEC [--n-train 4000 --n-eval 64]
SPEC is a comma list of family:stratum, for example add9:d6,add9:d8,add10:d10,terse10:k2.
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


def make(family: str, stratum: str, rng) -> dict:
    k = int(stratum[1:])
    if family in ("add9", "add10"):
        x, y = _digits(rng, k, 8), _digits(rng, k, 8)
        if family == "add9":
            return {"problem": f"The numbers {x} and {y} are written in base 9. Compute {x} + {y}. "
                               f"Give the answer in base 9.", "gold": to_base(int(x, 9) + int(y, 9), 9)}
        return {"problem": f"Compute {x} + {y}.", "gold": str(int(x) + int(y))}
    if family == "add7":
        x, y = _digits(rng, k, 6), _digits(rng, k, 6)
        return {"problem": f"The numbers {x} and {y} are written in base 7. Compute {x} + {y}. "
                           f"Give the answer in base 7.", "gold": to_base(int(x, 7) + int(y, 7), 7)}
    if family in ("mul10", "terse10"):
        x, y = rng.randint(10 ** (k - 1), 10 ** k - 1), rng.randint(10 ** (k - 1), 10 ** k - 1)
        it = {"problem": f"Compute {x} \\times {y}.", "gold": str(x * y)}
        if family == "terse10":
            it.update(prefix=TERSE_PREFIX, len_limit=TERSE_LEN)
        return it
    raise ValueError(f"unknown family {family}")


SPLIT_SEED = {"train": 0, "sel": 1, "conf": 2}


def build(spec: list, n_train: int, n_eval: int, master: int = 20261115) -> dict:
    pools, seen, dropped = {}, set(), {}
    for split in ("train", "sel", "conf"):           # train first: later splits drop any text already used
        for fam, strat in spec:
            n = n_train if split == "train" else n_eval
            rng = random.Random(seeds.int_seed(master, f"s02/{fam}/{strat}/{split}"))
            items, tries = [], 0
            while len(items) < n:
                tries += 1
                if tries > 50 * n:
                    raise RuntimeError(f"cannot draw {n} distinct problems for {fam}:{strat}:{split}")
                it = make(fam, strat, rng)
                key = (it.get("prefix", ""), it["problem"])
                if key in seen:
                    dropped[f"{fam}:{strat}:{split}"] = dropped.get(f"{fam}:{strat}:{split}", 0) + 1
                    continue
                seen.add(key)
                it.update(id=f"{fam}/{strat}/{split}/{len(items):05d}", task=fam, stratum=strat, match="digits")
                items.append(it)
            pools[f"{fam}_{strat}_{split}"] = items
    return {"version": "s02-tasks-v1", "master_seed": master, "spec": [f"{f}:{s}" for f, s in spec],
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
