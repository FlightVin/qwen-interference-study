"""Panel of candidate domains for the domain study (04-domains-to-explore.md, study Nov-12-2026-01). No training.

Writes a panel file in the format of `qis.evaluate_ckpt` ({"pools": {key: [{"id", "problem", "gold"}]}}). Every gold
answer is an integer or one letter, so the boxed-answer grader of the project applies unchanged.

Procedural pools (seeded, n problems each):
  mult2 ... mult6   product of two k-digit integers (a hardness dial)
  add10             sum of two 6-digit integers whose digits are all in 0-8
  add9              the SAME digit strings read in base 9; the answer is in base 9 (a stated-rule conflict with add10)
  mult10s, mult9    product of two 2-digit strings with digits 0-8, read in base 10 and in base 9
Public pools (downloaded when the node has network access; skipped with a note otherwise):
  gsm8k             openai/gsm8k, config main, test split, first n problems; gold = the number after "####"
  aime24            Maxwell-Jia/AIME_2024, all 30 problems
  arc               allenai/ai2_arc, config ARC-Challenge, test split, first n; gold = the answer letter

Usage: python -m qis.interf.build_probe_panel --out FILE.json [--n 100] [--seed 20261112]
"""
import argparse
import json
import random


def to_base(x, b):
    s = ""
    while x:
        s = str(x % b) + s
        x //= b
    return s or "0"


def digits9(rng, k):
    return str(rng.randint(1, 8)) + "".join(str(rng.randint(0, 8)) for _ in range(k - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()
    pools, notes = {}, {}
    for k in (2, 3, 4, 5, 6):
        rng = random.Random(a.seed + k)
        items = []
        for i in range(a.n):
            x, y = rng.randint(10 ** (k - 1), 10 ** k - 1), rng.randint(10 ** (k - 1), 10 ** k - 1)
            items.append({"id": f"mult{k}/{i:03d}", "problem": f"Compute {x} \\times {y}.", "gold": str(x * y)})
        pools[f"mult{k}"] = items
    rng = random.Random(a.seed + 100)
    p10, p9 = [], []
    for i in range(a.n):
        x, y = digits9(rng, 6), digits9(rng, 6)
        p10.append({"id": f"add10/{i:03d}", "problem": f"Compute {x} + {y}.", "gold": str(int(x) + int(y))})
        p9.append({"id": f"add9/{i:03d}", "problem": f"The numbers {x} and {y} are written in base 9. Compute {x} + {y}. "
                   "Give the answer in base 9.", "gold": to_base(int(x, 9) + int(y, 9), 9)})
    pools["add10"], pools["add9"] = p10, p9
    rng = random.Random(a.seed + 200)
    p10, p9 = [], []
    for i in range(a.n):
        x, y = digits9(rng, 2), digits9(rng, 2)
        p10.append({"id": f"mult10s/{i:03d}", "problem": f"Compute {x} \\times {y}.", "gold": str(int(x) * int(y))})
        p9.append({"id": f"mult9/{i:03d}", "problem": f"The numbers {x} and {y} are written in base 9. Compute "
                   f"{x} \\times {y}. Give the answer in base 9.", "gold": to_base(int(x, 9) * int(y, 9), 9)})
    pools["mult10s"], pools["mult9"] = p10, p9

    def public(name, fn):
        try:
            pools[name] = fn()
            notes[name] = f"ok, {len(pools[name])} problems"
        except Exception as e:  # noqa: BLE001 -- a missing public pool is reported, not fatal
            notes[name] = f"skipped: {type(e).__name__}: {e}"[:300]

    def gsm8k():
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = []
        for i in range(a.n):
            gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
            int(float(gold))
            out.append({"id": f"gsm8k/{i:03d}", "problem": ds[i]["question"], "gold": gold})
        return out

    def aime24():
        from datasets import load_dataset
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        keys = {k.lower(): k for k in ds.column_names}
        out = [{"id": f"aime24/{i:03d}", "problem": r[keys["problem"]], "gold": str(int(r[keys["answer"]]))}
               for i, r in enumerate(ds)]
        assert len(out) == 30
        return out

    def arc():
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        out = []
        for i in range(len(ds)):
            r = ds[i]
            labs = list(r["choices"]["label"])
            if labs[:4] != ["A", "B", "C", "D"] or len(labs) != 4:
                continue
            opts = "\n".join(f"({l}) {t}" for l, t in zip(labs, r["choices"]["text"]))
            out.append({"id": f"arc/{i:04d}", "problem": f"{r['question']}\n{opts}\n\nThe final answer is the letter of "
                        "the correct option.", "gold": r["answerKey"]})
            if len(out) == a.n:
                break
        return out
    public("gsm8k", gsm8k); public("aime24", aime24); public("arc", arc)
    json.dump({"version": "probe-v1", "seed": a.seed, "notes": notes, "pools": pools}, open(a.out, "w"))
    print(json.dumps({"pools": {k: len(v) for k, v in pools.items()}, "notes": notes}, indent=1))


if __name__ == "__main__":
    main()
