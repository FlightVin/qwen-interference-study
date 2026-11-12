# Count wrong-scored attempts whose separate \boxed{} values together match a multi-value gold (sealed test).
import gzip, json, os, re, sys, collections as C
from transformers import AutoTokenizer
P = "/data/user_data/vineethb/qis"
tok = AutoTokenizer.from_pretrained(P + "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
T = json.load(open(P + "/data/main-v1/target_test_sealed.json"))["pools"]["test"]
LIST = {"intermediate_algebra/test/00025", "intermediate_algebra/test/00171", "intermediate_algebra/test/00872",
        "intermediate_algebra/test/00888", "intermediate_algebra/test/00594", "intermediate_algebra/test/00891",
        "intermediate_algebra/test/00048", "intermediate_algebra/test/00482"}
def boxes(s):
    out, i = [], 0
    while True:
        j = s.find("\\boxed{", i)
        if j < 0: return out
        k, d = j + 7, 1
        while k < len(s) and d:
            d += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        out.append(s[j + 7:k - 1]); i = k
norm = lambda x: re.sub(r"\s|\\left|\\right|\\dfrac", "", x).replace("\\frac", "frac")
gold = {t["id"]: t.get("gold", t.get("answer")) for t in T}
res = C.Counter()
for root in ["test-eval", "test-eval-cgpu"]:
    for name in sorted(os.listdir(f"{P}/{root}")):
        f = f"{P}/{root}/{name}/records.jsonl.gz"
        if not os.path.exists(f): continue
        for r in map(json.loads, gzip.open(f, "rt")):
            pid = r["prompt_id"]
            if not any(pid.endswith(x) for x in LIST) or r["reward"] == 1 or r["finish"] == "length": continue
            bx = boxes(tok.decode(r["ids"], skip_special_tokens=True))
            g = {norm(x) for x in gold[pid].split(",")} if "\\pm" not in gold[pid] else None
            vals = {norm(v) for b in bx for v in b.split(",")}
            if len(bx) >= 2 and g and g <= vals and len(vals) == len(g):
                res[(root, name, pid[-5:])] += 1
for k, v in sorted(res.items()): print(*k, v)
print("total", sum(res.values()))
