# D144 arm-blind response audit: build packets (attempt 0 per policy) with opaque codes; key stays separate.
import gzip, json, random, hashlib, os, sys
from transformers import AutoTokenizer
P = "/data/user_data/vineethb/qis"
OUT = os.path.expanduser("~/ae-qis/audit")
tok = AutoTokenizer.from_pretrained(P + "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
sample = json.load(open(sys.argv[1]))["items"]
items = {it["id"]: it for it in json.load(open(P + "/data/stress/stress_v1.json"))["pools"]["stress"]}
POL = ["base", "H-s101-r128", "R-s101-r128", "U-s101-r128", "R-s202-r128", "U-s202-r128"]
want = set(sample["superhard"] + sample["harder"])
resp = {}
for pol in POL:
    for line in gzip.open(f"{P}/stress-eval/{pol}/records.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["prompt_id"] in want and int(r["sample"]) == 0:
            resp[(r["prompt_id"], pol)] = r
rng = random.Random("audit-v1-codes")
order = [x for pair in zip(sample["superhard"], sample["harder"]) for x in pair]  # 16 items, alternating tiers
groups = [order[4 * g:4 * g + 4] for g in range(4)]
key, used = {}, set()
def code():
    while True:
        c = "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ") for _ in range(4))
        if c not in used:
            used.add(c); return c
blocks = {}
for iid in order:
    it, ann = items[iid], items[iid]["annotation"]
    ann = ann if isinstance(ann, dict) else eval(ann)
    pols = POL[:]; rng.shuffle(pols)
    b = [f"## Item {iid}\n\n**Problem.**\n\n{it['problem']}\n\n**Gold answer.** `{it['gold']}`\n\n"
         f"**Reference route (annotation, not a full solution).** {ann.get('required_reasoning')}\n\n"
         f"**Key factoring step.** {ann.get('factoring_quote')}\n\n**Obstacle.** {ann.get('obstacle')}: {ann.get('obstacle_note')}\n\n"
         f"**Alternative route.** {ann.get('alt_route')}\n"]
    for pol in pols:
        r = resp[(iid, pol)]; c = code()
        key[c] = {"item": iid, "policy": pol, "tier": it["tier"], "reward": r["reward"], "finish": r["finish"], "n_tokens": r["n_tokens"]}
        text = tok.decode(r["ids"], skip_special_tokens=True)
        if len(text) > 8000:
            text = text[:5000] + f"\n\n[... {len(text) - 7500} characters omitted by the audit tool ...]\n\n" + text[-2500:]
        b.append(f"\n### Response {c}\n\nVerifier: {'CORRECT' if r['reward'] == 1 else 'WRONG'}; extracted answer `{r['extracted']}`; "
                 f"finish: {'hit the 12,288-token cap' if r['finish'] == 'length' else 'finished'}; {r['n_tokens']} tokens.\n\n````text\n{text}\n````\n")
    blocks[iid] = "".join(b)
for a in range(4):
    gi = [a, (a + 1) % 4]
    with open(f"{OUT}/packet_{a}.md", "w") as f:
        for g in gi:
            for iid in groups[g]:
                f.write(blocks[iid] + "\n")
json.dump({"groups": groups, "readers": {a: [a, (a + 1) % 4] for a in range(4)}, "key": key}, open(f"{OUT}/key.json", "w"), indent=1)
print(len(key), "responses;", [os.path.getsize(f"{OUT}/packet_{a}.md") for a in range(4)])
