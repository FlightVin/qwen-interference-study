"""a-data: print problem text and gold of flagged near-duplicate pairs (read-only, stdlib). Usage: python pair_texts.py W"""
import json, sys
P = "/data/user_data/vineethb/qis"; W = sys.argv[1]
M = {json.loads(l)["id"]: json.loads(l) for l in open(f"{P}/data/splits-v1/manifest.jsonl")}
O = [json.loads(l) for l in open(f"{W}/data/external/a17/omni_math_test.jsonl")]
def get(i):
    if i.startswith("omni/"):
        r = O[int(i[5:])]; return r["problem"], r.get("answer")
    r = M[i]; return r["problem"], f"{r['gold']} [split {r['split']}, group {r['group']} size {r['group_size']}]"
pairs = [("algebra/train/01269", "algebra/train/01291"), ("intermediate_algebra/train/00301", "intermediate_algebra/test/00383"),
         ("intermediate_algebra/train/00048", "intermediate_algebra/test/00891"), ("intermediate_algebra/train/00481", "omni/00486"),
         ("intermediate_algebra/train/00656", "omni/01374"), ("omni/01690", "omni/00682"),
         ("algebra/train/01083", "algebra/test/00353"), ("algebra/train/01571", "intermediate_algebra/test/00783"),
         ("algebra/train/00769", "algebra/train/01234"), ("intermediate_algebra/train/00107", "intermediate_algebra/test/00091"),
         ("algebra/train/00288", "algebra/test/01114")]
for a, b in pairs:
    print("=" * 20, a, "|", b)
    for i in (a, b):
        t, g = get(i); print(f"[{i}] gold={g}\n{t[:700]}\n")
