"""A26 pass B task 6b: response audit counted from the raw per-read files (laptop, stdlib only).

Inputs: research/evidence/a17/audit/{key.json, ann_0..3.json}, research/evidence/a17/panels/audit_sample_v1.json,
data/stress/stress_v1.json. The key's reward/finish/n_tokens were checked against the stress records on compute
(eval_stress.py: 96 entries, 0 mismatches). Does not read audit_merged.json or audit.md.
Usage: python3 audit_reads.py REPO OUT.json
"""
import collections
import hashlib
import json
import sys

repo, outp = sys.argv[1], sys.argv[2]
A = repo + "/research/evidence/a17/audit"
key = json.load(open(A + "/key.json"))
K = key["key"]
reads = collections.defaultdict(list)
for i in range(4):
    d = json.load(open(f"{A}/ann_{i}.json"))
    for code, v in d["responses"].items():
        reads[code].append({"reader": d["reader"], **v})

out = {"responses_in_key": len(K), "responses_read": len(reads), "reads_total": sum(len(v) for v in reads.values()),
       "reads_per_response": dict(collections.Counter(len(v) for v in reads.values())),
       "codes_read_not_in_key": sorted(set(reads) - set(K)), "codes_in_key_not_read": sorted(set(K) - set(reads)),
       "item_mismatch_read_vs_key": sum(1 for c, rs in reads.items() for r in rs if r["item"] != K[c]["item"])}
out["by_policy"] = dict(collections.Counter(v["policy"] for v in K.values()))
out["by_tier"] = dict(collections.Counter(v["tier"] for v in K.values()))
r0 = [c for c, v in K.items() if v["reward"] == 0]
out["reward0"] = len(r0)
out["reward0_capped"] = sum(K[c]["finish"] == "length" for c in r0)
out["reward1"] = sum(v["reward"] == 1 for v in K.values())
out["reward0_by_policy"] = dict(collections.Counter(K[c]["policy"] for c in r0))
out["reward0_capped_by_policy"] = dict(collections.Counter(K[c]["policy"] for c in r0 if K[c]["finish"] == "length"))

cats = sorted({r["category"] for rs in reads.values() for r in rs})
out["read_categories_all"] = dict(collections.Counter(r["category"] for rs in reads.values() for r in rs))
out["read_categories_reward0"] = dict(collections.Counter(r["category"] for c in r0 for r in reads[c]))
out["read_categories_reward1"] = dict(collections.Counter(r["category"] for c, v in K.items() if v["reward"] == 1 for r in reads[c]))
# agreement per response
agree = [c for c, rs in reads.items() if len(rs) == 2 and rs[0]["category"] == rs[1]["category"]]
out["agreement_category_all"] = {"agree": len(agree), "of": len(reads)}
out["agreement_category_reward0"] = {"agree": sum(c in agree for c in r0), "of": len(r0)}
out["agreement_category_reward0_uncapped"] = {"agree": sum(c in agree for c in r0 if K[c]["finish"] != "length"),
                                               "of": sum(K[c]["finish"] != "length" for c in r0)}
fu = [c for c, rs in reads.items() if len(rs) == 2 and rs[0]["factoring_used"] == rs[1]["factoring_used"]]
out["agreement_factoring_used_all"] = {"agree": len(fu), "of": len(reads)}
out["agreement_factoring_used_reward0"] = {"agree": sum(c in fu for c in r0), "of": len(r0)}
pairs = collections.Counter(tuple(sorted(r["category"] for r in reads[c])) for c in r0)
out["reward0_category_pairs"] = {" | ".join(k): v for k, v in sorted(pairs.items(), key=lambda x: -x[1])}
# consensus: categories agreed by both reads among reward-0
out["reward0_agreed_category"] = dict(collections.Counter(reads[c][0]["category"] for c in r0 if c in agree))
# consistency of reads with the key
out["reads_capped_flag_vs_key"] = {
    "agree": sum(1 for c, rs in reads.items() for r in rs if bool(r["capped"]) == (K[c]["finish"] == "length")),
    "of": out["reads_total"]}
out["reward1_reads_not_correct"] = [(c, K[c]["policy"], K[c]["item"], r["category"]) for c, v in K.items() if v["reward"] == 1
                                    for r in reads[c] if r["category"] != "correct"]
out["reward0_reads_correct"] = [(c, K[c]["policy"], K[c]["item"], r["category"]) for c in r0 for r in reads[c]
                                if r["category"] == "correct"]
gi = [c for c in r0 if any(r["category"] == "grader_issue" for r in reads[c])]
out["reward0_with_any_grader_issue_read"] = [(c, K[c]["policy"], K[c]["item"], [r["category"] for r in reads[c]]) for c in gi]
out["reward0_excluding_grader_issue_any_read"] = len(r0) - len(gi)
gi2 = [c for c in r0 if all(r["category"] == "grader_issue" for r in reads[c])]
out["reward0_excluding_grader_issue_both_reads"] = len(r0) - len(gi2)
# capped reward-0 responses: how readers classified them
out["capped_reward0_read_categories"] = dict(collections.Counter(r["category"] for c in r0 if K[c]["finish"] == "length" for r in reads[c]))

# audit sample rule: per tier, first 8 by sha256('audit-v1:'+id)
items = json.load(open(repo + "/data/stress/stress_v1.json"))["pools"]["stress"]
samp = json.load(open(repo + "/research/evidence/a17/panels/audit_sample_v1.json"))["items"]
rule = {}
for tier in ("superhard", "harder"):
    ids = sorted((it["id"] for it in items if it["tier"] == tier), key=lambda i: hashlib.sha256(("audit-v1:" + i).encode()).hexdigest())
    rule[tier] = {"recomputed_first8": ids[:8], "file": samp[tier], "same_set": set(ids[:8]) == set(samp[tier]),
                  "same_order": ids[:8] == samp[tier]}
out["sample_rule_check"] = rule
out["key_items_equal_sample"] = {v["item"] for v in K.values()} == set(samp["superhard"] + samp["harder"])
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "sample_rule_check"}, indent=1))
print({t: (v["same_set"], v["same_order"]) for t, v in rule.items()})
