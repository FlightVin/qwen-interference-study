"""A26: generated and prompt tokens per run from the 128 canonical training record files (stdlib only).
Response tokens = len(ids) per attempt; prompt tokens counted once per attempt if a prompt-length field exists."""
import gzip, json, os, sys
P = "/data/user_data/vineethb/qis/runs"
out = {}
for run in sys.argv[1:]:
    d = os.path.join(P, run, "records")
    gen = n = 0; keys = None; ptok = 0; pfield = None
    for r in range(128):
        with gzip.open(os.path.join(d, f"train_r{r:04d}.jsonl.gz"), "rt") as f:
            for line in f:
                x = json.loads(line)
                if keys is None:
                    keys = sorted(x)
                    pfield = next((k for k in ("prompt_tokens", "n_prompt", "prompt_len") if k in x), None)
                gen += len(x["ids"]); n += 1
                if pfield: ptok += x[pfield]
    out[run] = {"attempts": n, "gen_tokens": gen, "prompt_field": pfield, "prompt_tokens": ptok if pfield else None, "keys": keys}
print(json.dumps(out, indent=1))
