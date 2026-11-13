#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:20:00
# HYPOTHESIS: none -- study Nov-12-2026-01 setup. Read-only inventory: stored checkpoints, optimizer files, cached models of
# other Qwen3 sizes, free space, timing of generation and training per round, outbound network from a compute node.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"

phase list "test -s $AE_RESULTS/inventory.txt" '
    {
      echo "== ckpt"; ls -la "$PROJ_DATA"/ckpt/*/* 2>&1 | head -120
      echo "== hf hub"; ls "$PROJ_DATA/hf/hub" 2>&1
      echo "== shared hf cache (qwen)"; ls /data/hf_cache 2>&1 | head -5; ls /data/hf_cache/hub 2>/dev/null | grep -i qwen | head -40; ls /data/hf_cache 2>/dev/null | grep -i qwen | head -40
      echo "== /data/models"; ls /data/models 2>&1 | grep -i qwen | head -20
      echo "== df"; df -h /data/user_data/$USER; du -sh "$PROJ_DATA" 2>/dev/null
      echo "== PROJ_DATA top"; ls "$PROJ_DATA"
      echo "== network"; timeout 20 curl -sI https://huggingface.co | head -1 || echo "no outbound https"
      echo "== control base summary"; "$VENV/bin/python" -c "
import json,glob
for p in sorted(glob.glob(\"$PROJ_DATA/control-eval/*/summary.json\")):
    s=json.load(open(p)); print(p.split(\"/\")[-2], s[\"pass1\"], s[\"completions\"], round(s[\"seconds\"]), s[\"tokens\"])
"
      echo "== round timing R-s101"; "$VENV/bin/python" -c "
import json
rows=[json.loads(l) for l in open(\"$PROJ_DATA/runs/qis-main-R-s101-sym/metrics.jsonl\")]
r=[x for x in rows if x.get(\"kind\")==\"round\" or \"t_gen\" in x]
import statistics as st
for k in (\"t_gen\",\"t_oldlp\",\"t_train\",\"t_round\",\"len_mean\"):
    v=[x[k] for x in r if k in x]; print(k, len(v), round(st.mean(v),1), round(max(v),1))
print(sorted(rows[0].keys())[:40])
"
      echo "== diag records"; ls -la "$PROJ_DATA"/diag* 2>&1 | head -30
      echo "== node mem"; free -g | head -2
    } > "$AE_RESULTS/inventory.txt" 2>&1
'
