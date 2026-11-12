#!/bin/bash
# A26 pass A (a-code): environment and pinned-worktree provenance. Read-only.
# 1. venv-v1: key package versions (from dist-info names) and every dist-info / site-packages entry modified after the
#    first study training job started (2026-11-07 14:07 local = 19:07Z), which would mean the env changed mid-study.
# 2. Pinned worktrees ~/ae-qis/code/<expid>-<sha7>: HEAD == ledger sha (prefix), tracked files unmodified.
# Usage: bash env_check.sh <ledger-lines-file: "expid sha" per line>
P=/data/user_data/vineethb/qis
V=$P/env/venv-v1
echo "## python: $($V/bin/python -V 2>&1)  realpath=$(readlink -f $V/bin/python)"
SP=$(ls -d $V/lib/python3*/site-packages)
echo "## key packages"
ls -d $SP/*.dist-info | sed 's#.*/##' | grep -iE '^(torch|vllm|transformers|math_verify|latex2sympy2_extended|sympy|numpy|tokenizers|flash_attn|xformers|triton|safetensors|antlr4)[-_]' | sort
echo "## site-packages entries with mtime after 2026-11-07T19:07Z (count, then up to 20)"
find $SP -maxdepth 1 -newermt '2026-11-07 19:07:00 UTC' | wc -l
find $SP -maxdepth 1 -newermt '2026-11-07 19:07:00 UTC' -printf '%TY-%Tm-%TdT%TH:%TM %p\n' | sort | head -20
echo "## newest dist-info mtime"
ls -d --time-style=+%FT%T -lt $SP/*.dist-info | head -3 | awk '{print $6, $7}'
echo "## other venvs"
ls -d $P/env/* 2>/dev/null
echo "## pinned worktrees"
while read -r e s; do
  W=$HOME/ae-qis/code/$e-${s:0:7}
  if [ ! -d "$W" ]; then echo "$e ${s:0:7} MISSING"; continue; fi
  h=$(git -C "$W" rev-parse HEAD 2>/dev/null)
  m=$(git -C "$W" status --porcelain --untracked-files=no 2>/dev/null | wc -l)
  u=$(git -C "$W" status --porcelain 2>/dev/null | grep -c '^??')
  echo "$e ${s:0:7} head_ok=$([ "$h" = "$s" ] && echo 1 || echo 0) tracked_modified=$m untracked=$u"
done < "$1"
