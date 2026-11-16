# IMPLEMENTATION-LOG — how the orchestration machinery behaves here

<!-- Purpose: record how the machinery behaves in this project. Never record results here.
     Record two kinds of entry, each with a date:
     1. Harness changes (bin/, lib/, scripts/, templates/, selftest/, docs/, agent.md): symptom,
        cause, change, harness commit, verification (tests + self-tests), rule that follows.
     2. Operational lessons: Babel quirks, permission patterns, recovery procedures.
     Write in the agent.md "Writing standard" (ASD-STE100 style). Example: examples/implementation-log-example.md -->

## 2026-11-15 — study s02 start (operational)
- **Babel state.** `~/projects/qwen-interference-study` and `~/ae-qis` were absent (wipe of 2026-11-13). Re-cloned at
  `main`; `ae setup` passed (all three QOSes held). `/data/user_data/$USER/qis` rebuilt by the first job.
- **Lock inconsistency.** `uv pip install -r envs/<00>/pip-freeze.txt` fails: hydra-core 0.11.3 needs omegaconf<1.5,
  the lock pins omegaconf 2.4.0 (study 00 patched it in after the first install). Fix: `--no-deps`. Rule: install a
  freeze with `--no-deps`; it is the exact installed set, not a resolvable spec.
- **Dirty tree.** `ae submit` refuses untracked files in the project repo; commit new study folders before submitting.
- **Detached watcher.** I once started `ae-watch` with `&` inside a compound command; killed it at once. Rule: start
  `ae-watch` only as a Bash background task.
- **CPU QOS.** Measured 2026-11-15 23:20 EST: 4 of 64 account CPUs in use; the CPU env job still pended on Priority.
  `general` started the calibration job in < 1 min.
- **2026-11-15 23:48 — env rebuilt and verified.** venv-s02 installed in 11 min (job 11167515); `uv pip freeze` equals
  the lock (0-line diff); vllm 0.31.0, torch 2.13.0+cu130, transformers 5.10.4, CUDA 13.0. Model 70d244cc downloaded
  in 7 min (unauthenticated HF). Gate tests 61 passed; full suite 87 passed, 1 failed (`test_eval_extend.py::
  test_select_frozen_panel`, FileNotFoundError: a study-00 data path from before the reorganisation; unrelated).
  The job then failed on my recipe bug (`tail -5 f1 f2` is invalid on Babel's tail); fixed (`tail -n 3` per file).
- **2026-11-16 01:10 — canary passed (job 11168205, commit 0288b41).** Joint add9 d6 + countdown n4, lr 1e-6, 6 rounds.
  Round 105-130 s (gen 45-67 s, old log-probs 11-16 s, train 38-53 s) at mean length 1,040-1,530 tokens; device memory
  36 GiB (train) / 36 GiB (gen, gpu_mem_util 0.30 on a 48 GB card); grad norm 0.56-0.78, no clipping; clip fractions 0;
  per-role A0/mixed/A1 counts consistent. Forced exit after round 3 (exit 143) -> `scontrol requeue` -> restart=1 at
  00:51 resumed from round_0003, skipped the existing round-3 evals, completed round 6. Checkpoint snapshot 3.2-3.6 s
  (blocking write in this test mode). One minibatch had grad norm 0.0 (all four groups uniform): Adam then moves by
  momentum only - expected. Requeue cost ~10 min (the recipe sleeps after `scontrol requeue`).
- **2026-11-16 03:05 — storage.** `/data/user_data/vineethb` 345 GB used of 500 GB (this study ~112 GB; the rest is
  other projects on the shared username). Planned retained checkpoints could exceed the 156 GB free. Deleted the canary
  checkpoint (round_0006, test only) and the MBPP-only final (round_0032; no planned branch uses it), ~40 GB; NFS shows
  the space late. Rule: retain only checkpoints a planned branch needs; delete a run's final checkpoint after harvest
  unless an order/recovery experiment names it; check `df` before each batch that retains checkpoints.
- **2026-11-16 03:00 — runner options added (commit 102eae6):** `no_ckpt` (probes), `allow_uneven_minibatch`
  (opt-in round-robin for leftover groups; divisible roles unchanged, tested). Generated recipes now gate on
  `tests/test_s02.py` + `tests/test_grpo.py` on compute before training.

## 2026-11-16 06:19 EST — INCIDENT: /data/user_data/vineethb full (501G/501G, inodes 100%)
Symptom: three training jobs on three nodes ended at 06:19:20-21 (qis-s02-e3-add9mix-s2 11169764 status 5,
qis-s02-e3-add9d6-s1 11169765 status 5, qis-s02-e1-caesar-s2 11171036 status 1 during its round-16 checkpoint).
qis-s02-e3-add9d4-s1 (11169767) then failed at precheck: "mkdir: ... Disk quota exceeded" (06:23:38).
Evidence: diag job 11171686 `df -h` 06:22:37 EST: 501G size, 501G used, 0 avail; inodes 407,991/407,991.
Actions: held all pending study jobs (11169766, 11168361, 11168353, 11171038); read-only inventory job
qis-s02-diag-du (11171696) submitted before any deletion. The filesystem is shared with another project on the same
account; only study-s02 files may be deleted.
Rule: each s02 checkpoint is ~20 GB (FP32 model + Adam). Budget checkpoints before submitting: count retained
rounds + finals across all live runs and keep the total below free space minus a margin.
Mistake: at 06:27 I started `ae-watch` with `&` inside a compound command (forbidden). It had already exited; the
other project's watcher (different args) was not touched. Restarted correctly as a background task.
06:31 EST: prune-1 (job 11171791, recipe exp/qis-s02-prune-1.sh) deleted all s02 checkpoints except
qis-s02-e3-add9mix-s2/round_0016 (d6 = .00 at that round: collapse state). Deleted: add10-s1 r8, add9-s1 r32,
caesar-s1 r32, caesar-s2 r8 + r16.tmp (partial), add9d6-s1 r16, J(add9,add10) r8/r16/r32, J(caesar,terse) r8/r16/r32,
and run dir runs/qis-s02-e1-caesar-s2 (restarts from scratch). s02/ckpt 256G -> 20G. mix-s2 round_0024 did not exist
(its save failed at 06:19). df right after: 488G used, 13G free (unexpected; re-measuring with qis-s02-diag-df 11171795).
Fix (commit 4f40a34): recipes now (1) refuse to start with < 60G free on $PROJ_DATA, (2) delete non-retained
round_* checkpoints in the export phase. Tested on a fake tree (kept retained rounds, pruned r32). Recipes
regenerated at ea73639. Older pending recipes (terse R, add10 R at 14be5a9) lack both: prune their finals by hand.

## 2026-11-16 06:56 EST — bad node babel-l9-32
Symptom: E3b jobs 11171822, 11171823, 11171824 each failed ~20 s into `train` on babel-l9-32 with
`torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable`. Same code ran elsewhere.
Action: exclude in cluster/CARD.md, in s02 specs (`exclude` field, baeda29), and on pending jobs via
`ae slurm update <id> ExcNodeList=babel-l9-32`. One retry each (11171949, 11171953, 11171994).
Rule: the precheck's "GPU memory < 2 GB" test does not catch this fault; it shows only at CUDA context creation.

## 2026-11-16 12:37 EST — NFS write throughput collapse; reading a busy job safely
Symptom: R(caesar) lr 1e-6 (11174276) finished training and its r32 evals at 11:45 but kept running; J(add9,terse)
(11173148) wrote no metrics after 11:50 (round 24). Evidence (overlap step): the caesar run's round_0024 checkpoint
(model.pt 8.1 GB, optim.pt 13.8 GB) finished writing at 12:31-12:33, i.e. about 6 MB/s; the train process waits in
futex (saver). /data/user_data/vineethb at 336G/500G (165G free) — not full. Cause: NFS congestion (cause unknown;
three study jobs were checkpointing at once, plus other users of nas5).
Reading a busy job: `srun --jobid=<id> --overlap -n1 -c1 --mem=50M <cmd>` works within seconds; without --mem the
step waits for memory the job holds and hangs (two hung steps on 2026-11-16 04:38/04:40). `ssh <node>` from the login
node fails (host key verification).
Rule: for non-preemptible general runs whose checkpoints are not needed, set ckpt_every above rounds and no retained
rounds, so only one save happens; avoid concurrent saves.
13:04 EST: deleted retained checkpoints no longer needed (via overlap step in my job 11175477): qis-s02-e3-add9mix-s2/round_0016 (E3b done), qis-s02-e3-add9mix-s3r/round_0016 and round_0024 (not collapse states). Remaining: J(add9,terse) checkpoints (pruned by its export phase). Disk before: 320G used / 181G free.
