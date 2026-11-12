# DEFINITIONS v1 → v1.1: proposed changes

Purpose: list every change that audit passes A and B and the reconcile step require in `research/DEFINITIONS.md` v1,
with the reason and the evidence. Written 2026-11-11, ~10 PM EST, by agent `reconcile`. This file does not edit
DEFINITIONS; the orchestrator applies the changes as v1.1. Row references: `discrepancies.md` (I/N/W rows) and
`comparison.md` (S/M rows). "Both passes" means pass A and pass B verified the fact independently.

## Header
**C0. Version line.** Add: "v1.1 (date, A26 reconcile): corrections from audit passes A and B; reasons in
`research/audit-a26/definitions-v1.1-changes.md`." Keep v1 text of changed rows under a "superseded" note, as §1 requires.

## §1 Code references
**C18. Diagnostic code revisions.** Replace "Local diagnostic `diag_local.py` at `4b47386`" with: "`diag_local.py`
identical at 4b47386 (r64 job 11101908) and 594d408 (r0 job 11102499); the no-update reference job 11105174 ran at
042cebd, which adds `--reference-only` (22 lines)."
Reason: pass A's `eval-map.md` said 594d408 adds 21 lines; `code.md` said the file is identical. Evidence:
`git diff 4b47386 594d408 -- src/qis/diag_local.py` is empty; `git diff 4b47386 042cebd` changes it by 22 lines (W13).

## §2 Units
**C5. Add a row "training hardware".** "GPU type per trajectory segment: H-s101 L40 (rounds 1–128); R-s101, U-s101,
U-s202 L40S; H-s202 L40 rounds 1–60, L40S 61–128; R-s202 L40 rounds 1–16, L40S 17–128." In §3.B "training seed",
replace "GPU type" in the "can still differ" list with a pointer to this row.
Reason: arm contrasts are not hardware-matched in training (I8). Evidence: `start` events in each `metrics.jsonl`
(`code_sha`, `gpu`, `job`) and `sacct` AllocTRES; both passes (`pass-a/jobs.md` §3, `pass-b/train.md` §1; M10).

## §3.A Arms, pools and panels
**C6. Panel facts.**
- "dev" and "sealed test": add "Held out by ID, exact and normalised text, and the splits rule (5-shingle Jaccard
  ≥ 0.7). Two close variants of H_train problems remain under the stress rule (masked word-3-gram Jaccard ≥ 0.5):
  dev #42 `algebra/train/01291` ↔ H_train `algebra/train/01269` (1.0; not in J_H) and sealed test
  `intermediate_algebra/test/00383` ↔ H_train `intermediate_algebra/train/00301` (0.725); `00891` ↔ `00048` (0.500,
  phrasing only). Removing them changes no conclusion (`discrepancies.md` §3.1)." Replace "after removing
  near-duplicates of test" with "after removing 5-shingle near-duplicates of test".
  Reason and evidence: `pass-a/data-inventory.md` §4, §6 item 1; reconcile sensitivity `tools/reconcile/sens_heldout.json` (W6).
- "J_R", "J_U": add levels: J_R L1 1, L2 4, L3 9, L4 10, L5 4; J_U L1 4, L2 5, L3 7, L4 8, L5 8. "Rubric-selected,
  not restricted to levels 4–5." Evidence: `pass-a/data-inventory.md` §1, §6 item 2 (W11).
- "cue study": add "21 of 24 items have a pass-1 label only (allowed by the cue rule)". Evidence: same, §2 (W11).
- "stress panels": add "MATH items come from study splits (construction 44, study_dev 8, calib 21, excluded 2, test 54);
  none is within masked-3-gram 0.5 of a study item (max 0.448). Two items are multiple choice with a numeric gold
  (`math:number_theory/train/00662`, `00656`); a response whose last box is the option letter scores 0." Evidence:
  `pass-a/data-inventory.md` §6 item 4; `tools/reconcile/sens_multibox_hits.json` (N8, W11).

**C8. "response audit set".** Replace "33 after excluding the two grader-limitation responses" with "33 after excluding
the 2 responses whose both reads are `grader_issue` (both `math:precalculus/train/00152`); an any-read rule gives 31."
Reason: the count depends on the rule (M4, W12). Evidence: `pass-b/eval.md` §8; `pass-a/numbers.md` §3.5.

**C9. "shared hard prompt slots" and §4 item 4.** Replace "in all 8 sampled rounds" with "in 128/128 rounds per seed"
and delete "(code reading; not yet checked on data)" from §4 item 4. Add: "H and R/U put the shared groups into the same
minibatch pairs in 0/128 (seed 101) and 2/128 (seed 202) rounds; the predicted minibatch plan reproduces every step's
`valid_tokens` (512/512 per run)." Evidence: `pass-a/schedules.md` §3–4, `pass-b/train.md` §1, §4 (M12, M13, I11).

## §3.B Seeds, rounds, checkpoints and attempts
**C3. "run attempt / trajectory".** Replace the segment list with per-job written and canonical rows:
H-s101 11081726 → 1–128. R-s101 11081724 wrote 1–128 (canonical 1–124), 11092945 → 125–128.
U-s101 11081725 → 1–128. H-s202 11082192 wrote 1–64 (canonical 1–60), 11112338 wrote nothing, 11112377 → 61–128.
R-s202 11081729 wrote rounds 1–19 and 3 of 4 step rows of round 20 (canonical 1–16), 11094190 → 17–128.
U-s202 11094037 wrote 1–80 (canonical 1–64), 11097811 wrote 65–81 (none canonical), 11100159 → 65–128.
Add the rule: "canonical row = last write of its key: (round) for round rows, (round, minibatch) for step rows, (panel,
round) for evaluation rows; it equals the last-resume rule. Naive round-row counts are 132 (H-s202, R-s101), 131
(R-s202), 161 (U-s202)."
Reason: v1 omits the round-20 step rows and the job without canonical rows (W4). Evidence: `pass-b/train.md` §1;
`pass-a/schedules.md` §1 (527 step rows = 512 + 12 + 3; M11).

**C4. H-s202 r64 and the a2-orig recipe.** In "run attempt", add "the stress cell 'H-s202 r64' is a2-r60
(`round_0064` saved by job 11112377 at 2026-11-11 03:26:38Z, `37963fac…`); a2-orig left only `round_0064.tmp`."
Close §5 item 7 with: "the a2-orig recipe is preserved as `~/ae-qis/jobs/qis-main-H-s202-sym-7c44fa5-97c04ad04599.sh`;
it differs from the resumed recipe (`746ea7431648`) only by the `resume_audit` phase and `retain_rounds` [] → [64]."
Reason: v1 said the recipe is not in the workspace (W5); trajectory now verified by both passes (M8, I10).
Evidence: `pass-a/jobs.md` §5; `pass-a/eval-map.md` §7; `pass-b/eval.md` §2.

**C14. "common-GPU rerun".** Add the per-run result: "rerun − original pass@1: R-s101 +0.39 [−0.81, +1.61],
R-s202 +1.34 [+0.22, +2.56] (excludes 0), U-s101 +0.12 [−0.95, +1.15], U-s202 0 (4,096/4,096 identical). The cgpu-U
jobs were preempted; their records come from an earlier allocation (p5-16, p9-20; all L40S) and the last allocation
only skipped completed phases." Reason: the R-s202 move exceeds problem-and-attempt noise, so GPU type matters for
contrasts with mixed-hardware cells (I4, I8). Evidence: `pass-b/eval.md` §3; `pass-a/eval-map.md` §2 (S4, M18).

## §3.C Metrics
**C12. "reward" — grader limitations.** Replace the single note with: "Known limitations of the last-box rule: (a) the
gold split over several boxes; (b) an earlier box verifier-equivalent to gold and a different last box; (c) an option
letter boxed last on a multiple-choice item with numeric gold. Counts on finished reward-0 attempts (project verifier):
sealed test, 7 primary cells, (b) 27 and (a) 5 of 28,672; stress r128 cells (a)+(b) 2–13 per cell of 1,768. 'Any-box
rule' = credit (a) or (b); a sensitivity only, never primary." Reason: v1 lists one case; pass B found 17 exact-string
cases; reconcile measured all three kinds (N8). Evidence: `discrepancies.md` §3.2; `tools/reconcile/sens_multibox.json`.

**C13. "pass@1" — unknown status.** Add: "Unknown status occurred only in the base sealed-test cell: 2 of 4,096
attempts (`intermediate_algebra/test/00030`, samples 8, 29). Counting them correct would raise base pass@1 by 0.049
points." Close §5 item 4. Evidence: both passes (M14); `tools/reconcile/sens_heldout.json` → `unknown_test` (W7).

**C15. "length extension".** Add the executed checks and close §5 item 6: "in all 10 row sets, every capped original
has 12,288 tokens and was extended; rows cover every attempt once; non-extended rows are unchanged; still-capped rows
end at 16,384 with reward 0 (both passes). The regrade assert is verified by code and by completed jobs, not by a
per-attempt log." Add a definition: "new tokens per recovered answer = all new tokens on the extended attempts ÷
recovered attempts (R-s202 test 20,476); recovered attempts alone used 1,767 new tokens on average." Evidence:
`pass-a/eval-map.md` §5, `pass-a/traces.md` (b), `pass-b/eval.md` §7 (W8).

**C10. New term "earliest observed ordering".** "The first trained evaluation round (32, 64, 96 or 128) at which R − U
on a panel has its round-128 sign, and separately the round from which the sign no longer changes. Round 0 is reported
beside it as the noise reference: it evaluates identical base weights with identical seeds, and its R − U (−2.3 to 0.0
points) is cross-job sampling, not an ordering. Values: seed 101 dev, J_H, J_U r64 (J_U holds from r128), J_R r128;
seed 202 dev, J_H, J_U r32, J_R r64. The cap changes between r32 and r64." Reason: pass A counted round 0 and pass B did
not (M1); report v2's "between rounds 32 and 64 in both seeds" was wrong for seed 202 (I3). Evidence: `pass-b/train.md`
§3; `pass-a/numbers.md` §3.2; reconcile `compare.json` rows 123–138.

**C19. "crossed bootstrap" and two-seed means.** Add the values: "two-seed means R − U −0.13, R − H +3.65, U − H +3.78
points; paired t with df 1 half-widths 54.8, 16.0, 38.8 points." Reason: makes concrete why the problem-resampling
interval cannot stand for run-to-run uncertainty (I4). Evidence: `pass-b/eval.md` §4.

**C16. New term "canonical training GPU-h".** "GPU-h of the allocations whose job wrote canonical rows, minus the
measured time of superseded rounds and evaluations inside those jobs (an upper bound): H-s101 29.5, R-s101 26.0,
U-s101 22.6, H-s202 29.5, R-s202 22.3, U-s202 19.5. Jobs that wrote no canonical row (U-s202 11097811, H-s202 11112338)
are excluded. Mixed GPU types: tokens remain the hardware-free cost." Reason: report v2's "of which canonical" counted
whole chain jobs including 11097811 (N6). Evidence: `pass-b/train.md` §7; `tools/reconcile/costs_reconcile.json`
(study totals before A26: 315.6 GPU-h, 2,555.5 CPU-h, 260 jobs; M6, M7).

## §3.D GRPO units
**C11. "momentum-only step".** Add: "Verified by both passes on the same (round, minibatch) sets. In R and U every
momentum-only step holds 2 all-correct practice groups (practice groups were never all-wrong: 0 of 4,096 per run). Per
32-round quarter: H-s101 0/0/6/12, R-s101 0/9/24/33, U-s101 1/7/16/29, H-s202 0/0/7/13, R-s202 0/2/3/20, U-s202
1/8/23/33. The size of the weight change is not logged." In "optimizer update", add: "every optimizer step changed the
weights; an all-correct or all-wrong group adds no direct gradient but counts in the 1/32 divisor." Reason: report v2
said the practice slots updated nothing (I5). Evidence: `pass-b/train.md` §4; `pass-a/schedules.md` §4 (S14).

## §4 Differences found
**C1. Item 12.** Replace with: "H-s202 endpoint shards k0–k2 ran on L40S (65 problems), k3–k5 on RTX A6000 (63
problems); the recipe allowed any 48 GB GPU. H-s202 r128 stress shards: k0, k2 L40, k1 RTX A6000. Every other stress
cell ran on L40S." Reason: k2 ran on L40S, not "any 48 GB GPU" (W2). Evidence: `sacct` 11120634, 11120590/1/5,
11120596/8, 11121110; both passes (M9).

**C2. Item 6.** Add: "The offset also appears in R-s202 metrics step rows (`opt_step` 77–524 for rounds 17–128, +12)
and round-row counters `gen_tokens`, `prompt_tokens`, `hard_responses` after round 16. Report v2's R-s202 prompt-token
figure (1.48 M) comes from the counter; recompute it from records. Index R-s202 steps by (round, minibatch)."
Reason and evidence: `pass-b/train.md` §4, §9 item 2; `pass-a/code.md` D5 (W3, N7).

**C7. Item 1 and the diagnostic J_H.** Replace "no-update J_H was 0.606" with 0.605 (0.60546875). Add: "Reference GPUs:
r0 branches and `none` on L40 (job 11102499), r64 on RTX 6000 Ada (11101908), `none_rerun` and `none_cycled` on L40S
(11105174); `none_cycled` equals `none_rerun`. The rerun exceeds `none` by +3.1 (r0) and +2.5 (r64) J_H points, more
than the range of branch-type means in 5 of 6 cells, so mean ΔJ_H is positive against `none` and negative against
`none_cycled`. At r64 three branches (related 4101, 4102; unrelated 4103) were momentum-only and produced identical
evaluations." Reason: rounding error (N5); the reference dependence and hardware were not stated (I7, M17).
Evidence: `research/evidence/a15/diag_r64.jsonl`; `pass-b/train.md` §5; `pass-a/code.md` D10.

**New item 18 (interpretation).** "Training GPU type differs between runs and inside two runs (§2 'training hardware')."
Evidence: C5 (I8).

**New item 19 (numbers).** "Grader limitations beyond the split-box case (C12)." Evidence: N8.

**New item 20 (wording).** "Two close variants of H_train problems in dev and the sealed test (C6)." Evidence: W6.

## §5 Open items
**C17.** Mark items closed with their evidence: 1 (full schedule: `pass-a/schedules.md`, `pass-b/train.md` §1);
2 (minibatch grouping: C9); 3 (H-s202 GPUs: C1); 4 (unknown counts: C13); 5 (excluded segments: C3, `pass-a/code.md`
§3 "Superseded exclusion"); 6 (length extension: C15); 7 (a2-orig config and recipe: C4); 8 (text overlap: C6; no
near-duplicate within H_train, max 5-shingle 0.237). Item 9 (checkpoint integrity) stays open until the checkpoint
agent reports.
