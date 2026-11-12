# Report v3 — outline and writing rules (A26 §5)

Purpose: the binding plan for `reports/REPORT.md` v3. The writer follows it; the orchestrator reviews against it.

## Rules
- Reader: an ML researcher new to the project. No chat history, no internal jargon ("cell carried the gain", "link 1",
  "A17", "D150" in running text — decision IDs go in the appendix or in a trailing source note).
- Use `research/DEFINITIONS.md` v1.1 terms exactly: problem, attempt, group, round, checkpoint, run, optimizer step,
  pass@1, coverage at n, truncation, accuracy among finished, mixed group, momentum-only step. Never "learning signal".
  Name the sense of "hard" when it is not H_train.
- Every number comes from `reports/tables/v3/*.json` (audited) or an audit file; cite the table under each result.
  Absolute values beside every difference. Contrasts from unrounded data, printed in their own column, one rounding note
  per table. No inline `a − b = c` equations.
- Every range names its checkpoint set ("six round-128 endpoints", "five R/U and H-s101 checkpoints").
- Intervals: say what was resampled (problems; attempts fixed). Two training seeds per arm: never call a two-seed interval
  a run-to-run interval; never read an interval containing 0 as equivalence.
- Each result has one home. No opening synopsis, no "What we learned", no "Four practical answers".
- Writing standard (`project.md`, owner): short direct sentences, active voice, tables for data, no filler.
- Length target: main text ≈ 450–600 lines; appendices as needed.

## Sections
1. **Task.** What the model receives (one problem in the non-thinking chat template, instruction to box the answer),
   what it generates (up to 8,192 / 12,288 tokens), what earns reward (last box verifies against gold and response
   finished). One round: 16 prompt slots × 8 attempts. Slots 0–7: shared hard problems (same in every arm within a seed).
   Slots 8–15: the arm's allocation — more hard problems (H), direct factoring practice (R), direct modular-arithmetic
   practice (U). Why the allocation matters: it decides what the other half of every update trains on.
   Two verified examples from `data/` (quote text, gold, source ID, role): one R practice item (direct factoring
   exercise, e.g. a natural MATH construction item from the 63 pairs) and one H_train target where the solver must
   recognise that factoring helps. Say where each comes from and its role.
2. **Data and measurements.** One compact table, one row per set: purpose, source, distinct problems, overlap with other
   sets, affects training?, when and how often evaluated (n attempts, cap). Sets: H_train 46, R pool 63, U pool 63 (63
   pairs: 9 natural + 54 authored), calibration 59/56, dev 53, J_H 32 (= first 32 of dev), J_R 28, J_U 32, sealed test
   128, stress Harder 95 / SuperHard 126, control 64 (A26), cue 24, response audit 16 problems × 6 policies.
   Then: how "hard" (H_train: rubric + 2–12 of 32 base attempts correct at cap 8,192), "related", "unrelated", "Harder",
   "SuperHard" were assigned; J_R and J_U include MATH levels 1–3. Held-out status: by ID, text and the 5-shingle rule;
   two close variants of training problems remain (1 dev, 1 sealed test); removing them changes no conclusion. Link the
   manifests: `data/main_v1_manifest.json`, `data/panels/…`, stress, control, and the J lists (file + IDs).
3. **Research questions.** Primary (frozen before training): R − U on the sealed test at equal hard exposure, per
   seed. Secondary: R − H and U − H (spending all slots on hard problems). Exploratory (added later, with date and
   reason): learning dynamics and timing; coverage vs reliability; truncation and the length extension; local one-step
   interference; stress-panel transfer; cue; response audit; probe rise and control panel (A26). For each: the
   comparison that answers it.
4. **Procedure.** One GRPO round step by step (schedule → 128 attempts → grade → group advantages → 4 minibatches of
   4 groups → 4 AdamW steps, lr 2e-6, clip 0.2/0.2, no KL → weights to vLLM). Then the run: 128 rounds, cap 8,192 for
   rounds 1–32 and 12,288 for 33–128; in-run evaluations at rounds 0/32/64/96/128 (dev + probes, 8 attempts); checkpoint
   cadence and what survives. Model and template; seeds (training seed → schedule, request seeds, minibatch stream;
   evaluation seeds shared across arms; what still differs: sampled attempts, minibatch grouping H vs R/U, GPU type);
   training hardware per run (H-s101 L40; H-s202, R-s202 switched; others L40S). Endpoint = round-128 checkpoint;
   sealed test 128 × 32 at 12,288. Verifier: last box; unknown counted incorrect. Statistical unit: problem; paired
   problem bootstrap (10,000 draws) per seed. Why the common-GPU rerun exists (R and U originally ran on different GPU
   types) and why the round-116 fallback exists (H-s202 restarted late; the fallback was a pre-declared substitute;
   the real endpoint finished and replaces it).
5. **Results.** Each subsection: why needed → what compared → what happened (table + figure) → what explanations remain.
   5.1 Endpoints: one table, base + six endpoints (pass@1, pass@8, coverage at 32, truncation, accuracy among finished),
       then per-seed contrasts R − U, R − H, U − H with intervals. Common-GPU reruns and the fallback as rows below.
       Two-seed means described, not inferred.
   5.2 Learning dynamics and probes: dev curves over rounds and hard exposure; POPE-style J trajectories; earliest
       observed ordering per seed and panel (round 0 as noise reference; cap change between 32 and 64).
   5.3 Why J_U and J_R rise in every arm: taxonomy + control panel (D163).
   5.4 Coverage, truncation and local interference: coverage vs reliability; truncation decomposition (descriptive;
       selection caveat); group statistics and momentum-only steps; local one-step ΔJ (cap 8,192, reference noise,
       sign depends on reference); loss and recovery on training prompts at the sampling rate; stalls not measured;
       which links were measured on the same trajectory and which were never connected.
   5.5 Length extension: two observations (absolute gains for every checkpoint; R − U unchanged), recovered / finished
       wrong / still capped, token cost; what it cannot show.
   5.6 Stress panels: per tier, all trained checkpoints including H-s202; r64 cells labelled by trajectory; contrasts.
   5.7 Cue study and response audit (short): counts, denominators (96 / 35 / 13 capped / 192 reads / 33), excerpt and
       "WRONG" caveats, "fixable" = annotation about a hypothetical intervention.
6. **Interpretation, limitations, next decision.** Held across both seeds; changed sign; cannot settle (two runs per
   arm; hardware confounds; seed-level length drift vs arm effect). The single next comparison that reduces the most
   important uncertainty (more seeds of the current recipe on matched hardware, to estimate run-to-run spread of R − U),
   and why that is different from changing the practice pool or the cap (combining both answers another question).
7. **Appendices.** A complete tables (link `reports/tables/v3/`); B checkpoint inventory and lineage (H-s202 a2-orig vs
   a2-r60; what was pruned); C job and cost ledger (315.6 GPU-h / 2,555.5 CPU-h over 260 study jobs; per-run canonical
   GPU-h; A26 audit jobs separately); D hardware per evaluation; E hashes and commands to reproduce; F audit record
   (pass A, pass B, comparison, discrepancy log, DEFINITIONS v1.1); G unavailable cells and what could not be verified;
   H changes from report v2 (link the archive and the discrepancy log).
