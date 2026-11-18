# direction — living scientific direction and hourly review log (study s02)

Purpose: say what we are resolving now, how we will know, and what would change our mind. The review log at the
bottom gets one timestamped entry per hourly review (EST) and after each important event.

## Current direction (updated 2026-11-16 02:45 EST)

**Resolving now.** RQ1 screen with the first joint run, J(caesar, terse). Precondition met: caesar is acquired alone
from 4-5% (s8/s12) to 71-88% in 32 rounds (F3), so a caesar deficit in J cannot be blamed on intrinsic difficulty.
caesar training alone lowers the terse task by .34 (F3): a behavioural spill-over in one direction. J(caesar, terse)
asks whether the reverse also happens (terse training shortening or suppressing caesar) and whether either task's
acquisition is delayed relative to its own R.

**Planned RQ2 family (if J(caesar, terse) shows a caesar deficit).** caesar = T, terse = C. Vary initial T success
with the caesar strata (s4 / s8 / s12 in the same run) and, separately, by branching from R(caesar) checkpoints;
vary competing update strength with the C weight (0.5 / 1 / 2). Amplification predicts a larger, longer-lasting
deficit in s12 than in s4 at comparable local harm, and a rising A0 fraction of caesar groups in J before the
deficit appears. Order: terse first then caesar, vs caesar first then terse.

**Earlier framing (2026-11-15 23:28).** Preconditions for both questions: a target T with low but measurable success
that learns alone, and competitor tasks with distinct relations to T. Then the first RQ1 screen: does C harm, help or
not affect T's acquisition relative to the matched reference R(T)?

**Leading explanations for any deficit we see.**
1. Direct adverse updates from C (gradient conflict, measurable as a local update effect).
2. Optimizer-mediated coupling: C's gradients enlarge Adam's second moment and the clip norm, so T's steps shrink.
3. Feedback amplification (RQ2): T success falls, T groups go all-wrong, T's own signal vanishes.
4. Non-interference artefacts: intrinsic difficulty, format or length collapse (truncation, missing box), grading.

**What distinguishes them.**
- 1 vs 2: the local update effect of a C-only round (direction) vs J-vs-R deficit with matched T gradients; a
  later small control (reset or shared Adam state) only if the deficit is real.
- 3: deficit grows faster, or persists longer, in low-p strata than in high-p strata at comparable local harm; A0
  fraction of T groups rises in J relative to R before the deficit appears; recovery after C stops is slower when A0 is
  high. Discriminating branch: from a harmed checkpoint, continue T-only vs T-only with more samples per prompt.
- 4: per-role length, truncation, no-answer and over-length counts; grader audit of wrong answers.

**What would change our mind.** No pair shows a deficit beyond seed noise at short horizons → move to stronger,
controlled conflicts (brevity weight λ) to test RQ2's mechanism in isolation, and say plainly that natural pairs
here were benign. A deficit that equals the local effect and vanishes when C stops → transient harm, no amplification.

**Ruled out so far.** Nothing.

**Drift check.** Engineering so far: environment rebuild, grading and runner extensions needed for the first family.
No descriptive diagnostics without a question attached.

**Next start / stop / revise.** Canary (joint add9+countdown, forced exit + resume) after calibration 1. Then E1 as a
single-task matrix: train each task alone (= its reference R) and evaluate every task's panel every few rounds. One run
per task gives its own learning curve and its off-target effect on every other task (RQ1 matrix). Joint runs follow
only for the pairs that the matrix and calibration make informative.

**Amendment A1 (23:30).** Task set broadened: add9 (arithmetic rule), countdown (search puzzle), caesar (symbolic),
MBPP (code, execution-graded), ARC-Challenge (factual), terse answers (a behavioural objective, kept apart from skills).

**Capacity constraint (measured 23:34).** `preempt` estimates a start in ~2 days (224 running, >1,000 pending);
`general` has given starts within 1 min. Plan for ~2 GPUs; submit spare work to preempt anyway.

## Monitoring mechanism
- Events: `bin/ae-watch` as a background task of this Claude session (exits on any terminal job event).
- Hourly review: Claude Code session cron, every hour at :17 local time. **Limitation:** it lives only in this
  session and fires only while the session is idle; it expires after 7 days. If the session ends, a fresh
  orchestrator must re-create it (CronCreate, "17 * * * *") after the bootstrap.

## Review log
- **2026-11-15 23:12–23:28 EST — study start.** Read agent.md, handovers, main.pdf (via summary),
  trainer code. Babel clone and `~/ae-qis` were absent; recreated (`ae setup` pass). Environment job 11167443 (CPU)
  held; calibration job (general) builds the env itself. First calibration attempt 11167475 failed in 2 s: the
  study-00 lock is internally inconsistent (hydra-core 0.11.3 vs omegaconf 2.4.0); fixed with `--no-deps`;
  resubmitted as 11167515. Independent code review found 5 defects; all fixed (commit 6b31745). Next: calibration.
- **23:28–23:40 EST.** Amendment A1 recorded. New families and graders added (commit 627ad13), calibration 2 submitted
  (11167621, general). Canary 11167640 submitted to preempt after calibration 1. Preempt queue measured as nearly
  unusable; design compressed to a single-task matrix first.
- **2026-11-16 00:36 EST — hourly review.** Evidence: calibrations 1 and 2 harvested (FINDINGS F1); grader audit found
  three artefacts (code blocks, countdown value box, unboxed ARC letter), fixed in 602b151, regrade job 11168204 pending
  on the CPU queue. Canary 11168205 (general, commit 0288b41) initialising vLLM (first compile after the wipe). GPUs:
  general 1/2, preempt 0/6; CPU jobs 1/20. One general GPU idle for minutes, held deliberately by the canary rule.
  Decisions: lr 2e-6 (study-00 selection D57); E1 = single-task matrix of 7 tasks (specs/e1.json), released after the
  canary's first training round. RQ status: no learning data yet; RQ2 target candidates exist (caesar s8/s12 at 8%/4.5%,
  add9 d8 at 4%). Next discriminating action: canary metrics → release E1 (2 on general, rest on preempt).
- **2026-11-16 01:15 EST — deadline feasibility (two hours of work).** Measured: rounds ~2 min at ~1,000-token answers
  (less for caesar/MBPP/ARC); E1 run of 32 rounds + 7 all-task evals ≈ 1.5 h. Usable GPUs: 2 general (preempt: 6 E1
  jobs pending on Priority; estimates of days). Projection on 2 GPUs: E1 matrix (7 runs) done ≈ 06:00; joint pair runs
  (4-6) ≈ 11:00; RQ2 discriminating branches and one confirmation seed ≈ 17:00-18:00; synthesis by 21:00 feasible but
  with little slack. Prioritisation: (1) pairs with the target candidates caesar and add9 first; (2) RQ2 branches
  before broad confirmation; (3) drop countdown from joint runs if its truncation stays high (long, costly answers).
  Revisit at each hourly review. Canary passed (IMPLEMENTATION-LOG); E1 released at 00:41.
- **2026-11-16 01:40 EST — hourly review.** Ops: the ssh control master to Babel went stale at ~01:36 (commands hung);
  reset with `ssh -O exit babel`; watcher alive. Jobs: E1 caesar and add9 running on general (2/2); E1 mbpp started on
  preempt (the pool does start jobs); 4 pending on preempt (terse, add10, J(caesar,terse), J(add9,add10)); total
  preempt 5/6; CPU 1/20 (regrade still pending on Priority). Evidence (live, not yet harvested; exploratory):
  caesar alone learns fast (s4 .27 -> .56, s8 .08 -> .24, s12 .045 -> .11 by round 8; ~37 s/round), and caesar training
  lowers the terse behavioural task (sel pass@1 ~.48 base -> .23 r4 -> .105 r8) while terse answers lengthen
  (18 -> 65 tokens): a cross-task behavioural spill-over (written steps vs terse), not yet validated. RQ2 targets are
  learnable alone (needed precondition): caesar s8/s12 rise from low base. Decisions: none new. Next discriminating
  action: harvest caesar/add9; confirm the terse drop against the run's own round-0 eval; start J(caesar,terse) next.
- **2026-11-16 02:40 EST — hourly review (+ caesar harvest).** Evidence: E1 caesar harvested (F3). Validation of the
  terse drop: failures are `over_len` (correct product but > 24 tokens): over_len share .066 -> .891, mean length
  19 -> 107 tokens, truncation 0. So caesar training (step tables) overrides the "no working" instruction; the
  multiplication skill itself is intact. Classified: competing behavioural demand (A1 distinction), harmful,
  persistent over 32 rounds when terse is not trained. Ops: tool-permission checker failed transiently 02:40-02:45 (no
  shell); recovered. Jobs: general 2/2 (add9 R, J(caesar,terse) started 02:34); preempt 3 running (mbpp, terse, add10
  R) + 1 pending (J(add9,add10)) = 4/6; CPU 1/20. Next discriminating action: J(caesar,terse) vs R(caesar) and
  R(terse) — does joint training delay either acquisition, and does caesar's low-p stratum suffer more?
- **2026-11-16 03:10 EST — event review (add9 harvest).** The add9 single-task run shows a within-family RQ2 pattern
  (F5): d6 rises .18 -> .49 by r16, then collapses to .05 with all-wrong training groups r26-31, while d4 improves to
  .77; d8 never gets signal. This is now the most direct RQ2 lead. Competing explanations: (i) within-family
  competition (d4 updates displace what d6 needs; then scarce d6 feedback prevents recovery), (ii) training
  instability at lr 2e-6 unrelated to the mixture, (iii) seed noise. Discriminating runs submitted (E3a, preempt):
  seed-2 replicate; exposure-matched d6-only (3 groups), d8-only (2), d4-only (3). Prediction under (i): d6-only learns
  steadily; d4-only lowers d6; under (ii): d6-only also collapses. Ceilings: general 2/2, preempt 6/6. E2 probe of
  J(caesar,terse) waits for a slot. Storage pruned (IMPLEMENTATION-LOG).
- **2026-11-16 03:30 EST — event (live look, unvalidated).** J(caesar,terse) vs R(caesar), gains from each run's own r0
  (r0 differs by up to .05 between runs): r16 s4 +.46 vs +.42; s8 +.24 vs +.47; s12 +.21 vs +.31. The easy stratum
  catches up, the harder ones lag: the RQ2 prediction's direction, to be tested with paired intervals at r32. Terse in J
  rose .40 -> .95 by r4, now falling (.77 at r12) as its answers lengthen (16 -> 25 tokens): early cooperation then
  competition. E2 probe from J r8 (checkpoint ready 03:26) waits for a slot; E3a has priority for the next general GPU
  (order: add9 d6-only, add9 mix seed 2, d4-only, d8-only). add10 R was preempted and requeued (normal).
- **2026-11-16 03:36 EST — hourly review (short).** No job completed since 03:30. Ceilings: general 2/2 (J(caesar,terse),
  J(add9,add10)); preempt 1 running (terse R) + 5 pending (add10 R requeued after preemption; 4 x E3a) = 6/6; CPU 1/20.
  Cancelled regrade2 (11168204, pending 3 h on the CPU queue): superseded, because every E1 run's round-0 eval already
  measured mbpp/arc/countdown with the fixed grader (F3/F4). Forecast unchanged: E3a starts on general as the J runs end
  (~04:00-04:30); RQ2 add9 evidence ~06:00. Next discriminating action: move add9 d6-only to the first free general GPU.
- **2026-11-16 04:45 EST — hourly review.** Evidence: J(add9,add10) reached r32 (live read, unvalidated): add9 d4 .75,
  d6 .53, d8 .00, add10 d14 .61, terse .08. Against R(add9) r32 (d6 .05), the joint run did NOT collapse on d6.
  If this holds after harvest, the F5 collapse is not a stable property of add9 training at this lr; seed noise or
  instability are now the leading explanations, and RQ2 "amplification" inside add9 is weakened, not refuted. The
  seed-2 replicate (11169764) started on preempt 04:26 and is the decisive test; d6-only remains next. Terse falls
  to .08 in J(add9,add10) too (format spill-over, as F5). Ceilings: general 2/2 (both J runs, ending); preempt 1
  running + 5 pending = 6/6; CPU 0/20. Ops: `srun --overlap` into 11168639 hangs; two read steps killed at timeout,
  job unaffected. Decision: no cancellations; all queued work still discriminates. Next discriminating action: on J
  completion, paired D(t) for both J runs and move add9 d6-only to general.
- **2026-11-16 05:05 EST — event (J(caesar,terse) done, harvested, analysed).** F6: harder caesar strata lag R by
  +.10 to +.18 at r8-r16, then catch up by r24 (transient delay, one seed). During the lag J had fewer mixed and more
  all-wrong caesar groups (RQ2 chain visible, not persistent). Terse gave a real signal only in r1-2, so the delay
  arises with a mostly silent competitor; competing explanations: early adverse terse updates, run-to-run rollout
  noise (r0 already differs by .06 on s8). Decisions: (1) moved add9 d6-only (11169765) to general (pending
  Priority); (2) prepared seed-2 R(caesar) + J(caesar,terse) (specs/e1c.json) for the next general slots — it decides
  noise vs effect; (3) E2 probe at r8 deprioritised: at r8 terse is all-correct, so Conly ≈ null by construction.
  Bundle s02-v1 SHA recorded (data/<ID>/BUNDLES.md). Next discriminating action: seed-2 caesar pair on general.
- **2026-11-16 05:25 EST — event (J(add9,add10) done, harvested; seed-2 caesar pair submitted).** F7: add10 does
  not harm add9; J keeps d6 at .53 where R collapsed to .05 (D -.48 at r32). So F5's collapse is not caused by a
  competitor. Leading explanations for F5 now: seed noise, or instability from R's larger effective Adam step on add9
  (J shares the step with add10: D3 confound). This also changes how every J-vs-R gap must be read: J's T step is
  smaller under Adam, so "J behind R early" (F6) may partly be a slower effective learning rate, not interference.
  Submitted E1c: R(caesar) s2 11171036 (general), J(caesar,terse) s2 11171038 (preempt) at 9a6f560. Ceilings:
  general 2/2 (d6-only, caesar R s2 running); preempt 1 running (mix s2) + 5 pending (d4, d8, terse R, add10 R,
  J caesar s2) = 6/6. Step-size control for D3: when C's groups are all-equal, C adds no gradient and J's add9
  step equals R's; when C is active, Adam splits one step between roles. The matched control is therefore R(add9)
  at lower lr (1e-6), which halves the add9 step without a competitor. Next discriminating action: read mix s2 and
  d6-only at r16-r24; if mix s2 also collapses, run R(add9) lr 1e-6 seed 1 (instability vs competition).
- **2026-11-16 05:36 EST — hourly review (short).** No job ended since the 05:25 event review (F7). Running: add9 mix
  s2 (preempt), add9 d6-only and R(caesar) s2 (general). Pending on preempt: d4, d8, terse R, add10 R, J(caesar,terse)
  s2. Ceilings: general 2/2, preempt 6/6, CPU 0/20. Reviewed pending work: terse R and add10 R stay (they are the C
  references for F6/F7 and measure each C's off-target effect on T); E3a d4/d8 stay (needed to read the d6 result).
  Nothing cancelled. Watcher running (since 05:10). Next discriminating action unchanged: read mix s2 and d6-only at
  r16-r24 (~06:30-07:00); then decide the lr 1e-6 control.
- **2026-11-16 06:36 EST — hourly review (incident + F8).** Evidence: (1) the compute disk filled at 06:19 (501G/501G,
  study s02 checkpoints 256G of it — my runs caused it); three runs died (mix s2, d6-only, caesar R s2) and d4-only
  failed at precheck. (2) mix s2 and d6-only had completed all evals to r32 before dying, so F8 is valid: the d6
  collapse replicates (seed 2: d6 .30 → .00 at r16) and is absent with d6 alone (.18 → .47) and with add10 added
  (F7). d6-only takes the full Adam step and does not collapse, so the step-size alternative is weakened. RQ2 add9
  lead is restored, now as a mixture effect: something in the d4/d8 updates breaks d6, and d6 then gets no signal.
  Decisions: held all pending study jobs; pruned s02 checkpoints to 20G (kept mix s2 r16 = collapse state); recipes
  now refuse to start with < 60G free and prune non-retained checkpoints (4f40a34, ea73639). df still shows only 13G
  free after the prune; re-measuring (11171795) before releasing anything. Ceilings: general 1/2 (diag), preempt 4/6
  (all held), CPU 0/20. Next discriminating action: once space is confirmed, resubmit d4-only (does d4 training alone
  lower d6?) and caesar R s2 on general; probe from mix s2 r16 (one d6-only round vs one mixed round vs null).
- **2026-11-16 06:43 EST — event (space confirmed; restart).** diag-df 11171795: 265G used, 236G free, stable over 5 min
  (the 13G reading was NFS reclaim lag). Restarted at ea73639 (storage guard): caesar R s2 11171812 (general, running),
  d4-only 11171814 (general), J(caesar,terse) s2 11171816 and d8-only 11171817 (preempt). Cancelled terse R and add10
  R (held, old recipes; they are only C references — not needed for RQ1/RQ2 claims now). New family E3b (324525b):
  recovery from the collapse state (mix s2 r16, d6 .00): 8 rounds of (a) the original mixture, (b) d6 only, (c) d6
  only with 32 samples/prompt; no checkpoints; preempt 11171822-4. A one-round probe was rejected: at r16 all d6
  groups are all-wrong, so a one-round d6 update is zero by construction. Storage budget: at most 4 checkpointing
  runs at once (~40G peak each during a save). Ceilings: general 2/2, preempt 5/6, CPU 0/20. Next discriminating
  action: E3b readout (does d6 recover only when it gets successes?) and d4-only (does d4 alone lower d6?).
- **2026-11-16 06:59 EST — event (bad node).** All three E3b jobs failed in ~20 s on babel-l9-32: "CUDA-capable
  device(s) is/are busy or unavailable" (node fault). One recorded retry: excluded the node in specs (baeda29), in the
  card, and on pending d8-only / J(caesar,terse) s2 (ExcNodeList); resubmitted E3b as 11171949 (mixed), 11171953 (d6),
  11171994 (d6n32). Ceilings: general 2/2, preempt 5/6.
- **2026-11-16 07:36 EST — hourly review (short).** No job ended since 06:59. Running on general: d4-only (54 min),
  caesar R s2 (56 min), both expected to end ~08:00-08:20. Preempt: 5 pending, none started in 50 min (congested).
  Ceilings: general 2/2, preempt 5/6, CPU 0/20. Nothing cancelled: all pending work is on the RQ2 critical path
  (E3b, d8-only) or the F6 replication (J s2). Plan: when a general GPU frees, move E3b there in the order d6n32,
  d6, mixed (each ~30-40 min, no checkpoints). Feasibility of 21:00 EST: on track if E3b and d4-only read out by
  ~10:00; the report skeleton starts now from harvested runs (F3-F8). Next discriminating action unchanged.
- **2026-11-16 07:45 EST — event (d4-only done).** F9: d4 alone lifts d6 (+.21), so it does not cause the collapse.
  Collapse count: 2/2 mixed-strata runs, 0/3 others. Next discriminating run is d8-only (does training on an
  almost always all-wrong stratum harm d6? its rare mixed groups give large, noisy updates). Moved E3b d6n32
  (11171994) to general (freed GPU). Order for the next general slots: d8-only 11171817, then E3b d6, E3b mixed.
  Checkpoint auto-prune verified in a real run (d4-only round_0032 deleted at export).
- **2026-11-16 07:58 EST — event (caesar R s2 done).** F3 replicates (seed 2: s12 .04 → .64; terse .46 → .06).
  J(caesar,terse) s2 started on preempt (07:54). Moved d8-only (11171817) to general. general 2/2 (E3b d6n32,
  d8-only), preempt 1 running + 2 E3b pending + 1 CPU analysis job. Auto-prune worked again (caesar s2 r32).
- **2026-11-16 08:25 EST — event (E3b d6n32 done; partial readout).** From the collapse state (mix s2 r16), d6-only
  with 32 samples/prompt: training groups all-wrong in 7 of 8 rounds; round 20 had ONE mixed group (2/96 correct).
  Eval d6: sel .00 → .00 (r20) → .148 (r24); conf .004 → .008 → .113. So one rare success was followed by a large
  recovery during rounds 21-24, whose own gradients were zero. Candidate mechanism: Adam momentum (m decays by 0.9
  per step, so a single nonzero step keeps moving the weights for ~10 more steps; sum ≈ 10x one step). Consequence
  for every analysis: an all-wrong round is NOT a null update under Adam — it continues the previous direction.
  This also bears on the collapse itself: in the mixed run, d6's all-wrong rounds still receive the d4/d8-driven
  momentum. Wait for E3b d6 (8 samples) and E3b mixed (both running on preempt) before concluding.
- **2026-11-16 08:34 EST — event (E3b mixed done).** Continuing the original mixture from the collapse state also
  recovers d6 partly: sel .00 → .00 (r20) → .07 (r24); conf .004 → .016 → .072. (The original seed-2 run read .00 at
  r24; the branch differs only by non-reproducible vLLM sampling, so the collapse state is fragile, not absorbing.)
  d6n32 − mixed at r24 (paired, 64 problems): conf +.041 [−.002, +.092]; sel +.078 [+.023, +.133]. More d6 samples
  help a little; neither branch is stuck. Mixed-branch training groups: 2-3 mixed of 8 every round (d4 supplies them).
  Reading so far: the collapse is a deep but escapable dip, not a permanent loss; the RQ2 "no signal → no recovery"
  chain is too strong as stated — recovery happens even with almost no d6 signal (d6n32: one mixed group in 8
  rounds), via momentum and/or d4 transfer (F9). Waiting: E3b d6 (8 samples), d8-only.
- **2026-11-16 08:38 EST — hourly review.** Evidence since 07:36: F9 (d4 alone lifts d6), F3 replicated (caesar s2),
  E3b d6n32 and mixed (collapse state is escapable; recovery with almost no d6 signal). RQ2 position: the add9 d6
  collapse is real in mixed runs (2/2) but transient and escapable; the "fewer mixed groups → no recovery" chain
  holds only while momentum and transfer from other strata do not rescue it. Competing explanations still open:
  d8 updates (d8-only running), chance (seed 3 submitted). Decisions: submitted add9 mixed seed 3 (11172967,
  general, 417184d) to use the idle general GPU; nothing cancelled. Ceilings: general 2/2 (d8-only, mix s3),
  preempt 2 running (E3b d6, J caesar s2) + 1 CPU job pending; CPU 1/20; checkpointing runs 3/4. Next
  discriminating action: d8-only readout (~09:20) and E3b d6 (8 samples).
- **2026-11-16 08:47 EST — event (E3b complete).** F10: from the collapse state, d6 with 8 samples gets 0 successes in
  8 rounds and stays at .000; with 32 samples one success (r20) starts a recovery to .11-.15; the original mixture
  recovers partly (.07) through d4 transfer. Momentum alone does not rescue d6. This is the clearest RQ2 evidence:
  scarce successes make the deficit persist; a success from sampling or transfer ends it. Next: d8-only (cause of
  the collapse), mix s3 (frequency). Candidate follow-up if time allows: replicate the three branches from a second
  collapse state (mix s3 if it collapses) — branch-level noise is the main weakness of F10.
- **2026-11-16 08:49 EST — decision.** Cancelled add9 mixed seed 3 (11172967, ~10 min in) and resubmitted it as
  qis-s02-e3-add9mix-s3r (11173101, 8abdf93) with retain_rounds [16, 24], so E3b can be repeated from a second
  collapse state if seed 3 collapses. New expid because the cancelled run's directory holds a stale config that the
  recipe's config phase would reuse. Storage: +40G retained; 236G free at 06:39; 3 checkpointing runs.
- **2026-11-16 08:56 EST — event (J(caesar,terse) s2 done).** F11: the delay of harder caesar strata replicates (r16: s8
  +.12, s12 +.15; seed 2 r0 aligned, so not cross-run noise); in seed 2 it persists at r32 on s8 (+.12 [+.05,+.20]).
  Caesar answers are 5-15% shorter in J during the lag in both seeds. RQ1 classification for caesar under terse:
  delay of harder strata (2/2 seeds), transient in one seed, unresolved in the other. RQ2: the harder strata lose
  mixed groups during the lag (both seeds), consistent with the chain. Free GPUs: none on general (d8-only, mix s3r);
  preempt 0 study GPU jobs. Next: consider a third caesar seed or a length-controlled test only if time remains;
  priority stays with d8-only and mix s3r.
- **2026-11-16 08:57 EST — decision (RQ1 breadth).** Preempt had no study GPU job, so submitted E1d at 29cb12d:
  J(caesar,mbpp) s1 (11173146; cross-domain skills, A1: "do not assume different domains conflict") and
  J(add9,terse) s1 (11173148; behavioural objective vs arithmetic skill). References are the seed-1 R runs (14be5a9;
  run.py changes since then are opt-in only, default path identical). Checkpointing runs now 4 (d8-only, mix s3r,
  2 x E1d): at the storage limit.
- **2026-11-16 09:01 EST — event (d8-only done) and decision.** F12: d8 alone lowers d4 (−.25) and d6 (−.15) while d8
  itself is not learned; every d8 update came from rare-success groups. With F9 (d4 helps d6) and F10 (no success →
  no recovery), the add9 collapse has a candidate cause and mechanism. Discriminating test submitted: E3d d4+d6
  mixture without d8, seeds 1 and 2 (11173237, 11173240; 4eb0bd4) — predicts no collapse. Held J(add9,terse)
  (11173148) to stay within 4 checkpointing runs; E3d gets the next general GPU. Moved J(caesar,mbpp) to general
  (running). Remaining open: why add10 prevents the collapse (F7).
- **2026-11-16 09:36 EST — hourly review.** Evidence since 08:38: F10 (no success → no recovery), F11 (caesar delay
  replicates), F12 (d8 updates harm d4/d6). RQ2 add9 story now has a candidate source (d8), the chain, and the
  persistence step; RQ1 has caesar×terse (delay of harder strata, 2/2 seeds), add9×add10 (no harm), and off-target
  effects of single tasks. Competing explanations still open: d8 cause (E3d d4+d6 running, 2 seeds, preempt, started
  09:09-09:11), collapse frequency (mix s3r, general, 47 min). Decisions: started the mandated second critical review
  (independent agent, raw data only) of F8-F12 before they enter the report; nothing cancelled. Ceilings: general
  2/2, preempt 2 running + 1 held + 1 CPU pending; CPU 1/20; checkpointing runs 4/4. Next discriminating action: E3d
  readout (~10:15) — a d6 collapse without d8 would refute F12's mechanism.
- **2026-11-16 09:40 EST — event (strata analysis) — direction change.** F13: in seed 2, d8 groups gave no gradient
  after round 4, yet d6 collapsed at r13-14. So d8 cannot be the direct cause there; F12's mechanism reading is
  corrected (FINDINGS). The d6 collapse in seed 2 arose with only d4 + d6 updates active; the errors at collapse are
  in converting the operands to decimal. Live hypotheses: (a) d4+d6 dynamics alone (E3d tests directly); (b)
  stochastic instability of d6 at this lr (frequency: seed 3); (c) d8 matters in seed 1 only. Sent the reviewer this
  new evidence. E3d is now the decisive run for the report's RQ2 add9 claim.
- **2026-11-16 09:40 EST — review returned.** Second critical review: C1, C2 supported; C3, C4, C5 partial. F10 and F12
  corrected in FINDINGS (zero-gradient d6 branch = momentum-only; n32 confounded with a larger step; d6 answers
  shorten under d8 training). New leading alternative for the harm: sparse-update amplification under Adam (generic,
  not d8-specific), consistent with F13 (seed-2 collapse without d8 gradient). Report wording for RQ2 add9 will be:
  "collapse occurs in mixed runs (2/2) and recovery requires a nonzero update (success or transfer); its cause is
  unresolved between d4+d6 dynamics, sparse-update drift, and d8 content." Decision on a random-reward control after
  E3d reads out (~10:15): if d4+d6 collapses, d8 content is not needed and the control is lower priority.
- **2026-11-16 09:42 EST — review addendum.** Reviewer (with strata data) agrees with F13: d8 is not the trigger (2
  mixed d8 groups in mix-s2, none after r4; the no-collapse add10 run had 16). New lead: a single large clipped
  update (grad norm 3.06, round 12) precedes the seed-2 collapse. E3d (d4+d6, seeds 1-2) is the reviewer's own
  recommended discriminating run and is already running. Random-reward control dropped (d8 no longer the suspect).
- **2026-11-16 10:28 EST — event (mix s3r done).** F14: seed 3 does not collapse (d6 .18 → .54). Mixed-strata collapse
  frequency 2/3. With F13 and the review, the RQ2 add9 claim becomes: in this configuration the d6 skill can fall
  into an all-wrong state (2/3 seeds, timing varies), and once there it gets no corrective signal and stays low
  until a rare success or transfer restarts it (F10 persistence part, review-supported). The trigger is unknown and
  stochastic. Moved J(add9,terse) (11173148) to general (released). E3d (d4+d6) still running (~80 min, preempt).
- **2026-11-16 10:35 EST — event (J(caesar,mbpp) done).** F15: caesar lags R by +.09 to +.12 at r32 (all strata) while
  MBPP is unaffected; terse rises to .68 instead of collapsing as under caesar alone. The caesar lag is confounded
  with Adam step sharing (MBPP active throughout). RQ1 table now: caesar×terse delay of harder strata (2 seeds),
  caesar×MBPP asymmetric delay (1 seed, step-sharing confound), add9×add10 no harm (1 seed). Running: d4+d6 x2
  (preempt), J(add9,terse) (general). One general GPU free: decide next run below.
- **2026-11-16 10:35 EST — decision.** MBPP gave 23-27 mixed groups per 8-round block (active all run). Submitted the
  D3 step-size control R(caesar) lr 1e-6 seed 1 (11174276, general, 2399dee): it decides whether F15's caesar lag is
  step sharing or interference, and bounds the same confound for F7. Checkpointing runs 4/4.
- **2026-11-16 10:36 EST — hourly review (short).** Evidence since 09:36: F13 + review (d8 not the collapse trigger),
  F14 (collapse 2/3 seeds), F15 (caesar×MBPP asymmetric delay, step-sharing confound). Running: d4+d6 x2 (preempt,
  ~85 min, ending ~10:50), J(add9,terse) (general, 8 min); pending: caesar lr 1e-6 control (general). Ceilings:
  general 2/2, preempt 2/6, CPU 0/20; checkpointing 4/4. Nothing cancelled: all running work answers an open
  question. Feasibility: no new run families after ~13:30; report drafting starts when E3d and E1e read out
  (~12:30-13:30). Next discriminating action: E3d readout.
- **2026-11-16 10:41 EST — event (E3d s2 done).** F16 (provisional): d4+d6 seed 2 does not collapse (d6 .65 at r32;
  d8 untrained rises to .18), while the matched mixed s2 run collapsed. n=1 per arm against a 2/3 base rate, so not
  decisive. Waiting for E3d s1 (~10:55). Preempt now has 5 free GPUs; considered more E3d seeds (s3, s4) to reach a
  usable comparison (mixed 2/3 vs d4+d6 k/4) — decide at s1 readout given storage (checkpointing 4/4) and time.
- **2026-11-16 10:41 EST — decision.** Submitted E3d seed 3 (11174313, preempt, 305dc56) to estimate the d4+d6 collapse rate; checkpointing runs 4/4.
- **2026-11-16 11:09 EST — event (E3d s1 done).** d4+d6 seed 1 does not collapse (d6 .50 at r32) while its mixed twin
  did. Collapse: with d8 2/3, without d8 0/2 (s3 running). Suggestive that d8's presence raises collapse risk;
  not significant; mechanism unexplained given seed 2's two d8 updates. Single-large-update lead weakened (d4+d6 s1
  had norm 2.12 without collapse). Report wording stays "trigger unresolved"; persistence + recovery are the
  supported RQ2 claims. No further add9 runs after s3: budget goes to the report.
- **2026-11-16 11:36 EST — hourly review (short).** Evidence since 10:36: F16 (d4+d6 without d8: 0/2 collapse vs 2/3
  mixed, p≈0.4). Running: J(add9,terse) (general, 68 min), R(caesar) lr 1e-6 (general, 60 min), d4+d6 s3 (preempt;
  preempted 11:28, requeued, running again 11:31 from its last checkpoint). Ceilings: general 2/2, preempt 1/6, CPU
  0/20. Nothing cancelled. Decision: start the report draft now (reports/<ID>/), filling in E1e/E1d-add9-terse/E3d-s3
  as they land (~12:00-12:45). Next discriminating action: E1e readout (step-size explanation of F15).
- **2026-11-16 11:38 EST — report.** Draft written (report.md, runs.md, figures add9_d6/add9_strata/recovery/caesar_D/offtarget; cb32b53). Pending inputs: E1e, J(add9,terse), E3d s3.
- **2026-11-16 12:37 EST — hourly review.** No job ended since 11:09. R(caesar) lr 1e-6 finished training at 11:45 but
  is stuck behind slow checkpoint writes (round 24 took ~1 h at ~6 MB/s; NFS congestion; disk 165G free).
  J(add9,terse) has written nothing since round 24 (11:50) — same cause. d4+d6 s3 (preempt) running since 11:31.
  Decision: wait rather than cancel (cancelling skips the copy of metrics into results; no other run needs the GPU).
  Ceilings: general 2/2, preempt 1/6, CPU 0/20. Next: harvest E1e when its final save ends; report continues.
- **2026-11-16 12:45 EST — event (E1e done).** F17: halving lr costs caesar .14-.53 at r32, far more than MBPP's lag
  (.09-.12). F15's lag is explainable by Adam step sharing; reclassify caesar×MBPP as "no interference beyond step
  sharing" (RQ1). The caesar×terse delay (F6/F11) remains the one cross-task delay not explained by step sharing.
  general 1/2 free; no new families (13:30 cutoff approaching; report priority).
- **2026-11-16 12:46 EST — decision.** Submitted E1f caesar-terse seed 3 pair with no checkpoints (d2c5199): R s3 11175477 (general), J s3 11175478 (preempt). It firms up the one cross-task delay not explained by step sharing (F6/F11/F17). Last new submissions of the study (13:30 cutoff).
- **2026-11-16 13:03 EST — event (E3d s3 done).** d4+d6 seed 3: no collapse. Without d8 0/3 vs with d8 2/3 (p=0.4). Report: 'collapse only seen with d8 groups present; not significant; trigger unresolved'.
- **2026-11-16 13:06 EST — event (J(add9,terse) done).** F18: no harm to add9 (J ahead of the collapsed R); terse learned to .93. Moved J(caesar,terse) s3 to general (11175478). add9-mixture collapse: alone 2/3, with competitor 0/2, without d8 0/3.
- **2026-11-16 13:06 EST — report.** handover.md drafted. Final steps after caesar s3 pair: runs.md state column, records snapshot to research/<ID>/records-2026-11-16/, audit (recompute sample numbers), final report pass.
- **2026-11-16 13:36 EST — hourly review (short).** Evidence since 12:36: F17 (step sharing explains caesar×MBPP lag),
  F16 final (0/3 collapse without d8), F18 (add9×terse no harm). Running: caesar s3 pair on general (R 50 min, J 30
  min), no checkpoints. Ceilings: general 2/2, preempt 0/6, CPU 0/20. No new submissions (cutoff passed). Next:
  E1f readout (~14:30-15:00), then audit and final report.
- **2026-11-16 13:37 EST — audit.** Independent recomputation of 18 report numbers (F3, F6, F11, F15, F17, F18) from raw metrics: 0 mismatches; denominators 64x4 verified (research/<ID>/audit-2026-11-16.md, f4df3eb). F8-F12 covered by the review agent.
- **2026-11-16 14:05 EST — event (R(caesar) s3 done).** Acquisition replicates (s4 .92, s8 .85, s12 .66 at r32); terse drop weaker in this seed (.40 → .29 vs .06-.07 in seeds 1-2). Waiting for J(caesar,terse) s3 (general, ~55 min in).
- **2026-11-16 14:12 EST — event (E1f done; all study jobs finished).** F19: seed 3 shows no caesar delay; delay 2/3
  seeds; in each seed the delay co-occurs with shorter joint-run caesar answers (absent in s3, where answers were
  longer). RQ1 classification for caesar under terse: delay of harder strata in 2/3 seeds, linked to brevity
  spill-over (correlational). No jobs remain. Next: final report pass, runs.md state column, records snapshot.
- **2026-11-16 14:14 EST — final.** All 38 study jobs ended. Report finalised (report.md with F17-F19, runs.md with
  sacct states, handover.md, figures incl. seed 3). STATE marked FINAL. Records snapshot refreshed in
  research/<ID>/records-2026-11-16/. No study jobs running; watcher left to expire.
- **2026-11-16 14:37 EST — hourly review + decision.** No study jobs were running; report final (1fb1050). With ~6.5 h
  to the deadline, the weakest open point is the collapse trigger (2/3 with d8 vs 0/3 without, p=0.4). Submitted
  E3g at 338c994, no checkpoints: mixed s4, s5 (general; 11176691, 11176692) and d4+d6 s4, s5 (preempt; 11176693,
  11176696). Plan: fold results into the report by ~19:00 if they land; the report stands as is if they do not.
  Ceilings: general 2/2, preempt 2/6, CPU 0/20. Watcher re-armed.
- **2026-11-16 15:24 EST — event (E3g d4+d6 s4 done).** No collapse: d6 .17 → .48 (r32), d4 .52 → .70, d8 .04 → .12. Without d8: 0/4. Mixed s4, s5 running (general, ~47 min); d4+d6 s5 running (preempt).
- **2026-11-16 15:36 EST — hourly review (short).** Evidence since 14:36: d4+d6 s4 no collapse (0/4 without d8). Running: mixed s4, s5 (general, 59 min; add9 mixed runs take ~1.5-2 h), d4+d6 s5 (preempt, 25 min). Ceilings: general 2/2, preempt 1/6, CPU 0/20. Nothing cancelled. Next: mixed s4/s5 readout (~16:15-16:45), then update F16 and the report.
- **2026-11-16 15:59 EST — event (E3g d4+d6 s5 done).** No collapse: d6 .17 → .57, d4 .52 → .79, d8 .04 → .12. Without d8: 0/5. Waiting for mixed s4, s5 (general, 82 min).
- **2026-11-16 16:15 EST — event (E3g mixed s5 done).** No collapse: d6 .16 → .62, d8 .04 → .18. With d8: 2/4; without: 0/5 (Fisher two-sided p ≈ 0.17). Waiting for mixed s4 (general, 98 min).
- **2026-11-16 16:25 EST — event (E3g complete; all study jobs ended).** Mixed s4, s5 did not collapse: with d8 2/5, without 0/5 (p=0.44). Report, handover, figures updated; runs.md regenerated.
- **2026-11-16 16:36 EST — hourly review (short).** No study jobs; report final (c38ac74). Decision: no new experiment families before the deadline (the next tests need code changes: per-stratum gradient logging, a looser terse length limit; each needs a canary and review). Submitted a read-only storage check (qis-s02-diag-final) to close the handover's open item. Ceilings: general 1/2, preempt 0/6, CPU 0/20.
- **2026-11-16 16:38 EST — storage verified.** No checkpoints remain; s02 ~256 MB; 200G free. Handover updated.
- **2026-11-16 17:36 EST — hourly review (short).** No study jobs; nothing new. Report final (be2900d). Ceilings 0/2, 0/6, 0/20. Watcher not re-armed (nothing to watch).
- **2026-11-16 18:36 EST — hourly review (short).** No study jobs; nothing new. Report final (be2900d).
- **2026-11-16 19:00 EST — STUDY STOPPED by owner instruction** ("stop the study and run no other jobs"). No jobs queued or running; hourly cron a372f64b deleted; no watcher running. Usage: ~40.7 GPU-hours (general 30.4, preempt 10.2) across 53 sacct records (last attempt per job). Next: report rewrite only (no compute).
- **2026-11-16 19:03 EST — report rewritten** at owner request (narrative synthesis; no compute used). Study remains stopped.
- **2026-11-18 07:33 EST — report revision done.** Critique addressed; figures 2, 4-6, 9, 10 added; terse failure types and the momentum measurement (F20) in; all on preempt CPUs.
