# A26 x-transfer — pre-registration of the control-panel evaluation

Written 2026-11-11 (EST) by agent x-transfer, before the control panel was built and before any control-panel output existed.
Purpose: test whether the J_U and J_R gains seen in every arm are specific to number theory and factoring, or are a general
effect of GRPO training that also appears on subjects no arm practises.

## 1. Question and hypotheses
J_U and J_R rise by about 8–15 points from round 0 to round 128 in all three arms, including H and R, which never practise
modular arithmetic. The candidate explanations are E1 (answer format and termination), E2 (general carefulness), E3 (the
in-run cap rose from 8,192 to 12,288), E4 (hidden skill overlap with the shared hard problems) and E5 (noise).

- **H-general (E1/E2).** Training improves general answer behaviour. Prediction: every endpoint gains on a panel of subjects
  that no arm trains on (geometry, counting and probability, precalculus, prealgebra), by an amount comparable to its J_U gain.
- **H-specific (E4).** The J_U/J_R gains come from skills shared with the hard training problems (algebra, factoring).
  Prediction: control-panel gains are near zero while J_U/J_R gains stay positive.
- **E3 is removed by design.** Base and all endpoints are evaluated at the same cap, 12,288.
- **E5** is addressed by the paired problem bootstrap and by requiring the same direction in both seeds.

## 2. Panel rule (`control_v1`)
1. Source: MATH (`EleutherAI/hendrycks_math@21a5633`), official `test` split only, as listed in `$P/data/splits-v1/manifest.jsonl`
   (`orig_split == "test"`).
2. Subjects `geometry`, `counting_and_probability`, `precalculus`, `prealgebra`; levels `Level 4` and `Level 5`.
3. Exclude an item if (a) its ID (with or without a `math:` prefix) appears in any study file, (b) its exact or normalised text
   (`qis.data.stress_panels.norm`) equals a study item text, or (c) its masked word-3-gram Jaccard with a study item is ≥ 0.5
   (the D143/D145 scan). Study files: every JSON/JSONL under `$P/data/*` except `splits-v1` and `raw-21a5633` (the full MATH
   copy, not study sets), plus the repo `data/` tree. Unlike D143, the `stress` directories and candidate lists are included,
   so the scan is at least as strict.
4. Require a gold answer (`last_boxed` of the solution is not None) that passes `qis.verify.gold_eligible` on compute.
5. Order the remaining items of each subject by `sha256("a26-control/" + id)` (hex, ascending), where `id` is the bare manifest
   ID, e.g. `geometry/test/00012`. Take the first 16 per subject: 64 problems.
6. No model output is read at any step. The panel is frozen (sha256 recorded in `data/control/README.md`) before any evaluation.

## 3. Evaluation
Policies: base (Qwen3-1.7B `70d244cc`) and the six round-128 endpoints `$P/ckpt/qis-main-{H,R,U}-s{101,202}-sym/round_0128`.
`qis.evaluate_ckpt`, n 8, cap 12,288, seed label `control-v1`, panel key `control`, one general-pool GPU with 48 GB, the
same recipe for all seven policies. Seeds are identical across policies for each problem (paired). Output `$P/control-eval/<policy>`.

## 4. Metrics (per policy, overall and by subject)
- pass@1 = mean over problems of c/8.
- truncation share = attempts with finish `length` / attempts.
- accuracy among finished = correct / finished attempts (descriptive; it conditions on finishing).
- format classes from the record text: no box, multiple boxes with the last wrong, single box wrong (same classes as the taxonomy).

## 5. Comparisons
1. **Primary.** For each arm and seed: Δ_control = pass@1(endpoint) − pass@1(base) on control_v1, with a paired problem
   bootstrap 95% interval (10,000 resamples of the 64 problems, `numpy.random.default_rng(2026)`).
2. **Reference gains.** For the same run: Δ_JU and Δ_JR = in-run pass@1 at round 128 − round 0 (n 8; caps 12,288 vs 8,192),
   with the same bootstrap over their problems; also the cap-matched view Δ_JU,8k = success within 8,192 tokens at round 128
   − round-0 pass@1 (both effectively at cap 8,192).
3. **Ratio.** ρ = Δ_control / Δ_JU per run, and per arm on the seed mean.
4. Arm contrasts (H − R, H − U, R − U) on control, per seed. Descriptive only: two seeds cannot estimate run variance.

## 6. Decision rule (fixed now)
- **General transfer supported** if, in each arm, both seeds have Δ_control > 0 and at least one seed's interval excludes 0,
  and the seed-mean ρ ≥ 0.5.
- **Specific transfer supported** if, in each arm, the seed-mean Δ_control ≤ 0.02 (2 points) or both seeds' intervals include 0,
  and the seed-mean ρ < 0.25.
- Anything else is **mixed**; report per arm, without forcing a verdict.
- E1 vs E2 (only when general transfer is supported): if the base → endpoint change on control is mostly a drop in truncation,
  no-box or multi-box failures (≥ 50% of the gain), call it format/termination (E1); if it is mostly fewer single-box wrong
  answers among finished attempts, call it carefulness (E2).

## 7. Known limits
- The control subjects differ from J_U/J_R in base difficulty; a ceiling or floor can shrink gains. Base pass@1 is reported next
  to each gain.
- J_U/J_R gains come from in-run evaluations (different jobs, round-0 cap 8,192). This study does not re-evaluate them.
- Two training seeds per arm. Intervals cover problem and attempt sampling only, not training-run variation.
- Absence from study sets does not show absence from pretraining data (MATH test is public).
