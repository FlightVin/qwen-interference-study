# Stress-panel selection rubric v1 (A17 §5) — frozen before any candidate is selected

Purpose: define Harder-128 and SuperHard-128 by solution structure only. No model output, screening count or
difficulty number is used to select, order or exclude an item. Written 2026-11-11 ~02:05Z, before candidate export.

## 1. Eligibility (both tiers)
1. Single final answer that the frozen verifier (`qis.verify.gold_eligible`) accepts: integer, rational, radical,
   expression, or a short set/list. Exclude intervals, proofs, "show that", multi-part questions, figures needed to read
   the problem, and answers with units or words.
2. A reference solution exists in the source, or the agent annotator writes one; the factoring step is quoted from it.
3. Factoring is **central**: the reference solution needs a substantive factorization (technique codes F1–F8 of
   `research/RUBRIC.md` v1.3), and removing that step leaves no short route. A valid alternative method may exist; it is
   recorded in `alt_route` and does not exclude the item.
4. **No explicit cue.** The statement must not say "factor", "factorise", "factorization", "factored form", or name the
   technique. The item is not a direct factoring exercise (R-pool style: "Factor …", "Solve the quadratic …").
5. **Not used anywhere in the study**: not in H_train, E_related, E_unrelated, calibration panels, target_dev,
   target_test (sealed), the three probes, the cue panel, the A6 panel, or any screened candidate list. Checks: exact
   ID; exact text; normalized text (lowercase, whitespace and LaTeX spacing removed, numbers kept); word-3-gram Jaccard
   ≥ 0.5 with numbers masked (close variant); same solution template (same construction with changed constants) = close
   variant. A close variant of any used item is excluded. A close variant pair inside the candidate pool may enter one
   panel only (lower selection rank wins).

## 2. Tiers
- **Harder.** Factoring must be *recognized* inside a longer solution (≥ 3 distinct reasoning steps, the factoring step
  not the first line of an obvious template). Example structures: build an equation from a condition, then factor it;
  factor to reach a Diophantine or divisibility conclusion; factor a rational function to find domain/asymptote and use
  it further.
- **SuperHard.** Harder, plus at least one documented **obstacle** from this list (recorded per item, with a quote or
  one-sentence description):
  - `hidden_substitution`: the factorable structure appears only after a substitution or change of variable
    (e.g. y = x + 1/x, t = 2^x, symmetric sums);
  - `approach_choice`: an obvious approach stalls or explodes; the factoring route must be chosen over it;
  - `case_analysis`: after factoring, a case split (sign, parity, divisor pairs, integrality, extraneous roots) decides
    the answer;
  - `nonobvious_grouping`: the factorization needs adding/subtracting a term, completing a product (SFFT with a
    nontrivial multiplier), or Sophie Germain-type identities;
  - `multi_factorization`: two or more separate factorizations in sequence.
  Long statements, large coefficients or heavy arithmetic alone do not qualify.
- An item qualifying for SuperHard enters SuperHard only (it is not also eligible for Harder).

## 3. Per-item record (agent annotation, labelled `annotator: "agent:<model>"`)
`id` (source-native), `source_collection`, `source_ref` (dataset + revision + split/index or URL), `problem`, `gold`,
`answer_type`, `technique` (primary F-code; secondary list), `factoring_quote`, `required_reasoning` (one sentence),
`tier_proposed`, `obstacle` (SuperHard: code + note), `alt_route`, `cue_free` (yes/no), `confidence` (high/medium/low),
`family_id` (template family; items sharing a construction share it).

## 4. Sources and balance
- At least three identifiable source collections per panel, chosen from: MATH (EleutherAI/hendrycks_math, pinned),
  Omni-MATH (KbsdJames/Omni-MATH), OlympiadBench (English, text-only, open-ended math), AIME/AMC problem sets with
  published solutions.
- Overlap requirement: each source contributing ≥ 10 items to one panel should contribute ≥ 10 items to the other when
  the pool allows; technique mix reported per panel. Actual composition is published, whatever it is.
- Per-family cap: ≤ 2 items per `family_id` per panel; a family appears in one panel only.

## 5. Selection order (deterministic)
Within each tier, eligible items with confidence ≠ low are ordered by `sha256("stress-v1:" + tier + ":" + id)`.
Accept in that order subject to the family cap, the "one panel per family" rule and the per-source balance target
(no source > 50% of a panel while other sources still have eligible items). Stop at 128. If fewer than 128 qualify,
report the count and the binding reason.

## 6. Freeze and measurement
Freeze manifests (`data/stress/harder_v1.json`, `data/stress/superhard_v1.json`), family IDs, answers, this rubric and
SHA-256 hashes in a DECISIONS entry **before** any trained checkpoint is evaluated on them. Base-model difficulty is then
measured with fresh attempts (the A17 common recipe: n = 8, cap 12,288, frozen verifier). No item is removed after the
freeze because of a model outcome. Overlapping tiers or a floor effect are reported as results. Absence from every study
set does not show absence from pretraining data.
