# A26 pass A — data inventory (agent a-data)

Purpose: a verified inventory of every QIS data set, its overlaps, prompts and grading (A26 §2 "Data"). Written 2026-11-11
(EST). Terms follow `research/DEFINITIONS.md` v1. Labels: **verified** = I read primary evidence; **claimed** = only a
document says so; **unknown**. Unrounded numbers are in `data/a-data-*.json` next to this file.

**Evidence and tools.** Compute reads ran on CPU standby 11122432 (read-only on `$P = /data/user_data/vineethb/qis`;
outputs under `$P/audit-a26/a-data/`). `W` = pinned worktree `~/ae-qis/code/qis-a26-ckpt-k0-9b2fd05` (repo `data/`
is unchanged since `577aa9f`; laptop and `W` copies hash equal). Scripts in `tools/a-data/`:
`inventory_overlap.py` (stdlib; sets, hashes, overlap) → `a-data-inventory_overlap.json`; `render_grade.py` (project
code: `prompts.render`, `verify.Verifier`, `seeds`, `eval_extend.select`) → `a-data-render_grade.json`;
`labels_screening.py` (laptop, stdlib, repo files) → `a-data-labels_screening.json`; `pair_texts.py` (prints flagged
pairs); `schedule_audit.py` → `a-data-schedule_audit.json` (see `schedules.md`).

## 1. Set manifest (verified unless marked)

Family rule: H_train family = v1.3 primary operation (`tags_v13.jsonl` for MATH, `omni_pass2.jsonl` `primary` for
Omni; `select_hard_k8.py:26`); 46/46 bundle families equal the rule. Authored items carry `template_family`.
Stress families = union of masked-3-gram Jaccard ≥ 0.5 or same `family_hint` (`stress_panels.py:162-178`).
"MATH dup groups" = groups of `splits.py` (5-shingle Jaccard ≥ 0.7 via MinHash LSH).

| Set | Purpose | Source / split | Distinct | Families / groups | Natural / authored | Inclusion rule (code) | Use |
|---|---|---|---|---|---|---|---|
| H_train | hard training pool | MATH train 25 (int. algebra 23, algebra 2; L5 20, L4 4, L3 1; split `construction` 25/25) + Omni-MATH 21 | 46 | F1 11, F2 7, F3 11, F4 10, F8 7 | natural 46 | `select_hard_k8.py` + `select_hard.select`; bundle order = sorted IDs (`bundle_main.py:37`) | **training** (shared slots all arms; extra slots H); diag update prompts |
| R pool | related practice | MATH train 9 (`construction`; algebra 6, int. alg. 3; L3 4, L4 3, L5 2) + authored 54 (R1 2, R2 7, R3 9, R4 19, R5 17) | 63 | 41 template families; primary F1 17, F2 6, F3 11, F4 19, F8 10 | 9 / 54 | `match_aux.py` (12–24/32, within provenance); order = `match_k8.json` | **training** (R slots 8–15); diag |
| U pool | unrelated practice | MATH train 9 (`construction`; number theory; L2 3, L5 6) + authored 54 (U1 10, U2 11, U3 33) | 63 | 36 template families | 9 / 54 | `match_aux.py`; `unrelated[k]` paired with `related[k]` | **training** (U slots 8–15); diag |
| 63 pairs | matched R↔U | pairs from `match_k8.json` | 63 | 54 authored↔authored, 9 natural↔natural | — | 53 exact c matches; \|ΔC\| 1: 6, 2: 2, 3: 2; 1 pair dropped in balance (`syn-R4-040`/`syn-U3-061`) | training schedule |
| calib_train | lr / clip selection | MATH train, split `calib_train` (alg 14, int. alg 31, NT 14) | 59 | 59 groups | natural | `calib_panels.py` (1–5/16 low, 6–12/16 high at 8,192) | **selection** (8 `qis-cal8k-*` runs) |
| calib_val | lr / clip selection | MATH train, split `calib_val` (alg 8, int. alg 30, NT 18) | 56 | 56 groups | natural | same | **selection** |
| dev (target_dev) | in-run held-out target | MATH train, split `study_dev` 53/53 (alg 15, int. alg 38; L4 17, L5 36) | 53 | 53 groups | natural | `select_targets.py:44` (central, dp=no, L4–5, conf not low, no figure); 63 candidates → 53 | eval in-run, rounds 0/32/64/96/128, n 8 (claimed); also used for cap decisions (A11 dev16k, D81–D85) |
| J_H | hard probe | = dev[0:32], same order, text and gold (32/32) | 32 | 32 | natural | `select_targets.py:56` | derived from dev rows in-run; diag n 16 |
| J_R | practised-skill probe | MATH train `study_dev` (alg 17, int. alg 11; **L1 1, L2 4, L3 9**, L4 10, L5 4) | 28 | 28 | natural | `select_targets.py:50-51`; 31 candidates → 28 | eval in-run n 8; diag n 16 |
| J_U | unrelated-skill probe | MATH train `study_dev`, number theory (**L1 4, L2 5, L3 7**, L4 8, L5 8) | 32 | 32 | natural | `select_targets.py:52-53`; 47 candidates → first 32 | eval in-run n 8; diag n 16 |
| sealed test | primary endpoint | MATH official test, split `test` (int. alg 85, alg 43; L4 51, L5 77) | 128 | 128 groups (1 item in a 2-member group) | natural | `select_targets.py:46`; 209 candidates, first 128 eligible | eval only: base, 6 endpoints, H-s202 r116 fallback, 4 cgpu reruns, length-test (11 `summary.json`, n 32, cap 12,288, label `test-final`) |
| SuperHard | stress tier | MATH 60 (train 40, test 20; splits: construction 25, test 20, calib_val 6, calib_train 3, study_dev 4, excluded 2), Omni 52, OlympiadBench 14 | 126 | 126 stress families | natural | `stress_panels.py` (annotation tier; rubric v1) | eval only: 9 policies, n 8, cap 12,288, `stress-v1` |
| Harder | stress tier | MATH 69 (train 35, test 34; construction 19, test 34, calib_val 7, calib_train 5, study_dev 4), Omni 15, OlympiadBench 11 | 95 | 95 stress families | natural | same | eval only |
| cue | exploratory cue study | MATH official test (alg 10, int. alg 14; L4 5, L5 19) | 24 | 24 groups (1 in a 2-member group) | natural | rule in `cue_v1_*.json` meta; cued = uncued + cue sentence (24/24) | eval only: 5 policies × 2 forms, n 8 |
| response-audit set | failure reading | 8 SuperHard + 8 Harder IDs (`audit_sample_v1.json`) | 16 | — | natural | sha256(`audit-v1:`+id) order (D144) | eval reading only; 16/16 in stress, tier matches 16/16 |
| local-diag panels | one-step update test | update prompts: 4 per draw from H_train (`diag/hard`) and from pairs (`diag/pairs`), seeds 4101–4104; eval panels J_H, J_R, J_U | 12 hard, 16 pair indices | — | 9 natural pairs share | `diag_local.py:53-58` | diag r0, r64 only. Recomputed draws equal the recorded ones 24/24 |
| length tranche 1 / rest | length extension | stress items: tranche 1 = 32 per tier × attempts 0–3 (`eval_extend.select`); rest = the other 1,512 attempts | 64 / 221 | — | natural | `eval_extend.py:25-41` | t1 = planned set in 5/5 policies; rest disjoint, t1 ∪ rest = 221 × 8 |
| screening pools | difficulty screens (base model only) | `candidates-v1-study` hard 154, hard_reserve 30, related 110, unrelated 152; `omni-v1/omni_hard` 195; authored v1 198 + 196, v2 296 + 118; re-screen sets h_k8 107, related_k8 302, unrelated_k8 239 | 1,449 union (training-side) | — | mixed | `candidates.py`, `rescreen_set.py` | selection only (never trained) |

## 2. Labels and baseline difficulty (verified from `data/annotations`, `data/screening`)

Final label tuple = (label, direct_practice, modular); Omni uses proof_like in place of modular.

| Set | Annotation source (pass agreement) | Final label | Baseline screen (cap 8,192) | c range, mean | Job (claimed, ledger) |
|---|---|---|---|---|---|
| H_train | MATH: pass1 = pass2 (`pass2_hard`) 25/25; Omni: pass1 = pass2 21/21 | central / no | c/32, band core 31 (2–8), moderate 15 (9–12); 0 unknown; 2 truncated attempts | 2–12, 6.63 | `qis-rescreen-k8-h` 11074495 |
| R pool | authored pass1 = pass2 52, third read 2; natural pass1 = pass2 9 | central / yes | c/32 | 12–24, 19.00 | `qis-rescreen-k8-aux` 11074499 → 11074744 |
| U pool | authored pass1 = pass2 54; natural 9 | absent / no / modular yes | c/32 | 12–24, 18.90 | same |
| calib_train, calib_val | none (any focal label, by design) | — | c/16 (`calib16k8`) | 1–12, 6.63 / 6.91 | phase `calib` of 11074495 |
| dev, J_H, J_R, J_U | pass1 = pass2 (`pass2_targets_probes`) 53, 32, 28, 32 | dev/J_H central/no; J_R central/yes; J_U absent/no/yes | none (0 IDs in any screening count file) | — | — |
| sealed test | pass1 = pass2 127, third read 1 | central / no | none | — | — |
| cue | **pass 1 only 21/24**; pass1 = pass2 3 | central / no | none | — | — |
| stress | one agent read (`annotations/batch_*`); blind second read of 219/221 (2 SuperHard unread): tier agrees SuperHard 105/124, Harder 66/95; second-read eligible 207/219; gold agrees 219/219 | tier | none | — | stress build job not in ledger (unknown) |

Exclusions (verified from selection files): H_train eligible 46 → selected 46; excluded 61 (outside bands 60, verifier
unknown 1). Pairs: eligible R 77, U 82 → 63 pairs. SMD after repair: success 0.025, response length 0.190.

## 3. Fresh sha256 and where each hash appears (computed 2026-11-11; full list in `a-data-inventory_overlap.json`)

| File | sha256 (fresh) | Consumers that carry or check this hash |
|---|---|---|
| `$P/data/main-v1/bundle_main_v1.json` | `0e255243d8cc…8c51` | `bundle_sha` in all 11 config rows of the 6 canonical runs and the 3 a1 runs; `sha256sum -c` in 21 current recipes (main, diag, a6, deferred s303/sens); `manifest.json` records the same value |
| `$P/data/main-v1/target_test_sealed.json` | `2ffe70ddd5f8…29ca` | 28 recipes (test, fallback and H-s202 shards, cgpu, lentest); `manifest.json` records the same |
| `$P/data/stress/stress_v1.json` (= repo copy) | `757393bef80d…405d` | 17 recipes (`qis-stress-*`, `qis-len-*`); report v2 line 954 |
| `superhard_v1.json` / `harder_v1.json` | `0cf1cf93deb3…` / `0fb3490b38c5…` | report v2 line 703 only |
| `cue_v1_uncued.json` / `cue_v1_cued.json` | `c3399a42dbd3…` / `1b3aa7bf6ff6…` | `qis-cue-eval.sh` precheck |
| `$P/data/calib-panels-k8/bundle.json` | `a90a0d7143b1…` | `bundle_sha` of 8 `qis-cal8k-*` runs (the 2 `qis-cal-*` runs used superseded `calib-panels-v1`, `f7f34a1db6b9…`) |
| `$P/data/splits-v1/manifest.jsonl` | `6580712d65d4…` | no consumer checks it |
| `data/main_v1_manifest.json` (= `$P` copy) | `6d9e2f4e56a5…` | — |
| `data/panels/panels.json` (= `$P/data/targets-v1`) | `2ce7e35fc725…` | — |

Evaluation `summary.json` files record the checkpoint hash, not the panel hash; the panel hash is checked only in the
recipe precheck (job logs not read: the check's pass is **claimed**). Recipes are the current workspace text, not the
ledger `expsha` versions (unknown whether earlier versions held the same hashes). A new recipe family
`qis-control-*` checks an unlisted file `fc2ceb70ef92…` (outside this inventory).

## 4. Overlaps (verified; matrices in `a-data-inventory_overlap.json` → `pairs`, `within`, `extended`)

Rules: **J5** = `splits.py` 5-word shingles, near-duplicate if ≥ 0.7; **J3m** = `stress_panels.py` masked word 3-grams,
close variant if ≥ 0.5. Exact Jaccard over all pairs (no LSH). Normalised text = `stress_panels.norm` and a simpler
lowercase / LaTeX-spacing / whitespace normaliser.

**Intended.** J_H = dev[0:32] in order (32/32 IDs, text, gold). Audit set ⊂ stress (16/16). Diag draws ⊂ H_train and
pairs. Tranche 1 ⊂ stress. J_R, J_U and dev are disjoint by ID (0/0/0) and all three are `study_dev`; no training item is
in `study_dev` (H, R, U natural items are all `construction`).

**Unintended, by ID or text.** Across the 13 study sets (78 pairs): 0 shared IDs, 0 exact texts, 0 normalised texts,
except the intended J_H/dev. Within sets: 0 duplicate normalised texts. Max J5 between any two different sets: 0.692
(dev/J_H `algebra/train/01571` ↔ test `intermediate_algebra/test/00783`). **No pair reaches the splits rule (J5 ≥ 0.7).**

**Close variants (J3m ≥ 0.5) between sets.** These are same-template problems with changed numbers or expressions. The
splits rule cannot see them because numbers change the 5-shingles.

| Set pair | max J3m | Pairs ≥ 0.5 | Items (J3m) | Reading |
|---|---|---|---|---|
| **H_train ↔ dev** | 1.000 | 1 | `algebra/train/01269` (H) ↔ `algebra/train/01291` (dev #42, not in J_H) | same problem, new constants: "ab − 3a + 4b = 137" vs "ab − 6a + 5b = 373", min \|a − b\| |
| **H_train ↔ sealed test** | 0.725 | 2 | `int_alg/train/00301` ↔ `int_alg/test/00383` (test #30) (0.725); `int_alg/train/00048` ↔ `int_alg/test/00891` (0.500) | 00301/00383: same set-up (x + y = c, maximise a symmetric x⁴y … xy⁴ sum), different expression; 00048/00891: shared phrasing only |
| dev ↔ test | 0.773 | 5 (2 in J_H) | e.g. `algebra/train/01083` ↔ `algebra/test/00353` (ball height template) | evaluation-only sets |
| R ↔ calib_train / calib_val | 1.000 / 0.733 | 1 / 1 | `algebra/train/00769` ↔ `algebra/train/01234` | calibration only |
| dev ↔ cue; test ↔ cue | 0.667; 0.579 | 1; 1 | polynomial-interpolation template | below the cue rule (masked ≥ 0.7) |
| every other pair | ≤ 0.448 | 0 | — | stress vs every study set ≤ 0.448 (rule held) |

Within sets (J3m ≥ 0.5): R 48 pairs, U 90 (authored templates, expected), J_R 3 (one at 1.0: `algebra/train/01466` ↔
`00103`), J_U 3, sealed test 5 (max 0.906, `int_alg/test/00150` ↔ `00376`), calib_train 1. H_train 0 (max 0.331), dev 0.

**Extended universes.** Training pools (172) vs all 5,000 MATH test items: J5 max 0.630, no near-duplicate; 51 pairs
J3m ≥ 0.5, only 2 with sealed-test items (above). Omni items used (88: H_train 21, stress 67) vs all 12,500 MATH: J5
max 0.304, J3m max 0.476 — none. **Omni-MATH contains verbatim MATH problems**: `omni/00486` = H_train
`int_alg/train/00481`, `omni/01374` = H_train `int_alg/train/00656`, `omni/00518` = dev `int_alg/train/01262`,
`omni/01306` = test `int_alg/test/00620`, `omni/01717` ≈ calib_train (J5 1.0 or J3m 1.0). None of these Omni items is in
any study set (checked), so no cross-set duplicate results. Screening pools vs evaluation sets: 0 shared IDs or texts;
J5 max 0.524; J3m ≥ 0.5 with test 18, J_U 52 (modular templates), J_R 6, dev 2 (the H_train item above). Screening
exposed only the base model.

## 5. Prompt, gold and grading (verified on compute with project code)

Every prompt = Qwen3 chat template, non-thinking (`prompts.py:5-14`): user turn = `"Solve the following problem. Show
your working and put your final answer in \boxed{}.\n\n" + problem`, then `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
Rendered example (H_train `algebra/train/00288`, gold `(-11, -10)`):
`<|im_start|>user\nSolve the following problem. Show your working and put your final answer in \boxed{}.\n\nIf $y<0$,
find the range … =110$.  Express your answer using interval notation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
The cued cue form appends `\n\nConsider whether a factorization could simplify the problem.` to the problem. First item
and gold per set (full renders in JSON): R `syn-R1-017` 2; U `syn-U2-098` 52; dev `int_alg/train/01148` 15/2; J_R
`int_alg/train/01272` 2 ± √2; J_U `number_theory/train/00809` 2; test `int_alg/test/00874` −3(x+2)(x−1); calib_train
`int_alg/train/01059` 18/5; stress `omni:omni/01649` −9; cue `int_alg/test/00474` 10. Prompt tokens: max 345 (cue
cued), 0 items over the 1,024 cap in any set. MATH texts and golds equal the splits manifest in every set (e.g. 128/128
test); Omni texts equal Omni-MATH test lines (21/21 H, 67/67 stress).

**Grading** (`verify.py:113-145`): take the last `\boxed{}`/`\fbox{}` (`math_source.last_boxed`); none → `no_answer`
(or `truncated` if capped), reward 0; else Math-Verify `verify(parse(gold), parse(pred))` in a worker; capped → reward 0
whatever matches; timeout twice → `unknown` (training holds the round). Round 1 (`train_r0000`, seed 101) statuses:
H shared slots correct 11 / wrong 53; R practice 36 / 28; U practice 38 / 26 — no `no_answer` or `truncated` in these
roles in round 1, so all rejected examples are `wrong`. Re-grading the 18 examples with `Verifier` agreed 18/18.

| Role | Problem | Gold | Extracted | Status | Why |
|---|---|---|---|---|---|
| hard | `omni/02316` s0 | 29800 | 29800 | correct | last box = gold (an earlier box held "100 + 29700 = 29800"; only the last counts) |
| hard | `int_alg/train/00301` s1 | 400/11 | \frac{4400}{121} | correct | Math-Verify equates 4400/121 = 400/11 |
| hard | `int_alg/train/00076` s2 | 8,12 | 8, 12 | correct | parsed as the set {8, 12} |
| hard | `omni/02316` s1 | 29800 | 100 | wrong | wrong value |
| hard | `int_alg/train/00048` s0 | 0, 1/14, −1/12 | 0 | wrong | one of three solutions |
| hard | `int_alg/train/00301` s0 | 400/11 | 70 | wrong | wrong value |
| related | `syn-R1-013` s8 | 5 | 5 | correct | nested `\boxed{a+b=3+2=\boxed{5}}`: the last `\boxed{` start is the inner box |
| related | `syn-R4-064` s9 | 665/216 | \frac{665}{216} | correct | exact |
| related | `syn-R5-037` s10 | 3/2 + 2√5 | \frac{3}{2} + 2\sqrt{5} | correct | symbolic equality |
| related | `syn-R1-013` s8 | 5 | 6 | wrong | wrong value |
| related | `syn-R4-064` s9 | 665/216 | \frac{27}{8} | wrong | wrong value |
| related | `syn-R5-037` s10 | 3/2 + 2√5 | 5.39 | wrong | decimal approximation ≠ exact gold |
| unrelated | `syn-U2-012` s8 | 71 | 71 | correct | exact |
| unrelated | `syn-U1-032` s9 | 6 | 6 | correct | last box (a later sentence repeats it) |
| unrelated | `syn-U3-071` s10 | 208 | 208 | correct | exact |
| unrelated | `syn-U2-012` s8 | 71 | 67 | wrong | wrong value |
| unrelated | `syn-U1-032` s9 | 6 | 11 | wrong | wrong value |
| unrelated | `syn-U3-071` s10 | 208 | 149 | wrong | wrong value |

## 6. Disagreements with DEFINITIONS v1 and report v2

1. **Held-out sets contain a numeric variant of a training problem (new; interpretation).** dev #42
   `algebra/train/01291` is H_train `algebra/train/01269` with new constants (J3m 1.0). Sealed test #30
   `int_alg/test/00383` is a close variant of H_train `int_alg/train/00301` (J3m 0.725). DEFINITIONS §3.A "dev"
   ("after removing near-duplicates of test") and report v2 ("held out") are true only under the 5-shingle rule
   (J5 < 0.7); the project's own stress rule would flag both. Effect bound: 1/53 dev, 1/128 test (2/128 at J3m ≥ 0.5).
2. **Report v2 line 207/302 "J_R, J_U from study_dev, which no training pool draws from"**: verified. But J_R and J_U are
   not L4–5 (J_R 14/28 and J_U 16/32 are L1–3); no document states their levels.
3. **Cue labels**: 21/24 cue items have only a pass-1 label (allowed by the cue rule, flagged in its meta). DEFINITIONS
   §3.A "cue study" does not mention it.
4. **Stress MATH items come from study splits**: 44 `construction`, 8 `study_dev`, 21 `calib_*`, 2 `excluded`
   (near-duplicates of MATH test items), 54 `test`. Report v2 §7.1 says "unused problems"; true by ID and by the J3m rule
   (max 0.448 vs every study set). Not an error; the split mix is undocumented.
5. **Omni-MATH is not disjoint from MATH**: at least 5 Omni test items are verbatim MATH items (above). The H_train and stress
   Omni items are not among them (max J3m 0.476 vs all MATH). DEFINITIONS §5 item 8 is now closed: no near-duplicate
   within H_train (max J5 0.237, J3m 0.331) and none between Omni items and dev/test.
6. All other counts in DEFINITIONS §2 and §3.A (46, 63, 9 + 54, 53 exact, 53, 32, 28, 32, 128 = 85 + 43, 59, 56, 221 =
   126 + 95 with source splits, 24, 16, second read 171/219) agree with the files.

## 7. Unverifiable or not checked

- The stress build job (`qis-stress-select`) and the cue/audit selection runs are not in the ledger: unknown which job
  wrote `$P/data/stress/` (its `selection_report.json` hash `5cc7e781…` is recorded only here).
- Recipe prechecks (`sha256sum -c`) passing in each job: claimed (logs not read). Old recipe versions: unknown.
- dev evaluation rounds 0/32/64/96/128: claimed from DEFINITIONS; eval row counts (20 per single-segment run = 5 rounds ×
  4 panels) are consistent.
- Calibration candidate pools (200 + 200) and the 4,096-cap screens were not re-checked.
