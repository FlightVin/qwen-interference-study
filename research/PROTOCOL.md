# PROTOCOL (draft — not frozen)

Purpose: the frozen experimental design. Until the freeze line below is signed with a commit, this
file is a working draft. The canonical instruction remains root `project.md` (§1–§18).

## Status
- Draft. Frozen items so far: response cap 4,096 (D18); decoding T=1, top_p=1, top_k disabled
  (vLLM default 0), repetition penalty 1, native sampler (D14); model Qwen3-1.7B @ `70d244cc`;
  MATH @ `21a56338`; rubric v1.1 (D16).

## Amendments in force
- A1 (§15) precision audit + report figures; A2 (§16) auxiliary pool target and supplementation;
  A3 (§17) 30 CPUs; A4 (§18) H_train construction: full native stage 2 over 184 candidates, core
  band 2–8/32 and moderate band 9–12/32, target 64 (≥ half core), proceed at 48–64, external
  Omni-MATH tranche (≤200) or authored tranche or smaller pilot with uniform endpoint shortening,
  one frozen H_train manifest for all arms, H-only extra prompts balanced by source/difficulty.

## To freeze (pending)
H_train, E_related/E_unrelated pairs, target_dev, target_test, probe panels, calibration result,
learning rate and clipping, endpoint rounds, seeds mapping, evaluation schedule, analysis plan.
