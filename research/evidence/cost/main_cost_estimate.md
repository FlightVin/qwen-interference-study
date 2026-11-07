# Main-run cost estimate (§7 freeze rule) — prepared 2026-11-07 17:10Z, before selection

Purpose: decide the endpoint (128 rounds unless a run is projected above 24 active GPU-h → 64 for all).

## Inputs (measured)
- Base response length (stage 3, D52): H_train 2,199 tokens; E_related 1,249; E_unrelated 1,175. Calibration items 1,437–1,535 at round 1.
- Calibration round time per mean token: 0.10–0.12 s on L40S, 0.155 s on L40 (lr1e-6/sym ran on an L40).
- Length growth over 32 rounds: lr 1e-6 sym +3–18%; lr 2e-6 sym +43% (last 8 rounds vs round 1).
- Evaluation: 5 dev evaluations of 904 samples; 450–870 s each at calibration lengths.

## Mean response length per round (16 prompts × 8)
H arm = H_train only: 2,199. R/U arms = 8 hard + 8 auxiliary: (2,199 + 1,212) / 2 ≈ 1,706.

## Projection per 128-round run (worst node 0.155 s/token; checkpoint overhead +5%)
| selected lr | growth assumed over 128 rounds | H arm GPU-h | R/U arm GPU-h | ≤ 24? |
|---|---|---|---|---|
| 1e-6 | ×1.15 | 14.3 | 11.2 | yes |
| 2e-6 | ×1.5 | 18.6 | 14.6 | yes |
| 2e-6 | ×2.0 (pessimistic) | 24.6 | 19.2 | H arm marginal |

On L40S nodes the figures are ~30% lower. Total for 15 runs (3 H + 12 R/U): lr 1e-6 ≈ 177 GPU-h; lr 2e-6 ≈ 230 GPU-h (×1.5).
Wall on 2 general GPUs: ~4–5 days; less if preempt starts.

## Decision rule application (to be confirmed at freeze with the selected lr)
- Selected 1e-6 → 128 rounds.
- Selected 2e-6 → 128 rounds at the central (×1.5) projection; the pessimistic ×2.0 case is not the planning estimate and would bind only the H arm marginally.
Recipe wall limit raised from 24 h to 40 h (general max 2 days) so a slow-node run cannot time out; preemption requeue is unchanged.
