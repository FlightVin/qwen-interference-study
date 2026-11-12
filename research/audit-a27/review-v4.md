# A27 independent review of report v4 additions

Purpose: record the independent check of the text that v4 added to v3. Reviewer: a fresh agent with no access to the
orchestrator's reasoning; read-only; own recomputation of the per-quarter group table from
`research/audit-a26/pass-b/data/train_groups.json` (48/48 cells match `reports/tables/v4/t10_signal.md`).
Date: Thu 12 Nov 2026, ~10:58 AM EST. Every v4 number not listed below was checked against its source and verified.

| # | Location | Finding | Verdict | Fix applied |
|---|---|---|---|---|
| 1 | App. I, genuine failure | said q(3) = −30 fails; it holds (5·2·(−3) = −30); the broken condition is "no horizontal asymptote" | wrong | rewritten |
| 2 | App. I, same row | "finished" not in the trace | unsupported | removed |
| 3 | §6, limits of transfer | "cap is next most frequent"; failed-after-factoring (17) exceeds cap (11) | wrong | rewritten with counts |
| 4 | §5.4.3, §6, App. J | "tested mainly in rounds 1–64", "divisor" clause overstated | overstated | replaced by contrast shares per quarter |
| 5 | §5.4.3 | saturation jump in rounds 33–64 coincides with the cap increase | missing confound | added |
| 6 | §5.4.3, §5.4.4 | all-wrong groups "common/frequent" | overstated | percentages given |
| 7 | §5.4.3 | "mostly ... plus momentum" imprecise | imprecise | direct gradient only from mixed groups |
| 8 | §5.4.4 | POPE related easy "slightly accelerated" | partly supported | compared with unrelated; one hard problem noted |
| 9 | §6 | grader-limitation reads ambiguous | ambiguous | "both reads of 2, one read of 2 others" |
| 10 | §6 | execution "least frequent" | qualify | "of the four failure categories" |
| 11 | §5.4.4 | Schaul ingredient (b) wording | loose | "learning progress coupled to current performance" |

Orchestrator check after the fixes: H's extra hard groups with contrast in rounds 65–128 are 102–139 of 256 (40–54%);
a first fix draft said 43–56% and was corrected before commit.
