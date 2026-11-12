# Render inspection of report v4 additions (A27)

Purpose: record the visual check of the two new figures. Date: Thu 12 Nov 2026, ~11:35 AM EST. Inspector: orchestrator.

Method: each PNG was placed at GitHub's Markdown content width (830 px at a 1024 px viewport; 372 px at a 420 px
viewport) in a local HTML page and captured with headless Chrome. The report was not sent to an external renderer.
Figures 2–8 and 10–16 are the v3 files unchanged; their render record is `../v3/render-inspection.md` (14/14 pass).

| Figure | File | 830 px | 372 px | Fixes made |
|---|---|---|---|---|
| 1 study design | `f00_design.png` | pass | pass after fix (text was ~9 px; the evaluation line overflowed its box) | larger text, shorter labels, three-line evaluation box, separated comparison labels, removed the overlapping top caption |
| 9 groups without reward contrast | `f15_signal.png` | pass | pass | lower panel axis capped at 40% (was 0–100, mostly empty) |

Value check: the per-quarter counts in `../../tables/v4/t10_signal.md` sum to the per-run totals of the v3 group table
(§5.4.3) in all 12 run × pool rows; the in-text shares (78–92%, 44–57%, 8–22%) were recomputed from those counts.
